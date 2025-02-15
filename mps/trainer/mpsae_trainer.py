#!/usr/bin/env python3
"""
Module for training an MPS Adiabatic Encoder (MPSAE).
This file defines the mpsae_train() function.
"""

import time
import torch
import matplotlib.pyplot as plt
import geoopt

# Import the required models and optimizers.
from mps.simple_mps import SimpleMPS
from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps.StiefelOptimizers import StiefelAdam, StiefelSGD
from mps.radam import RiemannianAdam

# Import common utilities.
from mps.trainer.utils import loss_batch, calculate_accuracy, to_probs, plot_training_metrics

def mpsae_train(
    dataloader,
    device,
    N,
    d=2,
    l=2,
    mps_epochs=1,
    mps_lr=0.001,
    mps_optimize="greedy",
    manifold="Exact",
    optimizer_name="adam",
    K=1,
    max_epochs=10,
    min_epochs=3,
    patience=2,
    conv_strategy="absolute",
    conv_threshold=1e-4,
    lr=0.01,
    log_steps=10,
    weight_values=None,
    dtype=torch.float64,
):
    """
    Train an MPS Adiabatic Encoder.
    
    1. Trains a SimpleMPS on the provided dataloader.
    2. Initializes TPCP.
    3. Adiabatically interpolates between W = 0 and W= 1.
    
    Returns:
      dict with keys:
        - 'simple_mps_losses': list of SimpleMPS epoch losses
        - 'tpcp_metrics_by_w': dict mapping weight value to metrics (loss, accuracy, weight ratio)
    """
    if weight_values is None:
        weight_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # --- 1) Train SimpleMPS ---
    smps = SimpleMPS(N, 2, d, l, layers=1, device=device, dtype=dtype, optimize=mps_optimize)
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    nnloss = torch.nn.NLLLoss(reduction="mean")
    opt_smps = torch.optim.Adam(smps.parameters(), lr=mps_lr)
    smps_losses = []
    smps.train()
    print(f"\n=== Training SimpleMPS for {mps_epochs} epoch(s)... ===")
    for epoch in range(mps_epochs):
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            data = data.permute(1, 0, 2)  # [batch, N, 2] → [N, batch, 2]
            opt_smps.zero_grad()
            outputs = smps(data)
            outputs = logsoftmax(outputs)
            loss = nnloss(outputs, target)
            loss.backward()
            opt_smps.step()
            bs = target.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            preds = outputs.argmax(dim=-1)
            acc = (preds == target).float().sum().item()
            total_correct += acc
            print(f"[SimpleMPS] Epoch {epoch+1}, Step {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.6f} | Acc: {acc/bs:.2%}")
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        smps_losses.append(epoch_loss)
        print(f"[SimpleMPS] Epoch {epoch+1} | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.2%}")

    def get_tpcp_optimizer(parameters) -> torch.optim.Optimizer:
        if manifold in ["Cayley", "MatrixExp", "ForwardEuler"]:
            if optimizer_name.lower() == "adam":
                return StiefelAdam(parameters, lr=lr, expm_method=manifold)
            elif optimizer_name.lower() == "sgd":
                return StiefelSGD(parameters, lr=lr, expm_method=manifold)
            else:
                raise ValueError("optimizer must be 'adam' or 'sgd'")
        elif manifold in ["Exact", "Frobenius", "Canonical"]:
            if optimizer_name.lower() == "adam":
                return geoopt.optim.RiemannianAdam(parameters, lr=lr)
            elif optimizer_name.lower() == "sgd":
                return geoopt.optim.RiemannianSGD(parameters, lr=lr)
            else:
                raise ValueError("optimizer must be 'adam' or 'sgd'")
        elif manifold == "Original":
            if optimizer_name.lower() == "adam":
                return RiemannianAdam(parameters, lr=lr)
            else:
                raise ValueError("SGD not supported for Original update rule.")
        else:
            raise ValueError(f"Invalid manifold '{manifold}' or optimizer '{optimizer_name}'.")

    # --- 2) Build and Train TPCP Model ---
    manifold_map = {
        "Exact": ManifoldType.EXACT,
        "Frobenius": ManifoldType.FROBENIUS,
        "Canonical": ManifoldType.CANONICAL,
        "Cayley": ManifoldType.EXACT,
        "MatrixExp": ManifoldType.EXACT,
        "ForwardEuler": ManifoldType.EXACT,
        "Original": ManifoldType.EXACT,
    }
    if manifold not in manifold_map:
        raise ValueError(f"Invalid manifold='{manifold}'.")
    tpcp = MPSTPCP(N, K=K, d=2, with_probs=False, with_identity=True, manifold=manifold_map[manifold])

    tpcp.to(device)
    tpcp.train()
    tpcp.set_canonical_mps(smps)
    metrics = {"loss": [], "accuracy": [], "weight_rate": []}

    for w in weight_values:
        # Reset TPCP optimizer at each weight using the function
        opt_tpcp = get_tpcp_optimizer(tpcp.kraus_ops.parameters())
        weight_optimizer = torch.optim.Adam([tpcp.r], lr=lr)

        W = torch.zeros(tpcp.L, 2, dtype=dtype, device=device)
        W[:, 0] = 1
        W[:, 1] = w
        tpcp.initialize_W(W)
        print(f"\n=== Training TPCP with weight rate w={w:.1f} (max_epochs={max_epochs}) ===")
        epoch = 0
        prev_epoch_loss = None
        conv_counter = 0
        while epoch < max_epochs:
            epoch_loss_sum = 0.0
            total_samples = 0
            epoch_acc_sum = 0.0
            t0 = time.time()
            for step, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                bs = target.size(0)
                opt_tpcp.zero_grad()
                outputs = tpcp(data)
                outputs = to_probs(outputs)[:, 0]  # probability for label 0
                loss_val = loss_batch(outputs, target)
                loss_val.backward()
                opt_tpcp.step()
                weight_optimizer.step()
                tpcp.normalize_w_and_r()
                tpcp.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)
                epoch_loss_sum += loss_val.item()
                total_samples += bs
                batch_acc = calculate_accuracy(outputs.detach(), target)
                epoch_acc_sum += batch_acc.item() * bs
                if (step + 1) % log_steps == 0:
                    loss_per_data = loss_val.item() / bs
                    print(f"[TPCP::w={w:.1f}] Epoch {epoch+1}, Step {step+1}/{len(dataloader)} | Loss per Data: {loss_per_data:.6f} | Acc: {batch_acc.item():.2%}")
            if total_samples == 0:
                continue
            avg_loss = epoch_loss_sum / total_samples
            avg_acc = epoch_acc_sum / total_samples
            current_weight_rate = (tpcp.W[:, 1] / torch.sum(tpcp.W, dim=1)).mean().item() if hasattr(tpcp, "W") else w
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(avg_acc)
            metrics["weight_rate"].append(current_weight_rate)
            elapsed = time.time() - t0
            print(f"[TPCP::w={w:.1f}] Epoch {epoch+1} | Avg Loss: {avg_loss:.6f} | Acc: {avg_acc:.2%} | Weight Rate: {current_weight_rate:.4f} | Time: {elapsed:.2f}s")
            if epoch >= min_epochs:
                if prev_epoch_loss is not None:
                    if conv_strategy == "absolute":
                        condition = abs(avg_loss - prev_epoch_loss) < conv_threshold
                    elif conv_strategy == "relative":
                        condition = abs(avg_loss - prev_epoch_loss) / (abs(prev_epoch_loss) + 1e-8) < conv_threshold
                    else:
                        condition = False
                    if condition:
                        conv_counter += 1
                    else:
                        conv_counter = 0
                    if conv_counter >= patience:
                        print(f"Convergence reached after {conv_counter} consecutive epochs.")
                        break
            prev_epoch_loss = avg_loss
            epoch += 1

    x_axis = range(1, len(metrics["loss"]) + 1)
    plot_training_metrics(x_axis, metrics["loss"], metrics["accuracy"], metrics["weight_rate"],
                            "MPSAE Training Metrics over Epochs",
                            "mpsae_training_metrics.png")
    
    return {"simple_mps_losses": smps_losses, "tpcp_metrics_by_w": metrics}
