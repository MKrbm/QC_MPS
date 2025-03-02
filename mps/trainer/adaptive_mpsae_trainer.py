#!/usr/bin/env python3
"""
Module for training TPCP with a regularization scheduler.
This training function uses several scheduling strategies (linear, polynomial,
soft exponential, cosine, sr_base) to control the regularization coefficient (λ)
that multiplies the regularization term. The training proceeds in phases: at each phase,
the model is trained until convergence (using criteria similar to mpsae_train);
if a spike in the loss is detected, the λ update is temporarily held.
Once convergence is reached for the current λ, the scheduler increases λ and training continues.

Note: In this updated version, the target λ (lambda_final) is determined from the initial loss computed
on the first mini-batch when the MPSTPCP model is built.
"""

import time
import torch
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Import required functions and modules from the MPS package.
from mps import tpcp_mps  # contains MPSTPCP and regularize_weight
from mps.trainer import utils
from mps.trainer.utils import loss_batch, calculate_accuracy  # assumed to be defined
from mps.StiefelOptimizers import StiefelAdam
from mps.radam import RiemannianAdam
from mps.trainer.smps_trainer import smps_train


# =============================================================================
# Helper: Schedule Function
# =============================================================================
def get_scheduled_lambda(schedule_type, step, total_steps, lambda_initial, lambda_final, poly_power=2, k=0.1, srpq_target=None, current_loss=None):
    """
    Compute λ based on the scheduling strategy.

    Args:
      schedule_type (str): 'linear', 'polynomial', 'soft_exponential', 'cosine', or 'sr_base'
      step (int): current scheduling step (0 <= step <= total_steps)
      total_steps (int): total number of scheduling steps.
      lambda_initial (float): initial λ value.
      lambda_final (float): final (target) λ value.
      poly_power (float): power for polynomial schedule.
      k (float): rate parameter for soft exponential schedule.
      srpq_target (float, optional): target success rate per qubit for 'sr_base' schedule.
      current_loss (float, optional): current loss for 'sr_base' schedule.

    Returns:
      float: scheduled λ.
    """
    if schedule_type == "sr_base":
        if srpq_target is None or current_loss is None:
            raise ValueError("Must provide srpq_target and current_loss for 'sr_base' schedule.")
        neg_log_srpq = -math.log(srpq_target)
        # Set λ = loss / (-log(srpq_target)) * (1/10)
        return current_loss / max(neg_log_srpq, 1e-12) * 0.1
    elif schedule_type == "linear":
        return lambda_initial + (step / total_steps) * (lambda_final - lambda_initial)
    elif schedule_type == "polynomial":
        return lambda_initial + ((step / total_steps) ** poly_power) * (lambda_final - lambda_initial)
    elif schedule_type == "soft_exponential":
        return lambda_initial + (lambda_final - lambda_initial) * (1 - math.exp(-k * step)) / (1 - math.exp(-k * total_steps))
    elif schedule_type == "cosine":
        return lambda_initial + (lambda_final - lambda_initial) * (1 - math.cos(math.pi * step / total_steps)) / 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


# =============================================================================
# Main Training Function
# =============================================================================
def mpsae_adaptive_train(
    dataloader,
    device,
    N,
    K=1,
    chi=2,
    mps_epochs=10,
    mps_lr=0.01,
    mps_log_steps=10,
    total_schedule_steps=10,
    schedule_type="cosine",  # choose among 'linear', 'polynomial', 'soft_exponential', 'cosine', 'sr_base'
    poly_power=2,
    k=0.1,
    max_epochs=10,
    min_epochs=3,
    conv_threshold=1e-4,
    patience=2,
    conv_strategy="absolute",  # or "relative"
    spike_threshold=0.1,  # if loss increases >10% compared to previous epoch, consider it a spike
    log_steps=10,
    lr=0.0001,
    manifold="Original",
    mps_optimize="greedy",
    dtype=torch.float64,
):
    """
    Train a TPCP model while scheduling the regularization coefficient λ.

    The training proceeds in phases. In each phase a fixed λ is used; training
    continues until the convergence criteria are met (or until a max number of epochs).
    If a spike in the loss is detected, the scheduler holds λ from increasing.
    Once convergence is reached for the current λ, λ is updated according to the chosen schedule.

    Note:
      The target λ value (lambda_final) is determined using the initial loss computed
      from the first mini-batch when building the MPSTPCP model.

    Returns:
      dict: metrics containing lists for loss, accuracy, λ values, and SRPQ.
    """
    manifold_map = {
        "Exact": tpcp_mps.ManifoldType.EXACT,
        "Frobenius": tpcp_mps.ManifoldType.FROBENIUS,
        "Canonical": tpcp_mps.ManifoldType.CANONICAL,
        "Cayley": tpcp_mps.ManifoldType.EXACT,
        "MatrixExp": tpcp_mps.ManifoldType.EXACT,
        "ForwardEuler": tpcp_mps.ManifoldType.EXACT,
        "Original": tpcp_mps.ManifoldType.EXACT,
    }
    if manifold not in manifold_map:
        raise ValueError(f"Invalid manifold='{manifold}'.")

    # --- Step 1: Train SimpleMPS ---
    print("Starting SimpleMPS training...")
    smps = smps_train(
        dataloader,
        N=N,
        d=2,
        l=2,
        epochs=mps_epochs,
        lr=mps_lr,
        log_steps=mps_log_steps,
        dtype=dtype,
        device=device,
        optimize=mps_optimize,
    )
    print("Completed training for canonical SimpleMPS (smps).")

    # --- Step 2: Build and Prepare TPCP ---
    tpcp = tpcp_mps.MPSTPCP(
        N,
        K=K,
        d=2,
        enable_r=False,
        with_identity=True,
        manifold=tpcp_mps.ManifoldType.EXACT,
    )
    tpcp.to(device)
    tpcp.train()
    tpcp.set_canonical_mps(smps)
    
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    nnloss = torch.nn.NLLLoss(reduction="mean")

    # Initialize W: start with second column small (e.g., 0.05).
    W = torch.zeros(tpcp.L, 2, dtype=torch.float64, device=device)
    W[:, 0] = 1 
    W[:, 1] = 0.001
    tpcp.initialize_W(W)

    # --- Step 3: Determine lambda_final Using the Initial Loss Value ---
    data_batch, target_batch = next(iter(dataloader))
    data_batch, target_batch = data_batch.to(device), target_batch.to(device)
    with torch.no_grad():
        initial_probs, reg = tpcp(data_batch, return_probs=True, return_reg=True)
        softmax_initial_probs = logsoftmax(initial_probs)
        initial_accuracy = calculate_accuracy(initial_probs[:, 0], target_batch)
        print(f"Initial accuracy: {initial_accuracy.item():.2%}")
        initial_loss = nnloss(softmax_initial_probs, target_batch)
        avg_loss = initial_loss.item()
        initial_reg = reg
    print(f"Initial loss for λ determination: {initial_loss.item():.6f}")
    print(f"Initial regularization weight: {initial_reg.item():.6f}")
    # Update lambda_final based on the initial loss.
    lambda_final = initial_loss.item() / initial_reg.item() * 20
    print(f"Updated lambda_final set to: {lambda_final:.6f}")

    # --- Step 4: Scheduler Setup ---
    current_schedule_step = 0

    # === Generate target SRPQ list ===
    # Here we generate target success rate per qubit values (sqrt transformation applied)
    import numpy as np
    target_srpq = np.linspace((0.5**2), 1, total_schedule_steps + 1, endpoint=True) ** (1/2)
    target_srpq[-1] = 0.99
    print(f"Generated sqrt target_srpq schedule: {target_srpq}")

    # --- Step 5: Training Loop Over λ Phases ---
    metrics = {"loss": [], "accuracy": [], "lambda": [], "srpq": []}
    epoch_total = 0

    tpcp.train()

    for current_schedule_step in range(len(target_srpq)):
        # For sr_base strategy, use target_srpq for the current phase.
        if schedule_type == "sr_base":
            print(f"Target SRPQ for current phase: {target_srpq[current_schedule_step]:.6f}")
            current_lambda = get_scheduled_lambda(
                schedule_type,
                current_schedule_step,
                total_schedule_steps,
                0.05,
                lambda_final,
                poly_power,
                k,
                srpq_target=target_srpq[current_schedule_step],
                current_loss=avg_loss  # Using the avg_loss from the last epoch
            )
        else:
            current_lambda = get_scheduled_lambda(
                schedule_type,
                current_schedule_step,
                total_schedule_steps,
                0.1,
                lambda_final,
                poly_power,
                k,
            )

        print(f"\n=== Training Phase with λ = {current_lambda:.6f} (Schedule Step {current_schedule_step}/{total_schedule_steps}) ===")
        phase_epoch = 0
        conv_counter = 0
        prev_epoch_loss = None

        while phase_epoch < max_epochs:
            # Create optimizers: one for the Kraus ops and one for W and r.
            optimizer = RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr, betas=(0.9, 0.999))
            optimizer_weight = torch.optim.Adam([tpcp.W, tpcp.r], lr=lr)

            epoch_loss_sum = 0.0
            epoch_acc_sum = 0.0
            epoch_loss_with_reg_sum = 0.0
            epoch_srpq_sum = 0.0
            total_samples = 0
            t0 = time.time()

            # Loop over mini-batches.
            for step, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                optimizer_weight.zero_grad()

                outputs, reg = tpcp(data, return_probs=True, return_reg=True)
                softmax_outputs = logsoftmax(outputs)
                loss = nnloss(softmax_outputs, target)
                loss_with_reg = loss + current_lambda * reg
                loss_with_reg.backward()
                optimizer.step()
                optimizer_weight.step()
                tpcp.normalize_w_and_r()
                tpcp.proj_stiefel(check_on_manifold=True, print_log=False)
                # Compute success rate per qubit (SRPQ)
                srpq = torch.exp(-reg)

                bs = target.size(0)
                epoch_loss_sum += loss.item() * bs
                epoch_loss_with_reg_sum += loss_with_reg.item() * bs
                total_samples += bs
                batch_acc = calculate_accuracy(outputs[:, 0].detach(), target)
                epoch_acc_sum += batch_acc.item() * bs
                epoch_srpq_sum += srpq.item() * bs

                if (step + 1) % log_steps == 0 or step == 0:
                    print(
                        f"[λ {current_lambda:.6f}] Epoch {phase_epoch+1}, Step {step+1}/{len(dataloader)} | "
                        f"Reg: {reg.item():.6f} | Batch Loss: {loss.item():.6f} | "
                        f"Loss+Reg: {loss_with_reg.item():.6f} | Acc: {batch_acc.item():.2%} | "
                        f"SRPQ: {srpq.item():.6f}"
                    )

            # End of epoch: compute averages.
            avg_loss = epoch_loss_sum / total_samples
            avg_loss_with_reg = epoch_loss_with_reg_sum / total_samples
            avg_acc = epoch_acc_sum / total_samples
            avg_srpq = epoch_srpq_sum / total_samples
            elapsed = time.time() - t0
            print(
                f"[λ {current_lambda:.6f}] Epoch {phase_epoch+1} | Avg Loss: {avg_loss:.6f} | "
                f"Avg Loss+Reg: {avg_loss_with_reg:.6f} | Acc: {avg_acc:.2%} | "
                f"SRPQ: {avg_srpq:.6f} | Time: {elapsed:.2f}s"
            )

            # Record metrics.
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(avg_acc)
            metrics["lambda"].append(current_lambda)
            metrics["srpq"].append(avg_srpq)

            # --- Check for convergence ---
            if prev_epoch_loss is not None:
                # Check convergence criteria based on regularized loss.
                if phase_epoch >= min_epochs:
                    if conv_strategy == "absolute":
                        condition = abs(avg_loss_with_reg - prev_epoch_loss) < conv_threshold
                    elif conv_strategy == "relative":
                        condition = abs(avg_loss_with_reg - prev_epoch_loss) / (abs(prev_epoch_loss) + 1e-8) < conv_threshold
                    else:
                        condition = False
                    if condition:
                        conv_counter += 1
                    else:
                        conv_counter = 0
            prev_epoch_loss = avg_loss_with_reg
            phase_epoch += 1
            epoch_total += 1

            # --- For sr_base, also break if the target SRPQ is reached ---
            if schedule_type == "sr_base" and avg_srpq >= target_srpq[current_schedule_step]:
                print(f"Target SRPQ {target_srpq[current_schedule_step]:.6f} reached (current: {avg_srpq:.6f}).")
                break

            # If convergence is achieved for the current λ phase, break.
            if conv_counter >= patience:
                print(f"Convergence achieved for λ {current_lambda:.6f} after {conv_counter} stable epochs.")
                break

        # Update λ for the next phase.
        current_schedule_step += 1

        if current_schedule_step == len(target_srpq):
            break
        # inside the scheduling step loop (e.g., before entering epoch loop)

    # Optionally, plot the training metrics.
    x_axis = range(1, len(metrics["loss"]) + 1)
    # --- Save metrics to CSV ---
    # Build folder name from hyperparameters (excluding epoch-related parameters).
    folder_name1 = (
        f"schedule_{schedule_type}_steps_{total_schedule_steps}_"
        f"{manifold}"
    )
    folder_name2 = (
        f"lr_{lr}_epochs_{max_epochs}"
    )

    base_dir = Path("metrics/aemps")
    folder_path = base_dir / folder_name1 / folder_name2
    folder_path.mkdir(parents=True, exist_ok=True)

    # Use max_epochs as the epoch_size for the CSV file name.
    csv_filename = f"metrics.csv"
    csv_filepath = folder_path / csv_filename

    # Convert metrics dictionary to a DataFrame and save.
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(csv_filepath, index_label="epoch")

    utils.plot_training_metrics(
        x_axis,
        metrics["loss"],
        metrics["accuracy"],
        weight_ratio_vals=metrics.get("srpq", None),  # now using SRPQ
        lambda_vals=metrics.get("lambda", None),
        title="TPCP Training Metrics over Epochs",
        filename=(folder_path / "tpcp_training_metrics.png").resolve().as_posix(),
    )
    print(f"Metrics saved to: {csv_filepath}")

    # Save the trained model
    model_filename = f"tpcp_model_{max_epochs}.pt"
    model_filepath = folder_path / model_filename
    torch.save(tpcp.state_dict(), model_filepath)
    print(f"Model saved to: {model_filepath}")

    return metrics
