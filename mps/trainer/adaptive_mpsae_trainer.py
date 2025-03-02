#!/usr/bin/env python3
"""
Module for training TPCP with a regularization scheduler.
This training function uses several scheduling strategies (linear, polynomial,
soft exponential, cosine, sr_base) to control the regularization coefficient (λ)
that multiplies the regularization term. The training proceeds in phases: at each phase,
the model is trained until convergence (using criteria similar to mpsae_train);
if a spike in the loss is detected, the λ update is temporarily held.
Once convergence is reached for the current λ, the scheduler increases λ and training continues.

Note:
  In this updated version, the target λ (lambda_final) is determined from the initial loss computed
  on the first mini-batch when the MPSTPCP model is built.
  After training (either when the success rate SRPQ reaches 0.98 or the epochs finish),
  an extra final update training is performed in 10 phases. In each final update phase,
  we first compute for each row in tpcp.W (shape: [N_qubits-1, 2]) the index of the non-max element
  and a delta increment (so that if added over the remaining phases the non-max element becomes 1).
  Then we train for the given epoch size using loss (without regularization), record metrics,
  update the non-max entries by the computed delta, and finally plot & store the final update metrics.
"""

import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
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
      After training (either when the success rate reaches 0.98 or the epochs finish),
      an extra final update training is performed in 10 phases. In each final update phase,
      we first determine for each row in tpcp.W (with shape (N_qubits-1, 2)) the index of the non-max entry
      and compute a delta value (increment) that will push that entry to 1 linearly over the remaining phases.
      Then we train for the given epoch size using the unregularized loss, record metrics, and update tpcp.W.
      
    Returns:
      dict: metrics containing lists for loss, accuracy, λ values, and SRPQ (from the main training loop).
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

    # Initialize W: start with first column ones and second column small.
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
    # Generate target SRPQ list.
    target_srpq = np.linspace(0.5, 0.97, total_schedule_steps + 1, endpoint=True)

    print(f"Generated sqrt target_srpq schedule: {target_srpq}")

    # --- Step 5: Main Training Loop Over λ Phases ---
    metrics = {"loss": [], "accuracy": [], "lambda": [], "srpq": [], "reg": []}
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
            epoch_reg_sum = 0.0
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
                epoch_reg_sum += reg.item() * bs

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
            avg_reg = epoch_reg_sum / total_samples
            elapsed = time.time() - t0
            print(
                f"[λ {current_lambda:.6f}] Epoch {phase_epoch+1} | Avg Loss: {avg_loss:.6f} | "
                f"Avg Loss+Reg: {avg_loss_with_reg:.6f} | Acc: {avg_acc:.2%} | "
                f"SRPQ: {avg_srpq:.6f} | Reg: {avg_reg:.6f} | Time: {elapsed:.2f}s"
            )

            # Record metrics.
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(avg_acc)
            metrics["lambda"].append(current_lambda)
            metrics["srpq"].append(avg_srpq)
            metrics["reg"].append(avg_reg)

            # --- Check for convergence ---
            if prev_epoch_loss is not None:
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

            # --- For sr_base, break if the target SRPQ is reached ---
            if schedule_type == "sr_base" and avg_srpq >= target_srpq[current_schedule_step]:
                print(f"Target SRPQ {target_srpq[current_schedule_step]:.6f} reached (current: {avg_srpq:.6f}).")
                break

            # If the reg (log of srpq) reaches 1 / N, then stop it.
            if reg < 1 / N * 2:
                print(f"Reg (log of SRPQ) reached {1 / N:.6f} (current: {reg:.6f}). Ending current phase early.")
                break

            if conv_counter >= patience:
                print(f"Convergence achieved for λ {current_lambda:.6f} after {conv_counter} stable epochs.")
                break

    # --- Final Update Training: 10 Phases with Training using Loss (without regularization) ---
    tpcp.normalize_w_and_r()
    print("\n=== Starting Final Update Training Phase ===")
    final_update_phases = 10
    # Dictionary to record final update training metrics.
    final_update_metrics = {
        "update_epoch": [],
        "phase": [],
        "epoch_in_phase": [],
        "loss": [],
        "accuracy": [],
        "W_non_max_avg": [],
        "srpq": [],
        "reg": []
    }
    overall_update_epoch = 0
    # At the beginning of each phase, recalc the non-max index and delta for each row.

    non_max_idx_list = []
    delta_list = []
    assert torch.all(torch.max(tpcp.W, dim=1).values == 1), "Not all max values are ones"
    for i in range(tpcp.W.shape[0]):
        # Determine the non-max index by checking which entry is not (close to) 1.
        if tpcp.W[i, 0].item() != 1:    
            non_max_idx = 0
        else:
            non_max_idx = 1
        non_max_idx_list.append(non_max_idx)
        current_val = tpcp.W[i, non_max_idx].item()
        delta_i = (1 - current_val) / final_update_phases
        delta_list.append(delta_i)
    print(f"Max delta value: {max(delta_list):.6f}")
    
    # Set tpcp.W to be non-optimization params
    tpcp.W.requires_grad = False


    for phase in range(final_update_phases):

        # Train for the given epoch size (using loss, not loss_with_reg) in this phase.
        for epoch in range(max_epochs):
            t0 = time.time()
            epoch_loss_sum = 0.0
            epoch_acc_sum = 0.0
            epoch_srpq_sum = 0.0
            epoch_reg_sum = 0.0
            total_samples = 0

            # Create new optimizers.
            optimizer = RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr, betas=(0.9, 0.999))
            optimizer_weight = torch.optim.Adam([tpcp.r], lr=lr)

            for step, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                optimizer_weight.zero_grad()

                outputs, reg = tpcp(data, return_probs=True, return_reg=True)
                softmax_outputs = logsoftmax(outputs)
                loss = nnloss(softmax_outputs, target)
                loss.backward()
                optimizer.step()
                optimizer_weight.step()
                tpcp.normalize_w_and_r()
                tpcp.proj_stiefel(check_on_manifold=True, print_log=False)

                bs = target.size(0)
                epoch_loss_sum += loss.item() * bs
                batch_acc = calculate_accuracy(outputs[:, 0].detach(), target)
                epoch_acc_sum += batch_acc.item() * bs
                epoch_srpq_sum += torch.exp(-reg).item() * bs
                epoch_reg_sum += reg.item() * bs
                total_samples += bs

            avg_loss = epoch_loss_sum / total_samples
            avg_acc = epoch_acc_sum / total_samples
            avg_srpq = epoch_srpq_sum / total_samples
            avg_reg = epoch_reg_sum / total_samples
            elapsed = time.time() - t0

            # Compute average non-max value across rows for reporting.
            W_non_max_values = []
            for i in range(tpcp.W.shape[0]):
                W_non_max_values.append(tpcp.W[i, non_max_idx_list[i]].item())
            avg_W_non_max = sum(W_non_max_values) / len(W_non_max_values)

            print(f"Final update - Phase {phase+1}, Epoch {epoch+1}: Loss: {avg_loss:.6f}, Accuracy: {avg_acc:.2%}, Avg W non-max: {avg_W_non_max:.6f}, SRPQ: {avg_srpq:.6f}, Reg: {avg_reg:.6f}, Time: {elapsed:.2f}s")

            final_update_metrics["update_epoch"].append(overall_update_epoch + 1)
            final_update_metrics["phase"].append(phase + 1)
            final_update_metrics["epoch_in_phase"].append(epoch + 1)
            final_update_metrics["loss"].append(avg_loss)
            final_update_metrics["accuracy"].append(avg_acc)
            final_update_metrics["W_non_max_avg"].append(avg_W_non_max)
            final_update_metrics["srpq"].append(avg_srpq)
            final_update_metrics["reg"].append(avg_reg)
            overall_update_epoch += 1

        # After training in this phase, update the non-max entries in tpcp.W.
        for i in range(tpcp.W.shape[0]):
            non_max_idx = non_max_idx_list[i]
            new_val = tpcp.W[i, non_max_idx].item() + delta_list[i]
            # Clamp to 1.0.
            tpcp.W[i, non_max_idx] = min(new_val, 1.0)
        print(f"After Final Update Phase {phase+1}: min(tpcp.W) = {tpcp.W.min().item()}")

    # --- Final Assertion on tpcp.W ---
    for i in range(tpcp.W.shape[0]):
        if tpcp.W[i].max().item() != 1.0:
            raise AssertionError(f"Row {i} of tpcp.W does not have a maximum of 1 after final update training.")
    print("Final update training completed: all rows in tpcp.W now have a maximum value of 1.")

    # --- Save Metrics to CSV and Plot ---
    # Build folder name from hyperparameters (excluding epoch-related parameters).
    folder_name1 = f"schedule_{schedule_type}_steps_{total_schedule_steps}_{manifold}"
    folder_name2 = f"lr_{lr}_epochs_{max_epochs}"
    base_dir = Path("metrics/aemps")
    folder_path = base_dir / folder_name1 / folder_name2
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save main training metrics.
    csv_filename = "metrics.csv"
    csv_filepath = folder_path / csv_filename
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(csv_filepath, index_label="epoch")
    utils.plot_training_metrics(
        range(1, len(metrics["loss"]) + 1),
        metrics["loss"],
        metrics["accuracy"],
        weight_ratio_vals=metrics.get("srpq", None),
        lambda_vals=metrics.get("lambda", None),
        title="TPCP Training Metrics over Epochs",
        filename=(folder_path / "tpcp_training_metrics.png").resolve().as_posix(),
    )
    print(f"Main training metrics saved to: {csv_filepath}")

    # Save final update training metrics.
    final_update_csv_filepath = folder_path / "final_update_metrics.csv"
    df_final_update = pd.DataFrame(final_update_metrics)
    df_final_update.to_csv(final_update_csv_filepath, index=False)
    print(f"Final update metrics saved to: {final_update_csv_filepath}")

    # Plot final update metrics: Loss, Accuracy, and Avg non-max W.
    plt.figure()
    plt.plot(final_update_metrics["update_epoch"], final_update_metrics["loss"])
    plt.xlabel("Update Epoch")
    plt.ylabel("Loss")
    plt.title("Final Update Training Loss")
    final_loss_plot_path = folder_path / "final_update_loss.png"
    plt.savefig(final_loss_plot_path.as_posix())
    plt.close()

    plt.figure()
    plt.plot(final_update_metrics["update_epoch"], final_update_metrics["accuracy"])
    plt.xlabel("Update Epoch")
    plt.ylabel("Accuracy")
    plt.title("Final Update Training Accuracy")
    final_acc_plot_path = folder_path / "final_update_accuracy.png"
    plt.savefig(final_acc_plot_path.as_posix())
    plt.close()

    plt.figure()
    plt.plot(final_update_metrics["update_epoch"], final_update_metrics["W_non_max_avg"])
    plt.xlabel("Update Epoch")
    plt.ylabel("Avg Non-Max W Value")
    plt.title("Final Update W Non-Max Values")
    final_W_plot_path = folder_path / "final_update_W_non_max.png"
    plt.savefig(final_W_plot_path.as_posix())
    plt.close()

    # --- Save the trained model ---
    model_filename = f"tpcp_model_{max_epochs}.pt"
    model_filepath = folder_path / model_filename
    torch.save(tpcp.state_dict(), model_filepath)
    print(f"Model saved to: {model_filepath}")

    return metrics
