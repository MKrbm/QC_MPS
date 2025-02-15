#!/usr/bin/env python3
"""
Module for training TPCP with a regularization scheduler.
This training function uses several scheduling strategies (linear, polynomial,
soft exponential, cosine) to control the regularization coefficient (λ) that
multiplies the regularization term. The training proceeds in phases: at each phase,
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

# Import required functions and modules from the MPS package.
from mps import tpcp_mps  # contains MPSTPCP and regularize_weight
from mps.trainer import utils
from mps.trainer.utils import loss_batch, calculate_accuracy  # assumed to be defined
from mps.StiefelOptimizers import StiefelAdam
from mps.radam import RiemannianAdam


# =============================================================================
# Helper: Schedule Function
# =============================================================================
def get_scheduled_lambda(schedule_type, step, total_steps, lambda_final, poly_power=2, k=0.1):
    """
    Compute λ based on the scheduling strategy.
    
    Args:
      schedule_type (str): 'linear', 'polynomial', 'soft_exponential', or 'cosine'
      step (int): current scheduling step (0 <= step <= total_steps)
      total_steps (int): total number of scheduling steps.
      lambda_final (float): final (target) λ value.
      poly_power (float): power for polynomial schedule.
      k (float): rate parameter for soft exponential schedule.
    
    Returns:
      float: scheduled λ.
    """
    if schedule_type == 'linear':
        return (step / total_steps) * lambda_final
    elif schedule_type == 'polynomial':
        return ((step / total_steps) ** poly_power) * lambda_final
    elif schedule_type == 'soft_exponential':
        # λ = λ_final * (1 - exp(-k * step)) / (1 - exp(-k * total_steps))
        # (ensuring that λ(0)=0 and λ(total_steps)=λ_final)
        return lambda_final * (1 - math.exp(-k * step)) / (1 - math.exp(-k * total_steps))
    elif schedule_type == 'cosine':
        # cosine annealing from 0 to λ_final:
        # λ = λ_final * (1 - cos(pi * step/total_steps)) / 2
        return lambda_final * (1 - math.cos(math.pi * step / total_steps)) / 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

# =============================================================================
# Main Training Function
# =============================================================================
def mpsae_adaptive_train(
    trainloader,
    device,
    N,
    total_schedule_steps=10,
    schedule_type='linear',  # choose among 'linear', 'polynomial', 'soft_exponential', 'cosine'
    poly_power=2,
    k=0.1,
    epochs_per_phase=100,
    min_epochs=3,
    conv_threshold=1e-4,
    patience=2,
    conv_strategy="absolute",  # or "relative"
    spike_threshold=0.1,  # if loss increases >10% compared to previous epoch, consider it a spike
    log_steps=10,
    lr=0.0001
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
    
    Args:
      trainloader: data loader.
      device: torch.device.
      N: system size parameter for MPSTPCP.
      total_schedule_steps (int): number of steps in the λ schedule.
      lambda_final (float): target (final) value for λ (will be updated).
      schedule_type (str): scheduling strategy.
      poly_power (float): exponent for polynomial schedule.
      k (float): parameter for soft exponential schedule.
      epochs_per_phase (int): maximum epochs per λ phase.
      min_epochs (int): minimum epochs before checking convergence.
      conv_threshold (float): threshold for loss change (absolute or relative) to declare convergence.
      patience (int): number of consecutive epochs with low change to declare convergence.
      conv_strategy (str): "absolute" or "relative" difference.
      spike_threshold (float): relative increase in loss that signals a spike.
      log_steps (int): logging interval (in steps).
      lr (float): learning rate for optimizers.
    
    Returns:
      dict: metrics containing lists for loss, accuracy, and λ values.
    """
    # --------------------------
    # 1. Build and Prepare TPCP
    # --------------------------
    # Create the TPCP model. (Here we assume d=2, K=1, with_identity=True, and using EXACT manifold.)
    tpcp = tpcp_mps.MPSTPCP(N, K=1, d=2, with_probs=False, with_identity=True, manifold=tpcp_mps.ManifoldType.EXACT)
    tpcp.to(device)
    tpcp.train()
    
    # (Assume that a canonical SimpleMPS has already been trained and set as follows.)
    # For example: tpcp.set_canonical_mps(smps)
    # You must ensure that 'smps' is defined and trained before calling this function.
    
    # Initialize W as in your code: start with w=0.
    W = torch.zeros(tpcp.L, 2, dtype=torch.float64, device=device)
    W[:, 0] = 1
    W[:, 1] = 0
    tpcp.initialize_W(W)
    
    # -------------------------------------------------------
    # 1.5 Determine lambda_final Using the Initial Loss Value
    # -------------------------------------------------------
    # Get one mini-batch from the trainloader.
    data_batch, target_batch = next(iter(trainloader))
    data_batch, target_batch = data_batch.to(device), target_batch.to(device)
    with torch.no_grad():
        initial_outputs = tpcp(data_batch)
        # Convert model outputs to probabilities; assume the probability for label 0 is used.
        initial_probs = utils.to_probs(initial_outputs)[:, 0]
        initial_loss = loss_batch(initial_probs, target_batch)
        initial_reg_weight = tpcp_mps.regularize_weight(tpcp.W)
    print(f"Initial loss for λ determination: {initial_loss.item():.6f}")
    print(f"Initial regularization weight: {initial_reg_weight.item():.6f}")
    # Update lambda_final based on the initial loss.
    lambda_final = initial_loss.item() / initial_reg_weight.item()
    print(f"Updated lambda_final set to: {lambda_final:.6f}")
    
    # Create optimizers: one for the Kraus ops (using RiemannianAdam) and one for W and r.
    optimizer = RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_weight = torch.optim.Adam([tpcp.W, tpcp.r], lr=lr)
    
    # --------------------------
    # 2. Scheduler Setup
    # --------------------------
    current_schedule_step = 0
    current_lambda = get_scheduled_lambda(schedule_type, current_schedule_step, total_schedule_steps, lambda_final, poly_power, k)
    print(f"Starting training with regularization scheduler. Initial λ = {current_lambda:.6f}")
    
    # --------------------------
    # 3. Training Loop Over λ Phases
    # --------------------------
    metrics = {"loss": [], "accuracy": [], "lambda": [], "weight_rate": []}
    epoch_total = 0
    
    while current_lambda < lambda_final and current_schedule_step <= total_schedule_steps:
        print(f"\n=== Training Phase with λ = {current_lambda:.6f} (Schedule Step {current_schedule_step}/{total_schedule_steps}) ===")
        phase_epoch = 0
        conv_counter = 0
        prev_epoch_loss = None
        
        while phase_epoch < epochs_per_phase:
            epoch_loss_sum = 0.0
            epoch_acc_sum = 0.0
            total_samples = 0
            t0 = time.time()
            
            # Loop over mini-batches
            for step, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                optimizer_weight.zero_grad()
                
                outputs = tpcp(data)
                outputs = utils.to_probs(outputs)[:, 0]  # probability for label 0
                loss = loss_batch(outputs, target)
                
                # Compute the regularization weight (e.g., from tpcp.W)
                reg_weight = tpcp_mps.regularize_weight(tpcp.W)
                loss_with_reg = loss + current_lambda * reg_weight
                
                loss_with_reg.backward()
                optimizer.step()
                optimizer_weight.step()
                tpcp.normalize_w_and_r()
                tpcp.proj_stiefel(check_on_manifold=True, print_log=False)
                
                bs = target.size(0)
                epoch_loss_sum += loss.item() * bs  # record the primary loss (without reg)
                total_samples += bs
                batch_acc = calculate_accuracy(outputs.detach(), target)
                epoch_acc_sum += batch_acc.item() * bs
                
                if (step + 1) % log_steps == 0:
                    print(f"[λ {current_lambda:.6f}] Epoch {phase_epoch+1}, Step {step+1}/{len(trainloader)} | "
                          f"Batch Loss: {loss.item():.6f} | Reg: {reg_weight.item():.6f} | "
                          f"Loss+Reg: {loss_with_reg.item():.6f} | Acc: {batch_acc.item():.2%}")
            
            # End of epoch: compute averages.
            avg_loss = epoch_loss_sum / total_samples
            avg_acc = epoch_acc_sum / total_samples
            elapsed = time.time() - t0
            print(f"[λ {current_lambda:.6f}] Epoch {phase_epoch+1} | Avg Loss: {avg_loss:.6f} | Acc: {avg_acc:.2%} | Time: {elapsed:.2f}s")
            
            # Record metrics
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(avg_acc)
            metrics["lambda"].append(current_lambda)
            metrics["weight_rate"].append(reg_weight.item())
            
            # Hybrid: Check for spikes and convergence.
            if prev_epoch_loss is not None:
                # If loss increased by more than spike_threshold, hold λ update.
                if avg_loss > prev_epoch_loss * (1 + spike_threshold):
                    print(f"Spike detected: Loss increased from {prev_epoch_loss:.6f} to {avg_loss:.6f}. Holding λ update.")
                    conv_counter = 0  # reset convergence counter
                else:
                    # Check convergence criteria (absolute or relative difference)
                    if phase_epoch >= min_epochs:
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
            prev_epoch_loss = avg_loss
            phase_epoch += 1
            epoch_total += 1
            
            # If convergence is achieved for the current λ phase, break out.
            if conv_counter >= patience:
                print(f"Convergence achieved for λ {current_lambda:.6f} after {conv_counter} stable epochs.")
                break
        
        # Update λ for the next phase.
        current_schedule_step += 1
        new_lambda = get_scheduled_lambda(schedule_type, current_schedule_step, total_schedule_steps, lambda_final, poly_power, k)
        if new_lambda <= current_lambda:
            print("Scheduled λ did not increase; ending training.")
            break
        current_lambda = new_lambda
    
    # Optionally, plot the training metrics.
    x_axis = range(1, len(metrics["loss"]) + 1)
    utils.plot_training_metrics(x_axis, metrics["loss"], metrics["accuracy"],
                                weight_ratio_vals=metrics.get("weight_rate", None),
                                lambda_vals=metrics.get("lambda", None),
                                title="TPCP Training Metrics over Epochs",
                                filename="tpcp_training_metrics.png")
    
    return metrics
