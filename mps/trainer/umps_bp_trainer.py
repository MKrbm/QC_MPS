#!/usr/bin/env python3
"""
Module for training a unitary MPS (uMPS) model.
Defines the umps_train() function.
"""

import time
import torch
import matplotlib.pyplot as plt  # Ensure matplotlib is imported

from mps import umps
from mps import unitary_optimizer
from mps.trainer.utils import plot_training_metrics, loss_batch, calculate_accuracy, plot_gradient_norms

def umps_bp_train(
    dataloader,
    device,
    N,
    chi,
    d=2,
    l=2,
    layers=1,
    epochs=10,
    lr=0.01,
    log_steps=10,
    dtype=torch.float64,
):
    """
    Train a unitary MPS (uMPS) model on a given dataloader.
    
    Parameters:
      - dataloader: torch DataLoader that yields (data, target) batches.
      - device: torch.device to run training on.
      - N: number of sites (e.g. flattened image size).
      - chi: bond dimension.
      - d: physical dimension (default 2).
      - l: label dimension (default 2).
      - layers: number of layers (default 1).
      - epochs: number of training epochs.
      - lr: learning rate.
      - log_steps: number of batches between logging.
      - dtype: torch data type (default torch.float64).
    
    Returns:
      - all_losses: a list containing the loss after each batch.
    """
    # Build the uMPS model and its optimizer.
    model = umps.uMPS(N=N, chi=chi, d=d, l=l, layers=layers, device=device)
    optimizer = unitary_optimizer.Adam(model, lr=lr)
    all_losses = []      # Store loss for each batch.
    epoch_losses = []    # Store average loss per epoch.
    epoch_accs = []      # Store average accuracy per epoch.
    
    # Lists to store average gradient norms per epoch
    gradients_0 = []     # For 0-th qubit
    gradients_mid = []   # For N//2-th qubit
    gradients_last = []  # For N-1-th qubit
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0

        # Accumulate gradient norms for the epoch
        grad_epoch_0 = 0.0
        grad_epoch_mid = 0.0
        grad_epoch_last = 0.0

        t0 = time.time()
        
        for step, (data, target) in enumerate(dataloader):
            # Data comes in as [batch, H*W, 2] (after embedding). We need [site, batch, features].
            data = data.to(device).permute(1, 0, 2)
            target = target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)  # forward pass
            batch_loss = loss_batch(outputs, target)
            batch_loss.backward()
            
            # --- Record gradient norms for specific qubits ---
            # (Assumes that model.tensors is a list of site parameters.)
            # Use integer division for the middle qubit index.
            if model.params[0].grad is not None:
                g = model.params[0].grad.data.reshape(4,4)
                u = model.params[0].data.reshape(4,4)
                rg = unitary_optimizer.riemannian_gradient(u, g)
                g0 = rg.norm().item()
            else:
                g0 = 0.0
            if model.params[N // 2].grad is not None:
                gmid = model.params[N // 2].grad.data.reshape(4,4)
                u = model.params[N // 2].data.reshape(4,4)
                rg = unitary_optimizer.riemannian_gradient(u, gmid)
                gmid = rg.norm().item()
            else:
                gmid = 0.0
            if model.params[-1].grad is not None:
                glast = model.params[-1].grad.data.reshape(4,4)
                u = model.params[-1].data.reshape(4,4)  
                rg = unitary_optimizer.riemannian_gradient(u, glast)
                glast = rg.norm().item()
            else:
                glast = 0.0
            
            grad_epoch_0 += g0
            grad_epoch_mid += gmid
            grad_epoch_last += glast
            # ---------------------------------------------------
            
            optimizer.step()
            
            # Project back onto unitary (if needed)
            model.proj_unitary(check_on_unitary=True, print_log=False, rtol=1e-3)
            
            all_losses.append(batch_loss.item())
            epoch_loss += batch_loss.item()
            
            batch_acc = calculate_accuracy(outputs.detach(), target)
            epoch_acc += batch_acc.item()
            total_batches += 1
            
            if (step + 1) % log_steps == 0:
                elapsed = time.time() - t0
                loss_per_data = batch_loss.item() / data.shape[0]
                print(f"Epoch {epoch+1:02} | Step {step+1:03}/{len(dataloader):03} | "
                      f"Loss: {loss_per_data:.6f} | Acc: {batch_acc.item():.2%} | Time: {elapsed:.2f}s")
                t0 = time.time()
                
        # Compute average loss, accuracy, and gradient norms for this epoch.
        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_acc = epoch_acc / total_batches if total_batches > 0 else float("nan")
        avg_grad_0 = grad_epoch_0 / total_batches
        avg_grad_mid = grad_epoch_mid / total_batches
        avg_grad_last = grad_epoch_last / total_batches
        
        print(f"Epoch {epoch+1:02} | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2%} | "
              f"Gradients -> 0th: {avg_grad_0:.6e}, Mid: {avg_grad_mid:.6e}, Last: {avg_grad_last:.6e}")
        
        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
        gradients_0.append(avg_grad_0)
        gradients_mid.append(avg_grad_mid)
        gradients_last.append(avg_grad_last)
    
    # Plot the training metrics using the shared utility function.
    x_axis = list(range(1, epochs + 1))
    title = "uMPS Training Metrics"
    filename = "training_metrics.png"
    # plot_training_metrics(x_axis, epoch_losses, epoch_accs, None, None, title, filename)
    
    # # Plot the gradient norms for the selected qubits.
    # plot_gradient_norms(x_axis, gradients_0, gradients_mid, gradients_last)
    
    return all_losses

