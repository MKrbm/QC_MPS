#!/usr/bin/env python3
"""
Module for training a unitary MPS (uMPS) model.
Defines the mps_train() function.
"""

import time
import torch

from mps import umps
from mps import unitary_optimizer
from mps.trainer.utils import plot_training_metrics, loss_batch, calculate_accuracy

def umps_train(
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
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0
        t0 = time.time()
        
        for step, (data, target) in enumerate(dataloader):
            # Data comes in as [batch, H*W, 2] (after embedding). We need [site, batch, features].
            data = data.to(device).permute(1, 0, 2)
            target = target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)  # forward pass
            batch_loss = loss_batch(outputs, target)
            batch_loss.backward()
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
                print(f"Epoch {epoch+1:02} | Step {step+1:03}/{len(dataloader):03} | "
                      f"Loss: {batch_loss.item():.6f} | Acc: {batch_acc.item():.2%} | Time: {elapsed:.2f}s")
                t0 = time.time()
                
        avg_loss = epoch_loss / total_batches if total_batches > 0 else float("nan")
        avg_acc = epoch_acc / total_batches if total_batches > 0 else float("nan")
        print(f"Epoch {epoch+1:02} | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2%}")
        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
    
    # Plot the training metrics using the shared utility function.
    x_axis = list(range(1, epochs + 1))
    title = "uMPS Training Metrics"
    filename = "training_metrics.png"
    plot_training_metrics(x_axis, epoch_losses, epoch_accs, None, title, filename)
    
    return all_losses
