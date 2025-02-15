#!/usr/bin/env python3
"""
Module for training a simple MPS (sMPS) model.
Defines the smps_train() function.
"""

import time
import torch
from mps.simple_mps import SimpleMPS
from mps.trainer.utils import plot_training_metrics, loss_batch, calculate_accuracy


def smps_train(
    dataloader: torch.utils.data.DataLoader,
    N: int,
    d: int = 2,
    l: int = 2,
    epochs: int = 10,
    lr: float = 0.01,
    log_steps: int = 10,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    optimize: str = "greedy",
) -> SimpleMPS:
    """
    Train a simple MPS (sMPS) model on a given dataloader.
    The training step is combined similarly to the uMPS training in mps_trainer.py.
    
    Parameters:
      - dataloader: torch DataLoader that yields (data, target) batches.
      - device: torch.device to run training on.
      - N: number of sites.
      - chi: bond dimension.
      - d: physical dimension (default 2).
      - l: label dimension (default 2).
      - layers: number of layers (default 1).
      - epochs: number of training epochs.
      - lr: learning rate.
      - log_steps: number of batches between logging.
      - dtype: torch data type (default torch.float64).
    
    Returns:
    """
    # Build the sMPS model and its optimizer.
    # --- 1) Train SimpleMPS ---
    smps = SimpleMPS(N, chi = 2, d = d, l = l, layers=1, device=device, dtype=dtype, optimize=optimize)
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    nnloss = torch.nn.NLLLoss(reduction="mean")
    opt_smps = torch.optim.Adam(smps.parameters(), lr=lr)
    smps_losses = []
    smps.train()
    print(f"\n=== Training SimpleMPS for {epochs} epoch(s)... ===")
    for epoch in range(epochs):
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
    
    return smps