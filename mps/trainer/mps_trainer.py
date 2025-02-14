#!/usr/bin/env python3
"""
Module for training a unitary MPS (uMPS) model.
Defines the mps_train() function.
"""

import time
import torch
import matplotlib.pyplot as plt

from mps import umps
from mps import unitary_optimizer

def mps_train(
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
    Train a unitary MPS (uMPS) model.
    
    Returns a list of batch losses and plots a loss curve.
    """
    model = umps.uMPS(N=N, chi=chi, d=d, l=l, layers=layers, device=device)
    optimizer = unitary_optimizer.Adam(model, lr=lr)
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    nnloss = torch.nn.NLLLoss(reduction="mean")
    all_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0
        t0 = time.time()
        for step, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            data = data.permute(1, 0, 2)  # [batch, N, 2] → [N, batch, 2]
            optimizer.zero_grad()
            outputs = model(data)
            outputs = logsoftmax(outputs)
            loss_val = nnloss(outputs, target)
            loss_val.backward()
            optimizer.step()
            all_losses.append(loss_val.item())
            epoch_loss += loss_val.item()
            preds = outputs.argmax(dim=-1)
            batch_acc = (preds == target).float().mean()
            epoch_acc += batch_acc.item()
            total_batches += 1
            if (step+1) % log_steps == 0:
                elapsed = time.time() - t0
                print(f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | Loss: {loss_val.item():.6f} | Acc: {batch_acc.item():.2%} | Time: {elapsed:.2f}s")
                t0 = time.time()
        avg_loss = epoch_loss / total_batches if total_batches else float("nan")
        avg_acc = epoch_acc / total_batches if total_batches else float("nan")
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2%}")
    plt.figure(figsize=(8,5))
    plt.plot(all_losses, label="Training Loss")
    plt.xlabel("Iteration (batch)")
    plt.ylabel("Loss")
    plt.title("uMPS Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return all_losses
