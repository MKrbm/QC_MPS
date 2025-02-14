#!/usr/bin/env python3
"""
Common utility functions for training.
"""

import torch
import matplotlib.pyplot as plt

def loss_batch(outputs, labels):
    """
    Binary cross-entropy–style loss: for each sample use `output` if label==0,
    otherwise use (1 - output).
    """
    device = outputs.device
    loss_val = torch.zeros(1, device=device, dtype=outputs.dtype)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss_val -= torch.log(prob + 1e-8)
    return loss_val

def calculate_accuracy(outputs, labels):
    """
    Compute accuracy by thresholding outputs at 0.5.
    """
    predictions = (outputs < 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / labels.numel()

def to_probs(outputs):
    """
    Convert outputs into probabilities (normalize along the last dimension).
    """
    return outputs / outputs.sum(dim=-1, keepdim=True)

def plot_training_metrics(x_axis, loss_vals, accuracy_vals, weight_ratio_vals, title, filename):
    """
    Plot loss, accuracy, and weight ratio on three y-axes.
    If weight_ratio_vals is None, only plot loss and accuracy.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(x_axis, loss_vals, label="Loss", marker="o", color='tab:blue')
    ax2.plot(x_axis, accuracy_vals, label="Accuracy", marker="s", color='tab:orange')
    
    ax1.set_xlabel("Epoch (Cumulative)")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax2.set_ylabel("Accuracy", color='tab:orange')
    
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 1)
    
    if weight_ratio_vals is not None:
        ax3 = ax1.twinx()
        # Offset the third axis spine.
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(x_axis, weight_ratio_vals, label="Weight Ratio", marker="^", color='tab:green')
        ax3.set_ylabel("Weight Ratio", color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()
