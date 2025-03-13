#!/usr/bin/env python3
"""
Common utility functions for training.
"""

import torch
import matplotlib.pyplot as plt


def loss_batch(probs, targets, alpha=0.5, gamma=0.0, reduction='mean'):
    """
    Focal loss for binary classification with probabilities as input.
    
    Args:
        probs (Tensor): shape (N, 2), row i is [p_class0, p_class1].
        targets (Tensor): shape (N, ), each value in {0,1}.
        alpha (float): Weight for the positive class (class=1). 
                       The weight for class=0 becomes (1 - alpha).
        gamma (float): The focusing parameter.
        reduction (str): 'mean', 'sum', or 'none' for output reduction.
        
    Returns:
        Tensor: scalar (if mean or sum) or per-sample (if 'none') focal loss.
    """
    # Ensure targets is a LongTensor (for indexing)
    targets = targets.long()

    # p_t: probability assigned to the correct class for each sample
    # p_t[i] = probs[i, targets[i]]
    p_t = probs[torch.arange(len(probs)), targets]

    # alpha_t: weight is alpha for class=1, and (1-alpha) for class=0
    alpha_t = alpha * (targets == 1).float() + (1 - alpha) * (targets == 0).float()

    # focal factor = (1 - p_t)^gamma
    focal_factor = (1 - p_t) ** gamma

    # compute the focal loss (add a small epsilon to avoid log(0))
    loss = -alpha_t * focal_factor * torch.log(p_t + 1e-8)

    # reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # 'none'



def mean_risk(outputs, labels):
    """
    Computes the mean risk for a batch of outputs and labels.

    The risk is defined as the absolute difference between the predicted probability
    and the true label.

    Args:
        outputs (torch.Tensor): The predicted probabilities for each sample (N x 2).
        labels (torch.Tensor): The true labels for each sample (N).

    Returns:
        torch.Tensor: The mean risk over the batch.
    """
    # Select the predicted probabilities corresponding to the true labels
    predicted_probs = outputs[torch.arange(outputs.size(0)), labels]
    risk = torch.abs(predicted_probs - 1)
    return risk.mean()

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

def plot_training_metrics(x_axis, loss_vals, accuracy_vals, weight_ratio_vals=None, lambda_vals=None, title="", filename="plot.png"):
    """
    Plot loss, accuracy, weight ratio, and lambda on separate y-axes.
    
    Args:
      x_axis (list or array): x-axis values (typically cumulative epochs).
      loss_vals (list or array): Loss values.
      accuracy_vals (list or array): Accuracy values.
      weight_ratio_vals (list or array, optional): Weight ratio values. If None, weight ratio is not plotted.
      lambda_vals (list or array, optional): Regularization coefficient values. If None, λ is not plotted.
      title (str): Overall title for the plot.
      filename (str): Filename to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot loss on the left y-axis
    ax1.plot(x_axis, loss_vals, label="Loss", marker="o", color='tab:blue')
    ax1.set_xlabel("Epoch (Cumulative)")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot accuracy on the right y-axis
    ax2.plot(x_axis, accuracy_vals, label="Accuracy", marker="s", color='tab:orange')
    ax2.set_ylabel("Accuracy", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 1)
    
    # If weight ratio is provided, add a third axis.
    if weight_ratio_vals is not None:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(x_axis, weight_ratio_vals, label="Weight Ratio", marker="^", color='tab:green')
        ax3.set_ylabel("Weight Ratio", color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
    
    # If lambda values are provided, add an additional axis.
    if lambda_vals is not None:
        # If weight ratio was also plotted, offset further; otherwise, use an offset of 60.
        offset = 120 if weight_ratio_vals is not None else 60
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', offset))
        ax4.plot(x_axis, lambda_vals, label="λ", marker="v", color='tab:red')
        ax4.set_ylabel("λ", color='tab:red')
        ax4.tick_params(axis='y', labelcolor='tab:red')
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_gradient_norms(x_axis, grad0, grad_mid, grad_last):
    """
    Plot the average gradient norms for the 0-th, N//2-th, and N-1-th qubits over epochs.
    
    Args:
      x_axis (list or array): Epoch numbers.
      grad0 (list or array): Average gradient norm for the 0-th qubit per epoch.
      grad_mid (list or array): Average gradient norm for the middle (N//2-th) qubit per epoch.
      grad_last (list or array): Average gradient norm for the last (N-1-th) qubit per epoch.
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    axs[0].plot(x_axis, grad0, marker='o', color='tab:blue')
    axs[0].set_title("Gradient Norm - 0-th Qubit")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Gradient Norm")
    
    axs[1].plot(x_axis, grad_mid, marker='o', color='tab:green')
    axs[1].set_title("Gradient Norm - N//2-th Qubit")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Gradient Norm")
    
    axs[2].plot(x_axis, grad_last, marker='o', color='tab:red')
    axs[2].set_title("Gradient Norm - N-1-th Qubit")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Gradient Norm")
    
    fig.tight_layout()
    plt.savefig("gradient_norms.png", dpi=300)
    plt.show()