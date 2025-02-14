#!/usr/bin/env python3

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import geoopt
import matplotlib.pyplot as plt

# If your mps/ package is local, ensure it’s on sys.path or installed in editable mode:
# sys.path.append("/path/to/your/project/root")

from mps.simple_mps import SimpleMPS
from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps.StiefelOptimizers import StiefelAdam, StiefelSGD
from mps.radam import RiemannianAdam


###############################################################################
# MNIST dataset utilities
###############################################################################
def filter_digits(dataset, allowed_digits=[0, 1]):
    """Return a subset of MNIST dataset containing only allowed_digits (0 or 1)."""
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in allowed_digits:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)


def filiter_single_channel(img: torch.Tensor) -> torch.Tensor:
    """
    MNIST is loaded as shape [C, H, W].
    Take only the first channel => shape [H, W].
    """
    return img[0, ...]


def embedding_pixel(batch, label: int = 0):
    """
    Flatten each image from shape [H, W] => [H*W],
    then embed x => [x, 1-x], and L2-normalize along last dim.
    """
    pixel_size = batch.shape[-1] * batch.shape[-2]
    x = batch.view(*batch.shape[:-2], pixel_size)
    x = torch.stack([x, 1 - x], dim=-1)
    # L2-normalize with a small epsilon to avoid division by zero
    x = x / torch.sum(x, dim=-1, keepdim=True).clamp(min=1e-8)
    return x


###############################################################################
# Loss & Accuracy
###############################################################################
def loss_batch(outputs, labels):
    """
    Binary cross-entropy style loss for outputs in [0, 1].
    For label=0 => use outputs[i], for label=1 => use 1 - outputs[i].
    """
    device = outputs.device
    loss_val = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss_val -= torch.log(prob + 1e-8)
        if torch.isnan(loss_val):
            print(f"NaN in loss at index {i}, prob={prob}, output={outputs[i]}, label={labels[i]}")
    return loss_val


def calculate_accuracy(outputs, labels):
    """
    Threshold outputs at 0.5 to assign label 0 or 1 and compare to true labels.
    """
    predictions = (outputs < 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / labels.numel()


###############################################################################
# Main Training Function
###############################################################################
def train_tpcp_mnist(
    manifold: str = "Exact",
    optimizer_name: str = "adam",
    K: int = 1,
    batch_size: int = 128,
    max_epochs: int = 10,
    conv_strategy: str = "absolute",
    conv_threshold: float = 1e-4,
    min_epochs: int = 3,
    patience: int = 2,
    num_data: int | None = None,
    lr: float = 0.01,
    log_steps: int = 10,
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
):
    """
    Train a TPCP MPS model on MNIST (digits 0 and 1).

    This function:
      1) Loads and filters MNIST (resized to 16x16).
      2) Trains a SimpleMPS for one epoch (as a demo).
      3) Constructs an MPSTPCP model with the chosen manifold.
         (Allowed manifold choices include "Exact", "Frobenius", "Canonical",
         "Cayley", "MatrixExp", "ForwardEuler", and "Original".)
      4) Chooses the Riemannian optimizer based on the chosen manifold and optimizer_name.
      5) Trains the TPCP model for up to max_epochs, stopping early if the loss converges.
         Convergence is declared only if the change in loss is below conv_threshold
         for a number of consecutive epochs (patience), but not before min_epochs.
      6) Records per-epoch loss, accuracy, and the weight rate (average of W[:,1]/sum(W,1)).
      7) Returns a dictionary with the SimpleMPS loss history and a mapping from each w value
         to its training metrics.

    Args:
      manifold: Type of manifold/update rule (including "Original").
      optimizer_name: Which optimizer to use ("adam" or "sgd").
      K: Number of Kraus operators per site.
      batch_size: Batch size for training.
      max_epochs: Maximum number of epochs for TPCP training.
      conv_strategy: Convergence strategy ("absolute" or "relative").
      conv_threshold: Threshold for convergence.
      min_epochs: Minimum epochs to run before considering convergence.
      patience: Number of consecutive epochs satisfying convergence condition required for early stopping.
      num_data: If specified, limit the dataset to this many examples.
      lr: Learning rate.
      log_steps: Frequency (in steps) to log batch loss.
      device: Device to run the training on.

    Returns:
      A dictionary with keys:
         "simple_mps_losses": list of losses from SimpleMPS training.
         "tpcp_metrics_by_w": dict mapping each w value to a dict with keys "loss", "accuracy", "weight_rate".
    """
    # --------------------------
    # 1) Prepare MNIST dataset
    # --------------------------
    img_size = 16
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(filiter_single_channel),
            transforms.Lambda(embedding_pixel),
            transforms.Lambda(lambda x: x.to(torch.float64)),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    trainset = filter_digits(trainset, allowed_digits=[0, 1])
    if num_data is not None and num_data < len(trainset):
        trainset = torch.utils.data.Subset(trainset, range(num_data))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # --------------------------
    # 2) Train a SimpleMPS model (demo)
    # --------------------------
    N = img_size * img_size  # 256
    d = l = 2  # data input and label dimensions
    smps = SimpleMPS(
        N,
        2,
        d,
        l,
        layers=2,
        device=device,
        dtype=torch.float64,
        optimize="greedy",
    )
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    nnloss = torch.nn.NLLLoss(reduction="mean")
    opt_smps = torch.optim.Adam(smps.parameters(), lr=0.001)

    smps_losses = []
    smps.train()
    print("\n=== Training SimpleMPS for 1 epoch... ===")
    total_loss_smps = 0.0
    total_samples_smps = 0
    total_correct_smps = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        data = data.permute(1, 0, 2)  # shape [256, batch_size, 2]
        opt_smps.zero_grad()
        outputs = smps(data)  # shape [batch_size, 2]
        outputs = logsoftmax(outputs)
        loss = nnloss(outputs, target)
        loss.backward()
        opt_smps.step()

        bs = target.size(0)
        total_loss_smps += loss.item() * bs
        total_samples_smps += bs

        # Calculate batch accuracy
        preds = outputs.argmax(dim=-1)
        total_correct_smps += (preds == target).float().sum().item()

        if (batch_idx + 1) % log_steps == 0:
            avg_loss = total_loss_smps / total_samples_smps
            avg_accuracy = total_correct_smps / total_samples_smps
            smps_losses.append(avg_loss)
            print(f"[SimpleMPS] Batch {batch_idx+1}/{len(trainloader)} | Loss: {avg_loss:.6f} | Accuracy: {avg_accuracy:.2%}")
            total_loss_smps = 0.0
            total_samples_smps = 0
            total_correct_smps = 0

    # --------------------------
    # 3) Build TPCP model with chosen manifold
    # --------------------------
    manifold_map = {
        "Exact": ManifoldType.EXACT,
        "Frobenius": ManifoldType.FROBENIUS,
        "Canonical": ManifoldType.CANONICAL,
        "Cayley": ManifoldType.EXACT,
        "MatrixExp": ManifoldType.EXACT,
        "ForwardEuler": ManifoldType.EXACT,
        "Original": ManifoldType.EXACT,  # use EXACT underlying type for "Original"
    }
    if manifold not in manifold_map:
        raise ValueError(f"Invalid manifold='{manifold}'. Valid options: {list(manifold_map.keys())}.")

    tpcp = MPSTPCP(N, K=K, d=2, with_identity=True, manifold=manifold_map[manifold])
    tpcp.to(device)
    tpcp.train()
    # Optionally initialize TPCP from the SimpleMPS (canonical form)
    tpcp.set_canonical_mps(smps)

    # --------------------------
    # 4) Choose an optimizer for TPCP
    # --------------------------
    if manifold in ["Cayley", "MatrixExp", "ForwardEuler"]:
        if optimizer_name.lower() == "adam":
            opt_tpcp = StiefelAdam(tpcp.parameters(), lr=lr, expm_method=manifold)
        elif optimizer_name.lower() == "sgd":
            opt_tpcp = StiefelSGD(tpcp.parameters(), lr=lr, expm_method=manifold)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
    elif manifold in ["Exact", "Frobenius", "Canonical"]:
        if optimizer_name.lower() == "adam":
            opt_tpcp = geoopt.optim.RiemannianAdam(tpcp.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            opt_tpcp = geoopt.optim.RiemannianSGD(tpcp.parameters(), lr=lr)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
    elif manifold == "Original":
        if optimizer_name.lower() == "adam":
            opt_tpcp = RiemannianAdam(tpcp.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            raise ValueError("SGD is not supported for the original update rule.")
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")

    # --------------------------
    # 5) Training TPCP MPS with convergence checking and metrics recording
    # --------------------------
    # We iterate over different weighting factors w.
    w_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tpcp_metrics_by_w = dict()

    for w_ in w_values:
        # Re-initialize weight parameter W for current w_
        W = torch.zeros(tpcp.L, 2, dtype=torch.float64, device=device)
        W[:, 0] = 1
        W[:, 1] = w_
        tpcp.initialize_W(W)

        print(f"\n=== Training TPCP with initial weight rate w={w_:.1f} (max_epochs={max_epochs}) ===")
        epoch = 0
        prev_epoch_loss = None
        conv_counter = 0  # counts consecutive epochs satisfying convergence condition
        metrics = {"loss": [], "accuracy": [], "weight_rate": []}

        while epoch < max_epochs:
            epoch_loss_sum = 0.0
            total_samples = 0
            epoch_acc_sum = 0.0
            total_acc_samples = 0
            t0 = time.time()
            for step, (data, target) in enumerate(trainloader):
                data = data.to(device)
                target = target.to(device, dtype=torch.int64)
                bs = target.size(0)

                opt_tpcp.zero_grad()
                outputs = tpcp(data)
                if torch.isnan(outputs).any():
                    print(f"NaN detected in outputs at step {step}, skipping batch.")
                    continue

                loss_val = loss_batch(outputs, target)
                if torch.isnan(loss_val):
                    print("NaN detected in batch loss, skipping batch.")
                    continue

                loss_val.backward()
                opt_tpcp.step()

                # Optionally project back to the manifold
                tpcp.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)

                epoch_loss_sum += loss_val.item() * bs
                total_samples += bs
                batch_acc = calculate_accuracy(outputs.detach(), target)
                epoch_acc_sum += batch_acc.item() * bs
                total_acc_samples += bs

                if (step + 1) % log_steps == 0:
                    print(f"[TPCP::w={w_:.1f}] Epoch {epoch+1}, Step {step+1}/{len(trainloader)} | Batch Loss: {loss_val.item():.6f} | Batch Accuracy: {batch_acc.item():.2%}")

            if total_samples == 0:
                print("No samples processed in this epoch, skipping.")
                continue

            avg_loss = epoch_loss_sum / total_samples
            avg_acc = epoch_acc_sum / total_acc_samples

            # Compute the current weight rate (average over sites)
            current_weight_rate = (tpcp.W[:, 1] / torch.sum(tpcp.W, dim=1)).mean().item() if hasattr(tpcp, "W") else w_

            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(avg_acc)
            metrics["weight_rate"].append(current_weight_rate)

            elapsed = time.time() - t0
            print(f"[TPCP::w={w_:.1f}] Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}, Avg Acc: {avg_acc:.2%}, Weight Rate: {current_weight_rate:.4f} | Time: {elapsed:.2f}s")

            # Check convergence only after min_epochs have been reached
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
                        print(f"Convergence reached: Condition met for {conv_counter} consecutive epochs.")
                        break
            prev_epoch_loss = avg_loss
            epoch += 1

        tpcp_metrics_by_w[w_] = metrics

    return {"simple_mps_losses": smps_losses, "tpcp_metrics_by_w": tpcp_metrics_by_w}


def main():
    parser = argparse.ArgumentParser(description="Train TPCP MPS on MNIST (digits 0 & 1).")

    parser.add_argument(
        "--manifold",
        default="Exact",
        choices=["Exact", "Frobenius", "Canonical", "Cayley", "MatrixExp", "ForwardEuler", "Original"],
        help="Manifold/update rule for Kraus ops (default Exact).",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd"],
        help="Riemannian optimizer to use (default adam).",
    )
    parser.add_argument("--K", type=int, default=1, help="# of Kraus operators per site (default 1).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default 128).")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs for TPCP training (default 10).")
    parser.add_argument("--min_epochs", type=int, default=3, help="Minimum epochs before convergence is checked (default 3).")
    parser.add_argument("--patience", type=int, default=2, help="Number of consecutive epochs meeting convergence condition required to stop (default 2).")
    parser.add_argument("--conv_strategy", type=str, default="absolute", choices=["absolute", "relative"],
                        help="Convergence strategy for loss (absolute or relative).")
    parser.add_argument("--conv_threshold", type=float, default=1e-4, help="Threshold for loss convergence (default 1e-4).")
    parser.add_argument("--num_data", type=int, default=None, help="Limit dataset to N examples (default: entire dataset).")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01).")
    parser.add_argument("--log_steps", type=int, default=10, help="Steps between logging loss values (default 10).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: system-based).")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using fixed random seed: {args.seed}")
    else:
        print("Using system-based randomness.")

    results = train_tpcp_mnist(
        manifold=args.manifold,
        optimizer_name=args.optimizer,
        K=args.K,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        conv_strategy=args.conv_strategy,
        conv_threshold=args.conv_threshold,
        num_data=args.num_data,
        lr=args.lr,
        log_steps=args.log_steps,
    )

    # --------------------------
    # Plotting: For each w value, plot loss, accuracy, and weight_rate on the same axes.
    # --------------------------
    tpcp_metrics_by_w = results["tpcp_metrics_by_w"]
    n_plots = len(tpcp_metrics_by_w)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots), squeeze=False)
    for idx, (w_val, metrics) in enumerate(tpcp_metrics_by_w.items()):
        ax = axs[idx, 0]
        epochs_range = range(1, len(metrics["loss"]) + 1)
        ax.plot(epochs_range, metrics["loss"], label="Loss", marker="o")
        ax.plot(epochs_range, metrics["accuracy"], label="Accuracy", marker="s")
        ax.plot(epochs_range, metrics["weight_rate"], label="Weight Rate", marker="^")
        ax.set_title(f"TPCP Metrics for w = {w_val}")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
