#!/usr/bin/env python3

import argparse
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import geoopt
import matplotlib.pyplot as plt
import time

import sys

# If your mps/ package is local, ensure it’s on sys.path or installed in editable mode:
# sys.path.append("/path/to/your/project/root")

from mps.tpcp_mps import MPSTPCP, ManifoldType


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
    x = x / torch.norm(x, dim=-1).unsqueeze(-1)
    return x


###############################################################################
# Loss & Accuracy
###############################################################################
def loss_batch(outputs, labels):
    """
    Binary cross-entropy style loss for outputs in [0, 1].
    For label=0 => prob=outputs[i], else => 1 - outputs[i].
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss -= torch.log(prob + 1e-8)
        # Start of Selection
        if torch.isnan(loss):
            print(f"Loss is NaN at i={i}")
            print(prob, outputs[i], labels[i])
    return loss


def calculate_accuracy(outputs, labels):
    """
    Threshold 0.5 => label 0 or 1. Compare to true labels.
    """
    predictions = (outputs < 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / labels.numel()


###############################################################################
# Training function
###############################################################################
def train_tpcp_mnist(
    manifold: str,
    optimizer_name: str,
    K: int,
    batch_size: int,
    epochs: int,
    num_data: int | None = None,
    lr: float = 0.01,
    log_steps: int = 100,
):
    """
    Fully train an MPSTPCP model on MNIST (digits 0 and 1) with the given parameters:
      - manifold in {EXACT, FROBENIUS, CANONICAL}
      - optimizer_name in {adam, sgd}
      - K = # of Kraus operators per site
      - batch_size
      - epochs
      - num_data = how many total examples from MNIST (0 or 1). If None, use entire dataset.
      - lr = learning rate
      - log_steps = number of steps between logging loss values

    Returns:
      A list 'all_losses' containing the loss after each batch (iteration).
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
            transforms.Lambda(lambda x: x.to(torch.float64)),  # double precision
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    # Filter digits 0,1 only
    trainset = filter_digits(trainset, allowed_digits=[0, 1])

    # If num_data is specified, slice the dataset
    if num_data is not None and num_data < len(trainset):
        trainset = torch.utils.data.Subset(trainset, range(num_data))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    # --------------------------
    # 2) Build MPSTPCP model
    # --------------------------
    N = img_size * img_size  # 256
    d = 2
    # Convert string manifold to ManifoldType enum
    manifold_map = {
        "EXACT": ManifoldType.EXACT,
        "FROBENIUS": ManifoldType.FROBENIUS,
        "CANONICAL": ManifoldType.CANONICAL,
    }
    if manifold.upper() not in manifold_map:
        raise ValueError(f"Invalid manifold={manifold}. Use EXACT/FROBENIUS/CANONICAL.")

    model = MPSTPCP(
        N=N,
        K=K,
        d=d,
        with_identity=False,  # or True, depending on your preference
        manifold=manifold_map[manifold.upper()],
    )
    model.train()

    # --------------------------
    # 3) Choose Riemannian optimizer
    # --------------------------
    if optimizer_name.lower() == "adam":
        optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    # --------------------------
    # 4) Training loop
    # --------------------------
    all_losses = []  # We'll store the batch-wise loss here.

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0
        start_time = time.time()

        for step, (data, target) in enumerate(trainloader):
            # data => shape [bs, 256, 2]
            # target => shape [bs]
            optimizer.zero_grad()

            outputs = model(data)  # forward pass

            # Quick check for NaNs
            if torch.isnan(outputs).any():
                print("NaN detected in outputs, skipping batch.")
                continue

            batch_loss = loss_batch(outputs, target)
            batch_loss.backward()
            optimizer.step()
            # model.proj_stiefel(check_on_manifold=True, print_log=True)
            # print(f"Epoch {epoch+1}, Batch {total_batches}, Iteration {total_batches}: Batch Loss = {batch_loss.item():.4f}")

            # Assert if the params are TPCP using geoopt's built-in function
            # for param in model.parameters():
            #     p = param.data.clone().reshape(K, d**2, d**2)
            #     I = torch.zeros(d**2, d**2, dtype=p.dtype, device=p.device)
            #     for i in range(K):
            #         I += p[i].T @ p[i]
            #     print(torch.linalg.norm(I - torch.eye(d**2, dtype=p.dtype, device=p.device)))
            # if not torch.allclose(I, torch.eye(d**2, dtype=p.dtype, device=p.device)):
            #     print(f"Parameter is not TPCP at step {step}.")
            #     print(p)
            #     raise ValueError("Parameter is not TPCP.")

            # Record batch-wise loss for plotting
            all_losses.append(batch_loss.item())

            if torch.isnan(batch_loss).any():
                print("NaN detected in batch_loss, skipping batch.")
                continue
                
            model.proj_stiefel(check_on_manifold=True, print_log=False, rtol = 1e-3)

            # Compute stats
            epoch_loss += batch_loss.item()
            batch_acc = calculate_accuracy(outputs.detach(), target)
            epoch_acc += batch_acc.item()
            total_batches += 1

            # Log loss values at each log_steps
            if (step + 1) % log_steps == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch {epoch+1:02}/{epochs:02} | Step {step+1:03}/{len(trainloader):03} | Loss: {batch_loss.item():.6f} | Elapsed Time: {elapsed_time:.2f}s"
                )
                start_time = time.time()

        # Print epoch-level stats
        if total_batches > 0:
            avg_loss = epoch_loss / total_batches
            avg_acc = epoch_acc / total_batches
        else:
            avg_loss = float("nan")
            avg_acc = float("nan")

        print(
            f"Epoch {epoch+1:02}/{epochs:02} | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2%}"
        )

    return all_losses


def main():
    parser = argparse.ArgumentParser(
        description="Train TPCP MPS on MNIST (digits 0 & 1)."
    )

    parser.add_argument(
        "--manifold",
        default="EXACT",
        choices=["EXACT", "FROBENIUS", "CANONICAL"],
        help="Manifold type for Kraus ops (default EXACT).",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd"],
        help="Which Riemannian optimizer to use (default Adam).",
    )
    parser.add_argument(
        "--K", type=int, default=1, help="# of Kraus operators per site (default 1)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (default 128).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train (default 10)."
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=None,
        help="Limit the dataset to N examples (default: use entire dataset).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default 0.01)."
    )

    # New: user-provided random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None => system-based).",
    )

    # New: log_steps argument
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Number of steps between logging loss values (default 100).",
    )

    args = parser.parse_args()

    # -------------------------------------------
    # If user provides a seed, fix it
    # else we do nothing => system-based randomness
    # -------------------------------------------
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using fixed random seed: {args.seed}")
    else:
        print("Using system-based (time) random seed.")

    iteration_losses = train_tpcp_mnist(
        manifold=args.manifold,
        optimizer_name=args.optimizer,
        K=args.K,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_data=args.num_data,
        lr=args.lr,
        log_steps=args.log_steps,
    )

    # --------------------------
    # Plot the loss curve
    # --------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(iteration_losses, label="Training Loss")
    plt.xlabel("Iteration (batch)")
    plt.ylabel("Loss")
    plt.title(
        f"Loss Curve ({args.manifold} - {args.optimizer.upper()}, K={args.K}, seed={args.seed})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
