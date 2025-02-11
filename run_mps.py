#!/usr/bin/env python3

import argparse
import time
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys

# If your mps/ package is local, ensure it's on sys.path or installed in editable mode:
# sys.path.append("/path/to/your/project/root")

from mps import umps
from mps import unitary_optimizer


###############################################################################
# Dataset utilities (same as your first snippet)
###############################################################################
def filiter_single_channel(batch: torch.Tensor) -> torch.Tensor:
    """
    MNIST is loaded as shape [C, H, W].
    Take only the first channel => shape [H, W].
    """
    return batch[0, ...]


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


def filter_dataset(dataset, allowed_digits=[0, 1]):
    """Return a subset of MNIST dataset containing only allowed_digits (0 or 1)."""
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in allowed_digits:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)


###############################################################################
# Loss & Accuracy (same as your first snippet)
###############################################################################
def loss_batch(outputs, labels):
    """
    Binary cross-entropy style loss for outputs in [0, 1].
    For label=0 => prob=outputs[i], else => 1 - outputs[i].
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else 1 - outputs[i]
        loss -= torch.log(prob + 1e-8)
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
def train_umps_mnist(
    epochs: int,
    batch_size: int,
    lr: float,
    log_steps: int = 100,
    seed: int | None = None,
):
    """
    Train an uMPS model on MNIST (digits 0 and 1) with the given parameters:

      - epochs
      - batch_size
      - lr (learning rate)
      - log_steps (how many steps between logging loss values)
      - seed (random seed)

    Returns:
      A list 'all_losses' containing the loss after each batch (iteration).
    """
    # -------------------------------------------
    # 1) If user provides a seed, fix it
    # -------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using fixed random seed: {seed}")
    else:
        print("Using system-based (time) random seed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # -------------------------------------------
    # 2) Prepare MNIST dataset
    # -------------------------------------------
    img_size = 16
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(filiter_single_channel),
            transforms.Lambda(embedding_pixel),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    # Keep only digits 0 and 1
    trainset = filter_dataset(trainset, allowed_digits=[0, 1])

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
    )

    # -------------------------------------------
    # 3) Build the uMPS model and optimizer
    # -------------------------------------------
    N = img_size * img_size  # 256
    d = 2
    chi = 2  # from your snippet
    l = 2    # from your snippet
    layers = 1  # from your snippet

    umpsm = umps.uMPS(N=N, chi=chi, d=d, l=l, layers=layers, device=device)
    umpsm_op = unitary_optimizer.Adam(umpsm, lr=lr)

    # -------------------------------------------
    # 4) Training loop
    # -------------------------------------------
    all_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0
        start_time = time.time()

        for step, (data, target) in enumerate(trainloader):
            # data shape => [bs, 2, 16, 16] after filiter_single_channel? Actually:
            #   filiter_single_channel => [bs, H, W], 
            #   embedding_pixel => [bs, H*W, 2].
            # We want the shape [site, batch, features], so permute:
            data = data.to(device).permute(1, 0, 2)
            target = target.to(device)

            umpsm_op.zero_grad()
            outputs = umpsm(data)  # forward pass => shape [batch_size]

            batch_loss = loss_batch(outputs, target)
            batch_loss.backward()
            umpsm_op.step()

            umpsm.proj_unitary(check_on_unitary=True, print_log=False, rtol=1e-3)

            all_losses.append(batch_loss.item())
            epoch_loss += batch_loss.item()


            # Calculate accuracy
            batch_acc = calculate_accuracy(outputs.detach(), target)
            epoch_acc += batch_acc.item()

            total_batches += 1

            # Log every log_steps
            if (step + 1) % log_steps == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch {epoch+1:02}/{epochs:02} | "
                    f"Step {step+1:03}/{len(trainloader):03} | "
                    f"Loss: {batch_loss.item():.6f} | "
                    f"Elapsed: {elapsed_time:.2f}s"
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
            f"Epoch {epoch+1:02}/{epochs:02} | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"Avg Acc: {avg_acc:.2%}"
        )

    return all_losses


###############################################################################
# main() with argument parsing
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Train an uMPS model on MNIST (digits 0 & 1) with code style similar to the second snippet."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train (default 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (default 128).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default 0.01).",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Number of steps between logging loss values (default 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None => system-based).",
    )

    args = parser.parse_args()

    # Train the model
    all_losses = train_umps_mnist(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_steps=args.log_steps,
        seed=args.seed,
    )

    # Optionally, you can do additional things here, like plotting:
    # import matplotlib.pyplot as plt
    # plt.plot(all_losses)
    # plt.xlabel("Iteration (batch)")
    # plt.ylabel("Loss")
    # plt.title("Training Loss Curve (uMPS MNIST 0/1)")
    # plt.show()


if __name__ == "__main__":
    main()
