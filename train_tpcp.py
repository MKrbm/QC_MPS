#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import time

from mps.trainer.tpcp_trainer import tpcp_train
from mps.trainer.data_utils import create_mnist_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train TPCP MPS on MNIST (digits 0 & 1).")
    parser.add_argument("--manifold", type=str, default="Exact", choices=["Exact", "Frobenius", "Canonical", "Cayley", "MatrixExp", "ForwardEuler"], help="Manifold type (default Exact).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer (default adam).")
    parser.add_argument("--K", type=int, default=1, help="Number of Kraus operators (default 1).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default 128).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default 10).")
    parser.add_argument("--num_data", type=int, default=None, help="Number of samples (optional).")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01).")
    parser.add_argument("--log_steps", type=int, default=10, help="Log steps (default 10).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using fixed seed: {args.seed}")
    else:
        print("Using system-based randomness.")

    img_size = 16

    # Create DataLoader using the utility function from data_utils.
    dataloader = create_mnist_dataloader(
        allowed_digits=[0, 1],
        img_size=img_size,
        root="data",
        train=True,
        download=True,
        batch_size=args.batch_size,
        num_data=args.num_data
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _ = tpcp_train(
        dataloader=dataloader,
        device=device,
        N=img_size * img_size,
        d=2,
        K=args.K,
        with_identity=False,  # adjust as desired
        manifold=args.manifold,
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        lr=args.lr,
        log_steps=args.log_steps,
        dtype=torch.float64,
    )

if __name__ == "__main__":
    main()
