#!/usr/bin/env python3
import argparse
import time
import torch
import numpy as np

from mps.trainer.mps_trainer import umps_train
from mps.trainer.data_utils import create_mnist_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train uMPS on MNIST (digits 0 & 1).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default 10).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default 128).")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01).")
    parser.add_argument("--log_steps", type=int, default=10, help="Log steps (default 10).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    parser.add_argument("--num_data", type=int, default=None, help="Number of samples (optional).")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using fixed seed: {args.seed}")
    else:
        print("Using system-based randomness.")

    img_size = 16

    # Create the MNIST DataLoader using the utility function from data_utils.
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
    
    # uMPS parameters: N=img_size*img_size, chi=2, d=2, l=2, layers=1
    _ = umps_train(
        dataloader=dataloader,
        device=device,
        N=img_size * img_size,
        chi=2,
        d=2,
        l=2,
        layers=1,
        epochs=args.epochs,
        lr=args.lr,
        log_steps=args.log_steps,
        dtype=torch.float64,
    )

if __name__ == "__main__":
    main()
