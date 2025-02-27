#!/usr/bin/env python3
"""

Run training on a synthetic dataset.

Example usage:

python train_toy.py \
    --epochs 10 \
    --seed 2024 \
    --lr 0.0001 \
    umps

    
python train_toy.py \
    --epochs 10 \
    --seed 2024 \
    --lr 0.0001 \
    mpsae \
    --mode plain \
    --manifold Original \
    --conv_strategy relative \
    --conv_threshold 1e-2 \
    --min_epochs 10 \
    --simple_epochs 3 \
    --simple_lr 0.00001

python train_toy.py \
    --epochs 100 \
    --seed 2024 \
    --lr 0.0001 \
    mpsae \
    --schedule_steps 15 \
    --mode adaptive \
    --manifold Exact \
    --conv_strategy relative \
    --conv_threshold 1e-2 \
    --min_epochs 90 \
    --simple_epochs 200 \
    --simple_lr 0.00005
"""

import argparse
import torch
import numpy as np
from mps.trainer.data_utils import SyntheticDataset
from mps.trainer.model_trainer import run_training

def main():
    # Parse MNIST-related arguments.
    parser = argparse.ArgumentParser(
        description="Train a model on MNIST digits 0 & 1. "
                    "This script handles MNIST data loading only."
    )
    parser.add_argument("--log_steps", type=int, default=10, 
                        help="Log steps (default: 10).")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed (optional).")
    parser.add_argument("--num_data", type=int, default=None, 
                        help="Number of samples (optional).")
    parser.add_argument("--N", type=int, default=256, 
                        help="Number of features (default: 256).")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs (default: 10).")
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="Learning rate (default: 0.01).")
    
    # Parse known args and leave remaining args for model-specific options.
    args, remaining_args = parser.parse_known_args()

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a MNIST DataLoader for digits 0 and 1.
    dataset = SyntheticDataset(n=args.N, num_samples=args.num_data or 1000, seed=args.seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Delegate the model-specific training (parsing its arguments) to the separate module.
    run_training(args, remaining_args, dataloader, N=args.N)

if __name__ == "__main__":
    main()
