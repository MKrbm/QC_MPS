#!/usr/bin/env python3
"""
Example usage:

python train_mnist.py \
    --model mpsae \
    --simple_epochs 5 \
    --epochs 20 \
    --lr 0.0001 \
    --simple_lr 0.0001

    
python train_mnist.py \
    --epochs 10 \
    --seed 2024 \
    --lr 0.0001 \
    umps

python train_mnist.py \
    --epochs 10 \
    --seed 2024 \
    --lr 0.01 \
    tpcp \
    --manifold Original

python train_mnist.py \
    --epochs 10 \
    --seed 2024 \
    --lr 0.0001 \
    mpsae \
    --mode plain \
    --manifold Original \
    --conv_strategy relative \
    --conv_threshold 1e-2 \
    --simple_epochs 1 \
    --simple_lr 0.001

python train_mnist.py \
    --epochs 10 \
    --seed 2024 \
    --lr 0.0001 \
    mpsae \
    --mode adaptive \
    --manifold Original \
    --conv_strategy relative \
    --conv_threshold 1e-2 \
    --total_schedule_steps 20 \
    --schedule_type cosine \
    --simple_epochs 1 \
    --simple_lr 0.001 \
    --min_epochs 10

"""
import argparse
import torch
import numpy as np

# Import dataloader and other training functions.
from mps.trainer.data_utils import create_mnist_dataloader
from mps.trainer.mps_trainer import umps_train
from mps.trainer.umps_bp_trainer import umps_bp_train
from mps.trainer.mpsae_trainer import mpsae_train
from mps.trainer.tpcp_trainer import tpcp_train
from mps.trainer.adaptive_mpsae_trainer import mpsae_adaptive_train
from mps.trainer.model_trainer import run_training

def main():
    # Create the top-level parser.
    parser = argparse.ArgumentParser(
        description="Train a selected model on MNIST (digits 0 & 1) / synthetic data."
    )

    # Common arguments for all models.
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: 128)."
    )
    parser.add_argument(
        "--log_steps", type=int, default=10, help="Log steps (default: 10)."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (optional)."
    )
    parser.add_argument(
        "--num_data", type=int, default=None, help="Number of samples (optional)."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs (default: 10). If the model is adaptive, this will be the maximum number of epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)."
    )

    # Parse common arguments and leave remaining args for model-specific options.
    args, remaining_args = parser.parse_known_args()

    # Set random seeds if provided.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using fixed seed: {args.seed}")
    else:
        print("Using system-based randomness.")

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a MNIST DataLoader for digits 0 and 1.
    img_size = 16
    dataloader = create_mnist_dataloader(
        allowed_digits=[3, 8],
        img_size=img_size,
        root="data",
        train=True,
        download=True,
        batch_size=args.batch_size,
        num_data=args.num_data,
    )

    # Delegate the model-specific training to `run_training`.
    run_training(args, remaining_args, dataloader, N=img_size * img_size)

if __name__ == "__main__":
    main()
