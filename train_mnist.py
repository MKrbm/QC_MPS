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

def main():
    # Create the top-level parser.
    parser = argparse.ArgumentParser(
        description="Train a selected model (tpcp, umps, or mpsae) on MNIST (digits 0 & 1) / synthetic data."
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

    # Use subparsers to separate model-specific arguments.
    subparsers = parser.add_subparsers(
        dest="model",
        required=True,
        help="Select the model to train: 'tpcp', 'umps', or 'mpsae'.",
    )

    # ----- Subparser for TPCP training -----
    tpcp_parser = subparsers.add_parser(
        "tpcp", help="Train TPCP MPS on MNIST (digits 0 & 1)."
    )
    # Common TPCP parameters.
    # Plain training mode arguments.
    tpcp_parser.add_argument(
        "--manifold",
        type=str,
        default="Exact",
        choices=["Exact", "Frobenius", "Canonical", "Cayley", "MatrixExp", "ForwardEuler", "Original"],
        help="Manifold type (default: Exact; used in plain training mode only)."
    )
    tpcp_parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer (default: adam; used in plain training mode only)."
    )
    tpcp_parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="Number of Kraus operators (default: 1; used in plain training mode only)."
    )

    # ----- Subparser for UMPS training -----
    umps_parser = subparsers.add_parser(
        "umps", help="Train uMPS on MNIST (digits 0 & 1)."
    )
    umps_parser.add_argument(
        "--check_bp",
        action="store_true",
        help="Check BP training (default: False)."
    )
    

    # ----- Subparser for MPSAE training -----
    mpsae_parser = subparsers.add_parser(
        "mpsae", help="Train MPS Adiabatic Encoder (MPSAE) on synthetic data."
    )
    mpsae_parser.add_argument(
        "--n", type=int, default=256, help="Input vector dimension (default: 256)."
    )
    mpsae_parser.add_argument(
        "--simple_epochs",
        type=int,
        default=1,
        help="Epochs for SimpleMPS training (default: 1).",
    )
    mpsae_parser.add_argument(
        "--min_epochs",
        type=int,
        default=3,
        help="Min epochs before convergence (default: 3).",
    )
    mpsae_parser.add_argument(
        "--patience", type=int, default=2, help="Patience for convergence (default: 2)."
    )
    mpsae_parser.add_argument(
        "--conv_strategy",
        type=str,
        default="absolute",
        choices=["absolute", "relative"],
        help="Convergence strategy (default: absolute).",
    )
    mpsae_parser.add_argument(
        "--conv_threshold",
        type=float,
        default=1e-4,
        help="Convergence threshold (default: 1e-4).",
    )
    mpsae_parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="Number of Kraus operators per site (default: 1).",
    )
    mpsae_parser.add_argument(
        "--manifold",
        type=str,
        default="Exact",
        choices=[
            "Exact",
            "Frobenius",
            "Canonical",
            "Cayley",
            "MatrixExp",
            "ForwardEuler",
            "Original",
        ],
        help="Manifold type (default: Exact).",
    )
    mpsae_parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer (default: adam).",
    )
    mpsae_parser.add_argument(
        "--simple_lr",
        type=float,
        default=0.001,
        help="Learning rate for SimpleMPS training (default: 0.001).",
    )
    # Adaptive mode arguments.
    mpsae_parser.add_argument(
        "--mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "plain"],
        help="MPSAE training mode: 'adaptive' for adaptive lambda scheduling (default) or 'plain' for plain MPSAE training."
    )

    mpsae_parser.add_argument(
        "--total_schedule_steps",
        type=int,
        default=10,
        help="Total number of lambda schedule steps (default: 10)."
    )
    mpsae_parser.add_argument(
        "--schedule_type",
        type=str,
        default="cosine",
        choices=["linear", "polynomial", "soft_exponential", "cosine"],
        help="Type of lambda schedule (default: cosine)."
    )


    # Parse all the arguments.
    args = parser.parse_args()

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
        allowed_digits=[0, 1],
        img_size=img_size,
        root="data",
        train=True,
        download=True,
        batch_size=args.batch_size,
        num_data=args.num_data,
    )

    # Call the appropriate training function based on the model.
    if args.model == "tpcp":
        # Plain TPCP training mode:
        _ = tpcp_train(
            dataloader=dataloader,
            device=device,
            N=img_size * img_size,
            d=2,
            K=args.K,
            with_identity=False,  # adjust as desired.
            manifold=args.manifold,
            optimizer_name=args.optimizer,
            epochs=args.epochs,
            lr=args.lr,
            log_steps=args.log_steps,
            dtype=torch.float64,
        )
    elif args.model == "umps":
        if args.check_bp:
            _ = umps_bp_train(
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
        else:
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
    elif args.model == "mpsae":
        if args.mode == "adaptive":
            # Use the adaptive lambda training function for MPSAE.
            _ = mpsae_adaptive_train(
                dataloader=dataloader,
                device=device,
                N=args.n,
                mps_epochs=args.simple_epochs,
                mps_lr=args.simple_lr,
                mps_optimize="greedy",
                manifold=args.manifold,
                K=args.K,
                max_epochs=args.epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                conv_strategy=args.conv_strategy,
                conv_threshold=args.conv_threshold,
                lr=args.lr,
                log_steps=args.log_steps,
                dtype=torch.float64,
                total_schedule_steps=args.total_schedule_steps,
                schedule_type=args.schedule_type,  # Choose among 'linear', 'polynomial', 'soft_exponential', 'cosine'.
                poly_power=2,
                k=0.1,
                spike_threshold=0.1,
            )
        else:
            _ = mpsae_train(
                dataloader=dataloader,
                device=device,
                N=args.n,
                d=2,
                l=2,
                mps_epochs=args.simple_epochs,
                mps_lr=args.simple_lr,
                mps_optimize="greedy",
                manifold=args.manifold,
                optimizer_name=args.optimizer,
                K=args.K,
                max_epochs=args.epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                conv_strategy=args.conv_strategy,
                conv_threshold=args.conv_threshold,
                lr=args.lr,
                log_steps=args.log_steps,
                weight_values=None,
                dtype=torch.float64,
            )
    else:
        raise ValueError("Unknown model type specified.")


if __name__ == "__main__":
    main()
