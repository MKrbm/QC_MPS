#!/usr/bin/env python3
"""
Module for model-specific argument parsing and training.

This module defines a function to parse additional arguments for model selection
and then calls the corresponding training function (tpcp, umps, or mpsae).
"""

import argparse
import torch

# Import training functions.
from mps.trainer.tpcp_trainer import tpcp_train
from mps.trainer.mps_trainer import umps_train
from mps.trainer.mpsae_trainer import mpsae_train
from mps.trainer.adaptive_mpsae_trainer import mpsae_adaptive_train

def run_training(mnist_args, remaining_args, dataloader, N):
    """
    Parse model-specific arguments and run the selected model's training.
    
    Parameters:
      mnist_args     -- Namespace containing MNIST-related options.
      remaining_args -- List of remaining command-line arguments.
      dataloader     -- The MNIST DataLoader.
      N              -- Number of features.
    """
    parser = argparse.ArgumentParser(
        description="Model-specific arguments",
        add_help=True
    )
    subparsers = parser.add_subparsers(
        dest="model",
        required=True,
        help="Select the model to train: 'tpcp', 'umps', or 'mpsae'."
    )

    # ----- TPCP Subparser -----
    tpcp_parser = subparsers.add_parser("tpcp", 
        help="Train TPCP MPS on Dataset")
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

    # ----- UMPS Subparser -----
    umps_parser = subparsers.add_parser("umps", 
        help="Train uMPS on MNIST (digits 0 & 1).")
    # (No extra arguments for UMPS at the moment.)

    # ----- MPSAE Subparser -----
    mpsae_parser = subparsers.add_parser("mpsae", 
        help="Train MPS Adiabatic Encoder (MPSAE) on synthetic data.")
    mpsae_parser.add_argument(
        "--n", type=int, default=256, 
        help="Input vector dimension (default: 256)."
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
        help="Minimum epochs before convergence (default: 3).",
    )
    mpsae_parser.add_argument(
        "--patience", type=int, default=2, 
        help="Patience for convergence (default: 2)."
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
        choices=["Exact", "Frobenius", "Canonical", "Cayley", "MatrixExp", "ForwardEuler", "Original"],
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
    mpsae_parser.add_argument(
        "--mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "plain"],
        help="MPSAE training mode: 'adaptive' (default) or 'plain'."
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

    # Parse model-specific arguments.
    model_args = parser.parse_args(remaining_args)

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call the appropriate training function based on the selected model.
    if model_args.model == "tpcp":
        _ = tpcp_train(
            dataloader=dataloader,
            device=device,
            N=N,
            d=2,
            K=model_args.K,
            with_identity=False,  # adjust as desired.
            manifold=model_args.manifold,
            optimizer_name=model_args.optimizer,
            epochs=mnist_args.epochs,
            lr=mnist_args.lr,
            log_steps=mnist_args.log_steps,
            dtype=torch.float64,
        )
    elif model_args.model == "umps":
        _ = umps_train(
            dataloader=dataloader,
            device=device,
            N=N,
            chi=2,
            d=2,
            l=2,
            layers=1,
            epochs=mnist_args.epochs,
            lr=mnist_args.lr,
            log_steps=mnist_args.log_steps,
            dtype=torch.float64,
        )
    elif model_args.model == "mpsae":
        if model_args.mode == "adaptive":
            _ = mpsae_adaptive_train(
                dataloader=dataloader,
                device=device,
                N=N,
                mps_epochs=model_args.simple_epochs,
                mps_lr=model_args.simple_lr,
                mps_optimize="greedy",
                manifold=model_args.manifold,
                K=model_args.K,
                max_epochs=mnist_args.epochs,
                min_epochs=model_args.min_epochs,
                patience=model_args.patience,
                conv_strategy=model_args.conv_strategy,
                conv_threshold=model_args.conv_threshold,
                lr=mnist_args.lr,
                log_steps=mnist_args.log_steps,
                dtype=torch.float64,
                total_schedule_steps=model_args.total_schedule_steps,
                schedule_type=model_args.schedule_type,
                poly_power=2,
                k=0.1,
                spike_threshold=0.1,
            )
        else:
            _ = mpsae_train(
                dataloader=dataloader,
                device=device,
                N=N,
                d=2,
                l=2,
                mps_epochs=model_args.simple_epochs,
                mps_lr=model_args.simple_lr,
                mps_optimize="greedy",
                manifold=model_args.manifold,
                optimizer_name=model_args.optimizer,
                K=model_args.K,
                max_epochs=mnist_args.epochs,
                min_epochs=model_args.min_epochs,
                patience=model_args.patience,
                conv_strategy=model_args.conv_strategy,
                conv_threshold=model_args.conv_threshold,
                lr=mnist_args.lr,
                log_steps=mnist_args.log_steps,
                weight_values=None,
                dtype=torch.float64,
            )
    else:
        raise ValueError("Unknown model type specified.")
