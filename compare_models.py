#!/usr/bin/env python3
"""
Combined training script that simultaneously optimizes two equivalent models:
   - a TPCP–MPS model (MPSTPCP from mps.tpcp_mps) 
   - a uMPS model (from mps.umps)

The two models are initialized with exactly the same parameters using

    mpstpcp_model.kraus_ops.init_params()
    params = mpstpcp_model.kraus_ops.parameters()
    umps_model.initialize_MPS(torch.stack([p.reshape(2,2,2,2) for p in params]))

After training, their loss curves are compared and plotted.
"""

import argparse
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import geoopt

# Import the two model classes and the optimizer for uMPS:
from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps import umps
from mps import unitary_optimizer

###############################################################################
# Common dataset utilities
###############################################################################
def filter_digits(dataset, allowed_digits=[0, 1]):
    """Return a subset of MNIST containing only the allowed digits."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_digits]
    return torch.utils.data.Subset(dataset, indices)

def filiter_single_channel(img: torch.Tensor) -> torch.Tensor:
    """Take only the first channel of an image (assumes shape [C,H,W])."""
    return img[0, ...]

def embedding_pixel(batch: torch.Tensor) -> torch.Tensor:
    """
    Flatten each image from shape [H, W] to [H*W],
    then embed each pixel value x to a 2-dim vector [x, 1-x] and L2-normalize.
    """
    pixel_size = batch.shape[-1] * batch.shape[-2]
    x = batch.view(*batch.shape[:-2], pixel_size)
    x = torch.stack([x, 1 - x], dim=-1)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    return x

###############################################################################
# Loss and accuracy functions
###############################################################################
def loss_batch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy style loss. For a given output:
      - if label==0, use loss = -log(output)
      - if label==1, use loss = -log(1-output)
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss = loss - torch.log(prob + 1e-8)
    return loss

def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute binary accuracy based on a 0.5 threshold."""
    predictions = (outputs < 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / labels.numel()

###############################################################################
# Combined training function
###############################################################################
def train_both_models(args):
    """
    Train both models simultaneously on MNIST (digits 0 & 1). Both models will
    be initialized with the same starting parameters. For each minibatch, a forward
    pass is done for each model, their losses are computed (using the same loss function),
    gradients are backpropagated and parameters are updated.
    
    Returns:
       Two lists of loss values: one for the MPSTPCP model and one for the uMPS model.
    """
    # --------------------------------------------
    # Set device and random seed for reproducibility
    # --------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"[INFO] Using fixed random seed: {args.seed}")
    else:
        print("[INFO] Using system-based random seed.")

    # --------------------------------------------
    # Prepare the MNIST dataset (digits 0 & 1)
    # --------------------------------------------
    img_size = 16
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(filiter_single_channel),
        transforms.Lambda(embedding_pixel),
        # Ensure double precision (as used in MPSTPCP code)
        transforms.Lambda(lambda x: x.to(torch.float64)),
    ])

    trainset = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    trainset = filter_digits(trainset, allowed_digits=[0, 1])
    if args.num_data is not None and args.num_data < len(trainset):
        trainset = torch.utils.data.Subset(trainset, range(args.num_data))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True
    )
    print(f"[INFO] Using {len(trainset)} training examples.")

    # --------------------------------------------
    # Build both models.
    # For MPSTPCP, we use:
    #    N = number of sites = img_size * img_size,
    #    K = number of Kraus operators (args.K), and
    #    d = 2 (feature dimension).
    # For uMPS, we use: N, chi=2, d=2, l=2, layers=1.
    # --------------------------------------------
    N = img_size * img_size  # e.g. 256
    d = 2
    # Create the MPSTPCP model
    manifold_map = {
        "EXACT": ManifoldType.EXACT,
        "FROBENIUS": ManifoldType.FROBENIUS,
        "CANONICAL": ManifoldType.CANONICAL,
    }
    if args.manifold.upper() not in manifold_map:
        raise ValueError(f"Invalid manifold type: {args.manifold}.")
    mpstpcp_model = MPSTPCP(
        N=N,
        K=args.K,
        d=d,
        with_identity=False,  # or True, if desired
        manifold=manifold_map[args.manifold.upper()],
    )
    mpstpcp_model = mpstpcp_model.to(device)

    # Create the uMPS model with fixed parameters (chi, l, layers are set as in the snippet)
    chi = 2
    l = 2
    layers = 1
    umps_model = umps.uMPS(N=N, chi=chi, d=d, l=l, layers=layers, device=device)
    # Note: the uMPS model is assumed to be built so that its forward method expects input 
    # of shape [site, batch, feature_dim]. (We will permute the data accordingly.)

    # --------------------------------------------
    # Create the optimizers.
    # For the MPSTPCP model, use a Riemannian optimizer from geoopt.
    # For the uMPS model, use the provided unitary_optimizer.Adam.
    # --------------------------------------------
    if args.optimizer.lower() == "adam":
        mpstpcp_optimizer = geoopt.optim.RiemannianAdam(
            mpstpcp_model.parameters(), lr=args.lr
        )
    elif args.optimizer.lower() == "sgd":
        mpstpcp_optimizer = geoopt.optim.RiemannianSGD(
            mpstpcp_model.parameters(), lr=args.lr
        )
    else:
        raise ValueError("optimizer must be either 'adam' or 'sgd'")

    umps_optimizer = unitary_optimizer.Adam(umps_model, lr=args.lr)

    # --------------------------------------------
    # Synchronize the initial parameters:
    # Initialize MPSTPCP model's Kraus operators,
    # then get their parameters and use them to initialize uMPS.
    # --------------------------------------------
    mpstpcp_model.kraus_ops.init_params()
    # Collect parameters into a list.
    params = list(mpstpcp_model.kraus_ops.parameters())
    # Reshape each parameter to (2,2,2,2) and stack them.
    # (Make sure that the dimensions agree: each parameter in MPSTPCP must have 16 entries.)
    params_tensor = torch.stack([p.reshape(2, 2, 2, 2) for p in params])
    umps_model.initialize_MPS(params_tensor)

    # --------------------------------------------
    # Training loop (for both models simultaneously)
    # --------------------------------------------
    mpstpcp_losses = []
    umps_losses = []
    total_steps = 0

    for epoch in range(args.epochs):
        epoch_loss_mpstpcp = 0.0
        epoch_loss_umps = 0.0
        epoch_batches = 0
        start_time = time.time()

        for step, (data, target) in enumerate(trainloader):
            # data is of shape [batch, 256, 2]
            # For the MPSTPCP model, we use data as is.
            # For the uMPS model, we permute to shape [256, batch, 2].
            data = data.to(device)
            target = target.to(device)

            # Zero gradients for both optimizers.
            mpstpcp_optimizer.zero_grad()
            umps_optimizer.zero_grad()

            # Forward passes:
            outputs_mpstpcp = mpstpcp_model(data)  # Expected shape: [batch]
            outputs_umps = umps_model(data.permute(1, 0, 2))  # Expected shape: [batch]

            # Compute the same loss for both models.
            loss_mpstpcp = loss_batch(outputs_mpstpcp, target)
            loss_umps = loss_batch(outputs_umps, target)

            # Record loss values
            mpstpcp_losses.append(loss_mpstpcp.item())
            umps_losses.append(loss_umps.item())

            # Backward and optimizer step:
            loss_mpstpcp.backward()
            loss_umps.backward()
            mpstpcp_optimizer.step()
            umps_optimizer.step()

            # (Optional) re-project the parameters onto the manifolds.
            mpstpcp_model.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)
            umps_model.proj_unitary(check_on_unitary=True, print_log=False, rtol=1e-3)

            epoch_loss_mpstpcp += loss_mpstpcp.item()
            epoch_loss_umps += loss_umps.item()
            epoch_batches += 1
            total_steps += 1

            if (step + 1) % args.log_steps == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:02d} | Step {step+1:03d}/{len(trainloader):03d} | "
                      f"MPSTPCP Loss: {loss_mpstpcp.item():.6f} | uMPS Loss: {loss_umps.item():.6f} | "
                      f"Elapsed: {elapsed:.2f}s")
                
                # Calculate and print parameter differences
                param_diffs = []
                for p1, p2 in zip(mpstpcp_model.parameters(), umps_model.parameters()):
                    param_diffs.append(torch.norm(p1.reshape(4, 4) - p2.reshape(4, 4)).item())
                print(f"Parameter differences at step {step+1}: {np.mean(param_diffs)}")
                
                start_time = time.time()

        avg_loss_mpstpcp = epoch_loss_mpstpcp / epoch_batches if epoch_batches > 0 else float('nan')
        avg_loss_umps = epoch_loss_umps / epoch_batches if epoch_batches > 0 else float('nan')
        print(f"Epoch {epoch+1:02d} complete: Avg MPSTPCP Loss = {avg_loss_mpstpcp:.6f}, "
              f"Avg uMPS Loss = {avg_loss_umps:.6f}")

    return mpstpcp_losses, umps_losses

###############################################################################
# Main: argument parsing and training
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Simultaneous training of MPSTPCP and uMPS models on MNIST (digits 0 & 1)."
    )
    parser.add_argument("--manifold", type=str, default="EXACT",
                        choices=["EXACT", "FROBENIUS", "CANONICAL"],
                        help="Manifold type for Kraus ops (default: EXACT)")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd"],
                        help="Optimizer for MPSTPCP model (default: adam)")
    parser.add_argument("--K", type=int, default=1,
                        help="Number of Kraus operators per site for MPSTPCP (default: 1)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Steps between logging (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None => system-based)")
    parser.add_argument("--num_data", type=int, default=None,
                        help="Limit training to this many data points (default: all)")

    args = parser.parse_args()

    # Run the training for both models.
    mpstpcp_losses, umps_losses = train_both_models(args)

    # --------------------------------------------
    # Check if loss curves match (up to numerical tolerance)
    # --------------------------------------------
    mpstpcp_arr = np.array(mpstpcp_losses)
    umps_arr = np.array(umps_losses)
    if mpstpcp_arr.shape != umps_arr.shape:
        print("[WARNING] The two loss curves have different lengths!")
    else:
        if np.allclose(mpstpcp_arr, umps_arr, rtol=1e-7, atol=1e-7):
            print("[SUCCESS] Loss curves match exactly (within tolerance).")
        else:
            max_diff = np.abs(mpstpcp_arr - umps_arr).max()
            print(f"[INFO] Loss curves differ; maximum absolute difference = {max_diff:e}")

    # --------------------------------------------
    # Plot the loss curves for visual inspection.
    # --------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(mpstpcp_losses, label="MPSTPCP Loss")
    plt.plot(umps_losses, label="uMPS Loss", linestyle="--")
    plt.xlabel("Iteration (batch)")
    plt.ylabel("Loss")
    plt.title(f"Simultaneous Training Loss Curves\n(manifold={args.manifold}, optimizer={args.optimizer.upper()}, K={args.K}, seed={args.seed})")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
