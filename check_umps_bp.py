#!/usr/bin/env python
import os
import time
import random
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt  # Ensure matplotlib is imported

from mps import umps
from mps import unitary_optimizer
from mps.trainer.utils import plot_training_metrics, loss_batch, calculate_accuracy, plot_gradient_norms
from mps.trainer.data_utils import create_mnist_dataloader

def umps_bp_train(
    dataloader,
    device,
    N,
    chi,
    d=2,
    l=2,
    layers=1,
    epochs=10,
    lr=0.01,
    log_steps=10,
    dtype=torch.float64,
):
    """
    Train a unitary MPS (uMPS) model on a given dataloader.

    Returns:
      - A dictionary with the following keys:
          "all_losses": list of loss values (per batch)
          "epoch_losses": average loss per epoch,
          "epoch_accs": average accuracy per epoch,
          "gradients": list of average gradient norms for specific qubits per epoch.
    """
    # Build the uMPS model and its optimizer.
    model = umps.uMPS(N=N, chi=chi, d=d, l=l, layers=layers, device=device)
    optimizer = unitary_optimizer.Adam(model, lr=lr)
    all_losses = []      # Store loss for each batch.
    epoch_losses = []    # Store average loss per epoch.
    epoch_accs = []      # Store average accuracy per epoch.

    # List to store average gradient norms per epoch for specific qubits.
    gradients = {i: [] for i in range(0, N, N // 10)}

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0

        # Accumulate gradient norms for the epoch.
        grad_epoch = {i: 0.0 for i in range(0, N, N // 10)}

        t0 = time.time()

        for step, (data, target) in enumerate(dataloader):
            # Data comes in as [batch, H*W, 2] (after embedding). We need [site, batch, features].
            data = data.to(device).permute(1, 0, 2)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)  # Forward pass.
            batch_loss = loss_batch(outputs, target)
            batch_loss.backward()

            # --- Record gradient norms for specific qubits ---
            for i in range(0, N, N // 10):
                if model.params[i].grad is not None:
                    g = model.params[i].grad.data.reshape(4, 4)
                    u = model.params[i].data.reshape(4, 4)
                    rg = unitary_optimizer.riemannian_gradient(u, g)
                    grad_epoch[i] += rg.norm().item()
                else:
                    grad_epoch[i] += 0.0
            # ---------------------------------------------------

            optimizer.step()
            # Project back onto unitary (if needed).
            model.proj_unitary(check_on_unitary=True, print_log=False, rtol=1e-3)

            all_losses.append(batch_loss.item())
            epoch_loss += batch_loss.item()

            batch_acc = calculate_accuracy(outputs.detach(), target)
            epoch_acc += batch_acc.item()
            total_batches += 1

            if (step + 1) % log_steps == 0:
                elapsed = time.time() - t0
                loss_per_data = batch_loss.item() / data.shape[0]
                print(f"Epoch {epoch+1:02} | Step {step+1:03}/{len(dataloader):03} | "
                      f"Loss: {loss_per_data:.6f} | Acc: {batch_acc.item():.2%} | Time: {elapsed:.2f}s")
                t0 = time.time()

        # Compute average metrics for this epoch.
        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_acc = epoch_acc / total_batches if total_batches > 0 else float("nan")
        avg_grad = {i: grad_epoch[i] / total_batches for i in grad_epoch}

        print(f"Epoch {epoch+1:02} | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2%} | "
              f"Gradients -> " + ", ".join([f"{i}th: {avg_grad[i]:.6e}" for i in avg_grad]))

        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
        for i in avg_grad:
            gradients[i].append(avg_grad[i])

    return {
        "all_losses": all_losses,
        "epoch_losses": epoch_losses,
        "epoch_accs": epoch_accs,
        "gradients": gradients,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="uMPS training")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run")
    args = parser.parse_args()
    seed = args.seed
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----- Setup experiment parameters -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 16 * 16       # Flattened 16x16 images.
    chi = 2           # Bond dimension.
    epochs = 200      # Number of epochs.
    lr = 0.0001      # Learning rate.
    log_steps = 10    # Logging frequency.

    # Create folder for CSV metrics.
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    print("Metrics directory created.")

    dataloader = create_mnist_dataloader(
        allowed_digits=[3, 8],
        img_size=16,
        root="data",
        train=True,
        download=True,
        batch_size=128,
        num_data=None,
    )

    # Run one training session.
    metrics = umps_bp_train(
        dataloader=dataloader,
        device=device,
        N=N,
        chi=chi,
        d=2,
        l=2,
        layers=1,
        epochs=epochs,
        lr=lr,
        log_steps=log_steps,
        dtype=torch.float64,
    )

    # Save epoch metrics CSV.
    epoch_df = pd.DataFrame({
        "Epoch": list(range(1, len(metrics["epoch_losses"]) + 1)),
        "Avg_Loss": metrics["epoch_losses"],
        "Avg_Accuracy": metrics["epoch_accs"],
    })
    epoch_filename = os.path.join(metrics_dir, f"run_{seed:04d}_epoch_metrics.csv")
    epoch_df.to_csv(epoch_filename, index=False)

    # Save batch losses CSV.
    batch_df = pd.DataFrame({"Batch_Loss": metrics["all_losses"]})
    batch_filename = os.path.join(metrics_dir, f"run_{seed:04d}_batch_losses.csv")
    batch_df.to_csv(batch_filename, index=False)

    # Save gradients CSVs.
    for i in metrics["gradients"]:
        grad_df = pd.DataFrame({
            "Epoch": list(range(1, len(metrics["gradients"][i]) + 1)),
            f"Grad_{i}": metrics["gradients"][i],
        })
        grad_filename = os.path.join(metrics_dir, f"run_{seed:04d}_grad_{i}.csv")
        grad_df.to_csv(grad_filename, index=False)

    print(f"Run with seed {seed} metrics saved in folder '{metrics_dir}'.")
