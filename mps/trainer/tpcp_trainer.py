#!/usr/bin/env python3
"""
Module for training a TPCP MPS model.
Defines the tpcp_train() function.
"""

import time
import torch
import geoopt

from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps.StiefelOptimizers import StiefelAdam, StiefelSGD
from mps.radam import RiemannianAdam

from mps.trainer.utils import loss_batch, calculate_accuracy, plot_training_metrics

def tpcp_train(
    dataloader,
    device,
    N,
    d=2,
    K=1,
    with_identity=True,
    manifold="Exact",
    optimizer_name="adam",
    epochs=10,
    lr=0.01,
    log_steps=10,
    dtype=torch.float64,
):
    """
    Train a TPCP MPS model (for example, on MNIST).
    
    Returns a list of batch losses and displays training metrics.
    """
    manifold_map = {
        "Exact": ManifoldType.EXACT,
        "Frobenius": ManifoldType.FROBENIUS,
        "Canonical": ManifoldType.CANONICAL,
        "Cayley": ManifoldType.EXACT,
        "MatrixExp": ManifoldType.EXACT,
        "ForwardEuler": ManifoldType.EXACT,
        "Original": ManifoldType.EXACT,
    }
    if manifold not in manifold_map:
        raise ValueError(f"Invalid manifold '{manifold}'.")
    model = MPSTPCP(N=N, K=K, d=d, with_identity=with_identity, manifold=manifold_map[manifold])
    model.to(device)
    model.train()
    if manifold in ["Cayley", "MatrixExp", "ForwardEuler"]:
        if optimizer_name.lower() == "adam":
            optimizer = StiefelAdam(model.parameters(), lr=lr, expm_method=manifold)
        elif optimizer_name.lower() == "sgd":
            optimizer = StiefelSGD(model.parameters(), lr=lr, expm_method=manifold)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
    elif manifold in ["Exact", "Frobenius", "Canonical"]:
        if optimizer_name.lower() == "adam":
            optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, eps=1e-7)
        elif optimizer_name.lower() == "sgd":
            optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=lr)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
    elif manifold == "Original":
        if optimizer_name.lower() == "adam":
            optimizer = RiemannianAdam(model.parameters(), lr=lr)
        else:
            raise ValueError("SGD not supported for Original update rule.")

    all_losses = []      # List of loss for each batch.
    epoch_losses = []    # List to store average loss per epoch.
    epoch_accs = []      # List to store average accuracy per epoch.
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0
        t0 = time.time()
        for step, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            batch_loss = loss_batch(outputs, target)
            batch_loss.backward()
            optimizer.step()
            model.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)
            all_losses.append(batch_loss.item())
            epoch_loss += batch_loss.item()
            batch_acc = calculate_accuracy(outputs.detach(), target)
            epoch_acc += batch_acc.item()
            total_batches += 1
            if (step + 1) % log_steps == 0:
                elapsed = time.time() - t0
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(dataloader)} | Loss: {batch_loss.item():.6f} | Time: {elapsed:.2f}s")
                t0 = time.time()
        avg_loss = epoch_loss / total_batches if total_batches else float("nan")
        avg_acc = epoch_acc / total_batches if total_batches else float("nan")
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2%}")
        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
    
    # Use the utility function to plot training metrics.
    x_axis = list(range(1, epochs + 1))
    title = "TPCP Training Metrics"
    filename = "tpcp_training_metrics.png"  # Adjust the filename/path as needed.
    plot_training_metrics(x_axis, epoch_losses, epoch_accs, None, title, filename)
    
    return all_losses
