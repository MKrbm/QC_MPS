#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import geoopt
import matplotlib.pyplot as plt

# Adjust the imports for your local environment if needed
from mps.tpcp_mps import MPSTPCP, ManifoldType


###############################################################################
# 1) MNIST Preprocessing (digits 0 and 1), single-channel extraction, embedding
###############################################################################
def filter_digits(dataset, allowed_digits=[0, 1]):
    """Keep only samples whose labels are in allowed_digits."""
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in allowed_digits:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)

def single_channel(batch):
    """Extract only the first channel => shape [H, W]."""
    return batch[0, ...]

def embedding_pixel(batch):
    """
    Flatten image (H, W)->(H*W). 
    Convert each pixel x -> [x, 1 - x], and normalize in the last dimension.
    Result shape: [H*W, 2].
    """
    x = batch.view(-1)  # Flatten
    x = torch.stack([x, 1 - x], dim=-1)
    x = x / torch.norm(x, dim=-1, keepdim=True)  # normalize
    return x

def get_single_mnist_batch(batch_size=64, img_size=16):
    """
    Returns a single batch (data, labels) from MNIST (digits 0,1) 
    with shape data: [batch_size, N, 2]  (where N=img_size^2)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(single_channel),
        transforms.Lambda(embedding_pixel),
        transforms.Lambda(lambda x: x.to(torch.float64))  # Change dtype to float64
    ])

    # Load MNIST train
    mnist_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    # Filter for digits 0,1 only
    mnist_data = filter_digits(mnist_data, allowed_digits=[0, 1])

    # Create DataLoader with the full subset but we only need one batch
    dataloader = torch.utils.data.DataLoader(
        mnist_data,
        batch_size=batch_size,
        shuffle=True
    )

    data_iter = iter(dataloader)
    data, labels = next(data_iter)  # one batch

    # data => shape [bs, H*W, 2], labels => shape [bs]
    # We'll confirm that shape matches expectation: 
    # data shape = [batch_size, 16*16, 2], e.g. [64, 256, 2]
    # labels shape = [batch_size]
    return data, labels


###############################################################################
# 2) Loss / Accuracy Functions
###############################################################################
def loss_batch(outputs, labels):
    """
    Binary cross-entropy style loss for outputs in [0, 1].
    For label=0 => prob=outputs[i], else => 1 - outputs[i].
    """
    loss = torch.zeros(1, device=outputs.device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss -= torch.log(prob + 1e-8)
    return loss


###############################################################################
# 3) Experiment Setup: Compare manifold types & optimizers on one MNIST batch
###############################################################################
def run_optimization(data, labels, manifold_type, optimizer_type, steps=20, lr=0.01):
    """
    Runs an optimization loop on a single batch of data with MPSTPCP:
      - manifold_type in {EXACT, FROBENIUS, CANONICAL}
      - optimizer_type in {"SGD", "Adam"}
      - steps = number of optimization steps
      - lr = learning rate
    Returns a list of loss values for each step.
    """
    N = data.shape[1]  # e.g. 256
    d = data.shape[2]  # 2
    # Create MPSTPCP model
    model = MPSTPCP(
        N=N,
        K=1,
        d=d,
        with_identity=False,
        manifold=manifold_type
    )
    model.train()

    # Choose Riemannian optimizer
    if optimizer_type.lower() == "sgd":
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "adam":
        optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(data)          # shape [batch_size]
        loss = loss_batch(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Optionally: check manifold constraints for EXACT
        # if manifold_type == ManifoldType.EXACT:
        #     for kraus_op in model.kraus_ops:
        #         check_eye = kraus_op.detach().T @ kraus_op.detach()
        #         I = torch.eye(check_eye.shape[0], dtype=check_eye.dtype)
        #         assert torch.allclose(check_eye, I, atol=1e-7), \
        #             "Kraus op drifted off the EXACT Stiefel manifold"

    return loss_history


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # We get one batch from MNIST (digits 0,1)
    # shape: data => [batch_size, 256, 2], labels => [batch_size]
    data, labels = get_single_mnist_batch(batch_size=100, img_size=16)

    # We'll compare:
    # - EXACT, FROBENIUS, CANONICAL
    # - SGD, Adam
    manifold_dict = {
        "EXACT": ManifoldType.EXACT,
        "FROBENIUS": ManifoldType.FROBENIUS,
        "CANONICAL": ManifoldType.CANONICAL
    }
    optimizers = ["SGD", "Adam"]
    steps = 300
    lr = 0.005

    # Prepare a figure for all curves
    plt.figure(figsize=(10, 6))

    for m_name, m_type in manifold_dict.items():
        for opt in optimizers:
            print(f"Running {opt} on manifold={m_name} with single MNIST batch...")
            losses = run_optimization(
                data,
                labels,
                manifold_type=m_type,
                optimizer_type=opt,
                steps=steps,
                lr=lr
            )
            plt.plot(losses, label=f"{m_name} - {opt}")

    plt.title("Loss Curves for Different Manifolds and Optimizers")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
