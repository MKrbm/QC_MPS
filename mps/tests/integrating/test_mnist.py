import pytest
import torch
import random
import numpy as np

import geoopt
import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from mps import umps, unitary_optimizer
from mps.tpcp_mps import MPSTPCP, ManifoldType

# --------------------------
# 1) Utility functions
# --------------------------
def filiter_single_channel(batch: torch.Tensor) -> torch.Tensor:
    """
    Given an MNIST tensor shape [C, H, W], 
    keep only the first channel => shape [H, W].
    """
    return batch[0, ...]

def embedding_pixel(batch: torch.Tensor) -> torch.Tensor:
    """
    Flatten image (H, W) -> (H*W). 
    Create a 2D embedding as [ pixel, (1 - pixel) ] and normalize.
    """
    # shape: [H, W] -> flatten to [H*W]
    x = batch.view(-1)
    # shape: [H*W, 2]
    x = torch.stack([x, 1 - x], dim=-1)
    # Normalize along the last dim
    x = x / torch.norm(x, dim=-1, keepdim=True)
    return x

def filter_dataset(dataset, allowed_digits=[0, 1]):
    """
    Keep only the samples whose label is in `allowed_digits`.
    """
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in allowed_digits:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)

def loss_batch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy style loss for outputs in [0, 1].
    For label=0 => use prob=outputs[i], else prob=1 - outputs[i].
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else 1 - outputs[i]
        loss -= torch.log(prob + 1e-8)
    return loss

def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    outputs < 0.5 => predict '1', else '0' (or vice versa).
    Adjust accordingly to match your labeling.
    """
    predictions = (outputs < 0.5).float()  
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy.item()


@pytest.mark.integtest
def test_integration_mnist_tpcp_mps():
    """
    Integration test showing that uMPS and MPSTPCP (with EXACT manifold)
    produce the same forward outputs and loss on MNIST (digits 0 and 1),
    remain consistent through backward passes, and that the TPCP Kraus 
    ops remain on the Stiefel manifold.
    """
    # --------------------------
    # 2) Fix random seeds
    # --------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------
    # 3) Load MNIST (digits 0,1)
    # --------------------------
    img_size = 16
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(filiter_single_channel),
        transforms.Lambda(embedding_pixel),
        transforms.Lambda(lambda x: x.to(torch.float64))  # Change dtype to float64
    ])
    mnist_train = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    # Filter for digits 0 and 1
    mnist_train = filter_dataset(mnist_train, allowed_digits=[0, 1])

    trainloader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=64,
        shuffle=False
    )

    N = img_size * img_size  # 16*16 = 256
    d = 2                    # local dimension

    # --------------------------
    # 4) Create both models
    #    -  uMPS
    #    -  MPSTPCP (manifold=EXACT)
    # --------------------------
    umps_model = umps.uMPS(
        N=N,
        chi=2,  # bond dimension
        d=d,
        l=2,
        layers=1,
        device=torch.device("cpu"),
        init_with_identity=True
    )

    tpcp_model = MPSTPCP(
        N=N,
        K=1,              # single Kraus op
        d=d,
        with_identity=True,
        manifold=ManifoldType.EXACT  # EXACT Stiefel manifold
    )

    tpcp_model.kraus_ops.init_params()
    params = tpcp_model.kraus_ops.parameters()

    umps_model.initialize_MPS(torch.stack([p.reshape(d, d, d, d) for p in params]))
    # Check if both models return the same output for the initial input
    with torch.no_grad():
        data, target = next(iter(trainloader))
        out_umps_initial = umps_model(data.permute(1, 0, 2))
        out_tpcp_initial = tpcp_model(data)
    
    assert torch.allclose(out_umps_initial, out_tpcp_initial, atol=1e-6), \
        "Initial outputs differ between uMPS and MPSTPCP."

    

    # Check manifold is EXACT
    assert isinstance(tpcp_model.manifold, geoopt.manifolds.EuclideanStiefelExact), f"TPCP model is not using EXACT manifold {tpcp_model.manifold}"

    # --------------------------
    # 5) Define optimizers
    # --------------------------
    lr = 0.001
    optimizer_umps = unitary_optimizer.SGD(umps_model, lr=lr)
    optimizer_tpcp = geoopt.optim.RiemannianSGD(tpcp_model.kraus_ops.parameters(), lr=lr)

    # --------------------------
    # 6) Train for a small # of epochs
    # --------------------------
    num_epochs = 30  # Keep it small for test
    for epoch in range(num_epochs):
        total_loss_umps = 0.0
        total_loss_tpcp = 0.0
        total_batches = 0

        for i, (data, target) in enumerate(trainloader):
            if i >= 1:
                break
            # data shape: [batch_size, N, 2]
            # => reorder for uMPS: [N, batch_size, d]
            inputs_umps = data.permute(1, 0, 2)   # shape [N, bs, 2]
            inputs_tpcp = data                   # shape [bs, N, 2]

            # --------------------------
            # Forward pass (no grads)
            # --------------------------
            with torch.no_grad():
                out_umps_test = umps_model(inputs_umps)
                out_tpcp_test = tpcp_model(inputs_tpcp)
            
            # Compare outputs (they should match fairly closely)
            assert torch.allclose(out_umps_test, out_tpcp_test, atol=1e-6), \
                "Forward outputs differ between uMPS and MPSTPCP before training steps."

            # --------------------------
            # 7) Zero gradients & forward pass (with grads)
            # --------------------------
            optimizer_umps.zero_grad()
            optimizer_tpcp.zero_grad()

            out_umps = umps_model(inputs_umps)
            out_tpcp = tpcp_model(inputs_tpcp)

            # --------------------------
            # (a) Compare forward passes 
            # --------------------------
            assert torch.allclose(out_umps, out_tpcp, atol=1e-6), \
                "Forward outputs differ between uMPS and MPSTPCP."

            # --------------------------
            # (b) Compute & compare loss
            # --------------------------
            loss_umps = loss_batch(out_umps, target)
            loss_tpcp = loss_batch(out_tpcp, target)

            # They should match
            assert torch.allclose(loss_umps, loss_tpcp, atol=1e-6), \
                "Loss values differ between uMPS and MPSTPCP."
            
            print(f"loss iter {i} and epoch {epoch}: {loss_umps.item()}")

            # --------------------------
            # (c) Backward pass
            # --------------------------
            loss_umps.backward()
            loss_tpcp.backward()

            # --------------------------
            # (d) Optimization step
            # --------------------------
            optimizer_umps.step()
            optimizer_tpcp.step()

            # --------------------------
            # (e) Check parameter equality
            # --------------------------
            umps_params = list(umps_model.parameters())
            tpcp_params = list(tpcp_model.kraus_ops.parameters())
            for p_umps, p_tpcp in zip(umps_params, tpcp_params):
                assert torch.allclose(p_umps.reshape(d**2, d**2), p_tpcp, atol=1e-6), \
                    "Model parameters differ between uMPS and MPSTPCP during training."

            total_loss_umps += loss_umps.item()
            total_loss_tpcp += loss_tpcp.item()
            total_batches += 1

        avg_loss_umps = total_loss_umps / total_batches
        avg_loss_tpcp = total_loss_tpcp / total_batches

        # The two models should have very similar loss
        assert abs(avg_loss_umps - avg_loss_tpcp) < 1e-3, \
            "Loss mismatch between uMPS and TPCP after an epoch of training."
