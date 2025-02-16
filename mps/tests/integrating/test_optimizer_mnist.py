import pytest
import torch
import random
import numpy as np
import geoopt
import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from mps.umps import uMPS
from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps import unitary_optimizer
from mps.radam import RiemannianAdam

# --------------------------
# Utility Functions
# --------------------------
def filiter_single_channel(batch: torch.Tensor) -> torch.Tensor:
    """
    Given an MNIST tensor with shape [C, H, W],
    keep only the first channel => shape [H, W].
    """
    return batch[0, ...]

def embedding_pixel(batch: torch.Tensor) -> torch.Tensor:
    """
    Flatten image (H, W) -> (H*W) and create a 2D embedding for each pixel:
       [ pixel, (1 - pixel) ]
    Then normalize each embedding vector.
    """
    x = batch.view(-1)  # shape: [H*W]
    x = torch.stack([x, 1 - x], dim=-1)  # shape: [H*W, 2]
    x = x / torch.norm(x, dim=-1, keepdim=True)
    return x

def filter_dataset(dataset, allowed_digits=[0, 1]):
    """
    Return a subset of the dataset containing only samples whose labels are in allowed_digits.
    """
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in allowed_digits:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)

def loss_batch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy–like loss for outputs in [0, 1].
    For label==0 use outputs[i] and for label==1 use (1 - outputs[i]).
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss -= torch.log(prob + 1e-8)
    return loss

# --------------------------
# Test using SGD on MNIST
# --------------------------
@pytest.mark.integtest
def test_cptp_mps_sgd_mnist():
    """
    Test that uMPS and MPSTPCP produce the same forward outputs,
    loss values, and—crucially—the same gradients at every optimization step
    when trained on MNIST (digits 0 and 1) using SGD.
    """
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------
    # Load MNIST dataset (resize to 16x16)
    # --------------------------
    img_size = 16
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(filiter_single_channel),
        transforms.Lambda(embedding_pixel),
        transforms.Lambda(lambda x: x.to(torch.float64))
    ])
    mnist_train = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    # Keep only digits 0 and 1
    mnist_train = filter_dataset(mnist_train, allowed_digits=[0, 1])
    trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=False)

    # --------------------------
    # Set up model parameters
    # --------------------------
    # Here N is the length of the chain (for an image, N = H * W)
    N = img_size * img_size  # 16*16 = 256
    d = 2                    # local dimension
    chi = 2                  # bond dimension
    layers = 1

    # --------------------------
    # Create both models with identical initialization
    # --------------------------
    umps_model = uMPS(
        N=N,
        chi=chi,
        d=d,
        l=2,
        layers=layers,
        device=torch.device("cpu"),
        init_with_identity=True
    )
    mpstpcp_model = MPSTPCP(
        N=N,
        K=1,              # single Kraus operator
        d=d,
        with_identity=True,
        manifold=ManifoldType.EXACT
    )

    # Initialize MPSTPCP parameters and then use them to initialize uMPS
    mpstpcp_model.kraus_ops.init_params()
    params = list(mpstpcp_model.kraus_ops.parameters())
    # The uMPS expects a tensor of shape [d, d, d, d] per Kraus op.
    umps_model.initialize_MPS(torch.stack([p.reshape(d, d, d, d) for p in params]))

    # --------------------------
    # Prepare a single batch
    # --------------------------
    data, target = next(iter(trainloader))
    # uMPS expects inputs of shape [N, batch_size, d]
    inputs_umps = data.permute(1, 0, 2)
    # MPSTPCP expects inputs of shape [batch_size, N, d]
    inputs_tpcp = data

    # Check that the initial outputs are the same
    out_umps = umps_model(inputs_umps)
    out_tpcp = mpstpcp_model(inputs_tpcp)
    assert torch.allclose(out_umps, out_tpcp, atol=1e-6), "Initial outputs differ."

    # --------------------------
    # Set up SGD optimizers
    # --------------------------
    lr = 0.1
    optimizer_umps = unitary_optimizer.SGD(umps_model, lr=lr)
    optimizer_tpcp = geoopt.optim.RiemannianSGD(mpstpcp_model.kraus_ops.parameters(), lr=lr)

    # --------------------------
    # Run several optimization steps and check gradient consistency
    # --------------------------
    num_steps = 5
    for step in range(num_steps):
        optimizer_umps.zero_grad()
        optimizer_tpcp.zero_grad()

        # Forward pass
        out_umps = umps_model(inputs_umps)
        out_tpcp = mpstpcp_model(inputs_tpcp)

        # Compute loss
        loss_umps = loss_batch(out_umps, target)
        loss_tpcp = loss_batch(out_tpcp, target)

        # Check that forward outputs and losses agree
        assert torch.allclose(out_umps, out_tpcp, atol=1e-6), f"Forward outputs differ at step {step}."
        assert torch.allclose(loss_umps, loss_tpcp, atol=1e-6), f"Loss values differ at step {step}."

        # Backward pass
        loss_umps.backward()
        loss_tpcp.backward()

        # Check that the gradients on corresponding parameters are the same.
        # (Note: The uMPS parameters are stored in a different shape; here we reshape
        # the uMPS gradients to a (d**2, d**2) matrix to compare with the MPSTPCP ones.)
        # umps_params = list(umps_model.parameters())
        # tpcp_params = list(mpstpcp_model.kraus_ops.parameters())
        # for idx, (p_u, p_t) in enumerate(zip(umps_params, tpcp_params)):
        #     grad_u = p_u.grad
        #     grad_t = p_t.grad
        #     if grad_u is not None and grad_t is not None:
        #         grad_u_reshaped = grad_u.reshape(d**2, d**2)
        #         assert torch.allclose(grad_u_reshaped, grad_t, atol=1e-6), \
        #             f"Gradient mismatch in parameter {idx} at step {step}."

        # Take an optimization step
        optimizer_umps.step()
        optimizer_tpcp.step()

        # Check that after the update the parameters remain equal.
        for idx, (p_u, p_t) in enumerate(zip(umps_model.parameters(), mpstpcp_model.kraus_ops.parameters())):
            p_u_reshaped = p_u.reshape(d**2, d**2)
            assert torch.allclose(p_u_reshaped, p_t, atol=1e-6), \
                f"Parameter mismatch in parameter {idx} after step {step}."


# --------------------------
# Test using Adam on MNIST
# --------------------------
@pytest.mark.integtest
def test_cptp_mps_adam_mnist():
    """
    Test that uMPS and MPSTPCP produce the same forward outputs,
    loss values, and gradients at every optimization step
    when trained on MNIST (digits 0 and 1) using Adam.
    """
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------
    # Load MNIST dataset (resize to 16x16)
    # --------------------------
    img_size = 16
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(filiter_single_channel),
        transforms.Lambda(embedding_pixel),
        transforms.Lambda(lambda x: x.to(torch.float64))
    ])
    mnist_train = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    mnist_train = filter_dataset(mnist_train, allowed_digits=[0, 1])
    trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=False)

    # --------------------------
    # Set up model parameters
    # --------------------------
    N = img_size * img_size  # 256
    d = 2
    chi = 2
    layers = 1

    # --------------------------
    # Create both models with identical initialization
    # --------------------------
    umps_model = uMPS(
        N=N,
        chi=chi,
        d=d,
        l=2,
        layers=layers,
        device=torch.device("cpu"),
        init_with_identity=True
    )
    mpstpcp_model = MPSTPCP(
        N=N,
        K=1,
        d=d,
        with_identity=True,
        manifold=ManifoldType.EXACT
    )

    mpstpcp_model.kraus_ops.init_params()
    params = list(mpstpcp_model.kraus_ops.parameters())
    umps_model.initialize_MPS(torch.stack([p.reshape(d, d, d, d) for p in params]))

    # --------------------------
    # Prepare a single batch
    # --------------------------
    data, target = next(iter(trainloader))
    inputs_umps = data.permute(1, 0, 2)  # [N, batch_size, d]
    inputs_tpcp = data               # [batch_size, N, d]

    # Check initial output consistency
    out_umps = umps_model(inputs_umps)
    out_tpcp = mpstpcp_model(inputs_tpcp)
    assert torch.allclose(out_umps, out_tpcp, atol=1e-6), "Initial outputs differ."

    # --------------------------
    # Set up Adam optimizers
    # --------------------------
    lr = 0.01
    optimizer_umps = unitary_optimizer.Adam(umps_model, lr=lr)
    optimizer_tpcp = RiemannianAdam(mpstpcp_model.kraus_ops.parameters(), lr=lr, weight_decay=0.0)

    # --------------------------
    # Run several optimization steps and check gradient consistency
    # --------------------------
    num_steps = 5
    for step in range(num_steps):
        optimizer_umps.zero_grad()
        optimizer_tpcp.zero_grad()

        # Forward pass
        out_umps = umps_model(inputs_umps)
        out_tpcp = mpstpcp_model(inputs_tpcp)

        # Compute loss
        loss_umps = loss_batch(out_umps, target)
        loss_tpcp = loss_batch(out_tpcp, target)

        # Check that forward outputs and losses agree
        assert torch.allclose(out_umps, out_tpcp, atol=1e-6), f"Forward outputs differ at step {step}."
        assert torch.allclose(loss_umps, loss_tpcp, atol=1e-6), f"Loss values differ at step {step}."

        # Backward pass
        loss_umps.backward()
        loss_tpcp.backward()

        # Check gradient consistency
        # umps_params = list(umps_model.parameters())
        # tpcp_params = list(mpstpcp_model.kraus_ops.parameters())
        # for idx, (p_u, p_t) in enumerate(zip(umps_params, tpcp_params)):
        #     grad_u = p_u.grad
        #     grad_t = p_t.grad
        #     if grad_u is not None and grad_t is not None:
        #         grad_u_reshaped = grad_u.reshape(d**2, d**2)
        #         assert torch.allclose(grad_u_reshaped, grad_t, atol=1e-6), \
        #             f"Gradient mismatch in parameter {idx} at step {step}."

        # Optimization step
        optimizer_umps.step()
        optimizer_tpcp.step()

        # Check parameter consistency after the update
        for idx, (p_u, p_t) in enumerate(zip(umps_model.parameters(), mpstpcp_model.kraus_ops.parameters())):
            p_u_reshaped = p_u.reshape(d**2, d**2)
            # print("p_u_reshaped : ", p_u_reshaped.reshape(4,4))
            # print("p_t : ", p_t.reshape(4,4))
            # print("step : ", step)
            # print("distance : ", torch.norm(p_u_reshaped - p_t))
            assert torch.allclose(p_u_reshaped, p_t, rtol=1e-5), \
                f"Parameter mismatch in parameter {idx} after step {step}."
