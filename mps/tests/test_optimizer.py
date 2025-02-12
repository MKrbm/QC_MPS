import pytest
import torch
import random
import numpy as np
import geoopt

from mps.umps import uMPS
from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps import unitary_optimizer

def loss_batch(outputs, labels):
    """
    Compute a binary cross-entropy style loss.
    For each sample, if label==0 then use outputs[i] else (1 - outputs[i]).
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss -= torch.log(prob + 1e-8)
    return loss

@pytest.mark.integtest
def test_cptp_mps_sgd():
    """
    Verify that one can use SGD optimizers to update uMPS and MPSTPCP models identically.
    
    This test:
      - Fixes the random seeds.
      - Creates a small synthetic dataset.
      - Instantiates a uMPS model and a corresponding MPSTPCP model (with Stiefel EXACT manifold).
      - Sets up SGD optimizers (using unitary_optimizer.SGD for uMPS and geoopt.optim.RiemannianSGD for MPSTPCP).
      - Runs several optimization steps and, after each step, asserts that the first parameter
        of uMPS (after reshaping and transposition) is close to the first Kraus operator of MPSTPCP.
    """
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create synthetic data
    bs = 3       # batch size
    N = 64        # number of qubits
    data = torch.randn(bs, N, 2, dtype=torch.float64)
    data = data / torch.norm(data, dim=-1, keepdim=True)
    target = torch.randint(0, 2, (bs,), dtype=torch.float64)

    # Create models with identical initialization
    chi = 2
    layers = 1
    umps_model = uMPS(
        N=N,
        chi=chi,
        d=2,
        l=2,
        layers=layers,
        device=torch.device("cpu"),
        init_with_identity=False
    )
    mpstpcp_model = MPSTPCP(
        N=N,
        K=1,
        d=2,
        with_identity=False,
        manifold=ManifoldType.EXACT
    )

    mpstpcp_model.kraus_ops.init_params()
    params = mpstpcp_model.kraus_ops.parameters()

    umps_model.initialize_MPS(torch.stack([p.reshape(2, 2, 2, 2) for p in params]))
    # Check if both models return the same output for the initial input

    # Prepare input shapes:
    # uMPS expects input of shape (N, bs, 2) while MPSTPCP expects (bs, N, 2)
    input_for_umps = data.permute(1, 0, 2)
    input_for_tpcp = data

    # Set models to training mode
    umps_model.train()
    mpstpcp_model.train()


    # Forward pass
    outputs_umps = umps_model(input_for_umps)
    outputs_tpcp = mpstpcp_model(input_for_tpcp)

    # Ensure the outputs are the same
    assert torch.allclose(outputs_umps, outputs_tpcp, atol=1e-6), \
        "Outputs differ between uMPS and MPSTPCP"

    # Set up SGD optimizers
    lr = 0.1
    optimizer_umps = unitary_optimizer.SGD(umps_model, lr=lr)
    optimizer_mpstpcp = geoopt.optim.RiemannianSGD(mpstpcp_model.parameters(), lr=lr)

    # Run several optimization steps and check that parameter updates match
    num_steps = 100
    for step in range(num_steps):
        optimizer_umps.zero_grad()
        optimizer_mpstpcp.zero_grad()

        # Forward pass
        outputs_umps = umps_model(input_for_umps)
        outputs_tpcp = mpstpcp_model(input_for_tpcp)

        # Compute loss
        loss_umps = loss_batch(outputs_umps, target)
        loss_tpcp = loss_batch(outputs_tpcp, target)

        
        # Ensure the outputs are the same
        assert torch.allclose(outputs_umps.detach(), outputs_tpcp.detach(), atol=1e-6), \
            "Outputs differ between uMPS and MPSTPCP"


        # Backward pass
        loss_umps.backward()
        loss_tpcp.backward()

        # Step in the optimizers
        optimizer_umps.step()
        optimizer_mpstpcp.step()

        # Compare updated parameters.
        # For uMPS, the first parameter is stored in umps_model.params[0] and we reshape it to (4, 4).
        # For MPSTPCP, the first Kraus operator is in mpstpcp_model.kraus_ops[0].
        new_mps = umps_model.params[0].reshape(4, 4)
        new_tpcp = mpstpcp_model.kraus_ops[0]

        # Ensure the shapes are the same
        assert new_mps.shape == new_tpcp.shape, "Updated parameter shapes mismatch in SGD"

        # Note: The uMPS parameter has an opposite convention so we compare its transpose.
        assert torch.allclose(new_mps, new_tpcp, atol=1e-6), \
            f"Parameters differ between uMPS and MPSTPCP after SGD step {step+1}"

@pytest.mark.integtest
def test_cptp_mps_adam():
    """
    Verify that one can use Adam optimizers to update uMPS and MPSTPCP models identically.
    
    This test:
      - Fixes the random seeds.
      - Creates a small synthetic dataset.
      - Instantiates a uMPS model and a corresponding MPSTPCP model (with Stiefel EXACT manifold).
      - Sets up Adam optimizers (using unitary_optimizer.Adam for uMPS and geoopt.optim.RiemannianAdam for MPSTPCP).
      - Runs several optimization steps and, after each step, asserts that the first parameter
        of uMPS (after reshaping and transposition) is close to the first Kraus operator of MPSTPCP.
    """
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create synthetic data
    bs = 3       # batch size
    N = 64        # number of qubits
    data = torch.randn(bs, N, 2, dtype=torch.float64)
    data = data / torch.norm(data, dim=-1, keepdim=True)
    target = torch.randint(0, 2, (bs,), dtype=torch.float64)

    # Create models with identical initialization
    chi = 2
    layers = 1
    umps_model = uMPS(
        N=N,
        chi=chi,
        d=2,
        l=2,
        layers=layers,
        device=torch.device("cpu"),
        init_with_identity=False
    )
    mpstpcp_model = MPSTPCP(
        N=N,
        K=1,
        d=2,
        with_identity=False,
        manifold=ManifoldType.EXACT
    )

    mpstpcp_model.kraus_ops.init_params()
    params = mpstpcp_model.kraus_ops.parameters()

    umps_model.initialize_MPS(torch.stack([p.reshape(2, 2, 2, 2) for p in params]))

    # Prepare input shapes
    input_for_umps = data.permute(1, 0, 2)
    input_for_tpcp = data

    # Set models to training mode
    umps_model.train()
    mpstpcp_model.train()

    # Set up Adam optimizers
    lr = 0.01
    optimizer_umps = unitary_optimizer.Adam(umps_model, lr=lr)
    optimizer_mpstpcp = geoopt.optim.RiemannianAdam(mpstpcp_model.parameters(), lr=lr, weight_decay=0.0)

    # Run several optimization steps and check that parameter updates match
    num_steps = 5
    for step in range(num_steps):
        optimizer_umps.zero_grad()
        optimizer_mpstpcp.zero_grad()

        # Forward pass
        outputs_umps = umps_model(input_for_umps)
        outputs_tpcp = mpstpcp_model(input_for_tpcp)

        # Compute loss
        loss_umps = loss_batch(outputs_umps, target)
        loss_tpcp = loss_batch(outputs_tpcp, target)

        # Ensure the outputs are the same
        assert torch.allclose(outputs_umps.detach(), outputs_tpcp.detach(), atol=1e-6), \
            "Outputs differ between uMPS and MPSTPCP"

        # Backward pass
        loss_umps.backward()
        loss_tpcp.backward()

        # Step in the optimizers
        optimizer_umps.step()
        optimizer_mpstpcp.step()

        # Compare updated parameters.
        new_mps = umps_model.params[0].reshape(4, 4)
        new_tpcp = mpstpcp_model.kraus_ops[0]

        # Check shapes
        assert new_mps.shape == new_tpcp.shape, "Updated parameter shapes mismatch in Adam"

        # Compare with the necessary transpose for uMPS
        assert torch.allclose(new_mps, new_tpcp, atol=1e-6), \
            f"Parameters differ between uMPS and MPSTPCP after Adam step {step+1}"
