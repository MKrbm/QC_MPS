import pytest
import torch
import random
import numpy as np

import geoopt

from mps.umps import uMPS
from mps.tpcp_mps import MPSTPCP
from mps import unitary_optimizer
from mps.tpcp_mps import ManifoldType

def loss_batch(outputs, labels):
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss -= torch.log(prob + 1e-8)
    return loss

def calculate_accuracy(outputs, labels):
    predictions = (outputs < 0.5).float()  # output < 0.5 means prediction of 1 (or vice versa)
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy.item()

@pytest.mark.integtest
def test_umps_tpcp_mps_integration():
    """
    Integration test verifying:
      (1) Same forward outputs
      (2) Same Riemannian gradient
      (3) Same parameter after one step of optimization
      (4) Loss decreases after optimization
      (5) Same forward outputs after one step of optimization
      (6) Same forward outputs after 10 steps of optimization
    for uMPS vs. MPSTPCP (Stiefel EXACT).
    """
    # --------------------------
    # 1) Fix random seeds
    # --------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------
    # 2) Create synthetic data
    # --------------------------
    bs = 3       # batch size
    N = 2        # number of qubits
    data = torch.randn(bs, N, 2, dtype=torch.float64)
    data = data / torch.norm(data, dim=-1, keepdim=True)
    target = torch.randint(0, 2, (bs,), dtype=torch.float64)

    # --------------------------
    # 3) Create and compare models
    # --------------------------
    chi = 2
    layers = 1
    umps_model = uMPS(N=N, chi=chi, d=2, l=2, layers=layers, device=torch.device("cpu"), init_with_identity=True)
    mpstpcp_model = MPSTPCP(N=N, K=1, d=2, with_identity=True, manifold=ManifoldType.EXACT)

    umps_model.

    # Prepare input shapes
    input_for_umps = data.permute(1, 0, 2)  # shape (N, bs, 2)
    input_for_tpcp = data                  # shape (bs, N, 2)

    # Forward pass
    outputs_umps = umps_model(input_for_umps)
    outputs_tpcp = mpstpcp_model(input_for_tpcp)

    # --------------------------
    # (1) Check outputs match
    # --------------------------
    assert torch.allclose(outputs_umps, outputs_tpcp, atol=1e-6), \
        "Forward pass outputs differ between uMPS and MPSTPCP"

    # --------------------------
    # 4) Set up optimizers
    # --------------------------
    lr = 0.1
    optimizer_umps = unitary_optimizer.SGD(umps_model, lr=lr)
    optimizer_mpstpcp = geoopt.optim.RiemannianSGD(mpstpcp_model.parameters(), lr=lr)

    # Set both models to training mode
    umps_model.train()
    mpstpcp_model.train()

    # Zero out gradients
    optimizer_umps.zero_grad()
    optimizer_mpstpcp.zero_grad()

    # --------------------------
    # 5) Compute loss & backward
    # --------------------------

    loss_umps = loss_batch(outputs_umps, target)
    loss_tpcp = loss_batch(outputs_tpcp, target)

    loss_umps.backward()
    loss_tpcp.backward()

    # --------------------------
    # (2) Check Riemannian gradient
    #     Compare a single parameter from each model
    # --------------------------
    # For uMPS, we manually reshape the first parameter to (4, 4)
    # (assuming N=2, chi=2 => dimension is 4)
    u_mps = umps_model.params[0].reshape(4, 4)
    grad_mps = umps_model.params[0].grad.reshape(4, 4)
    rg_mps = unitary_optimizer.riemannian_gradient(u_mps, grad_mps)
    # To associate the correct direction, we negate the Riemannian gradient
    rg_mps = - rg_mps

    # For MPSTPCP, we pick the first Kraus op
    u_tpcp = mpstpcp_model.kraus_ops[0]       # shape (4, 4)
    grad_tpcp = u_tpcp.grad                   # same shape
    rg_tpcp = u_tpcp.manifold.egrad2rgrad(u_tpcp, grad_tpcp)

    # print(rg_mps.T)
    # print(rg_tpcp)
    assert rg_mps.shape == rg_tpcp.shape, "Riemannian gradients shape mismatch"
    assert torch.allclose(rg_mps, rg_tpcp, atol=1e-6), \
        "Riemannian gradients differ between uMPS and MPSTPCP"

    # --------------------------
    # Save old parameters (for debug or check)
    # so we can confirm they change in the same way
    # --------------------------
    old_mps_param = umps_model.params[0].detach().clone()
    old_tpcp_param = mpstpcp_model.kraus_ops[0].detach().clone()

    # --------------------------
    # 6) Compute loss before optimization
    # --------------------------
    loss_before_umps = loss_umps.item()
    loss_before_tpcp = loss_tpcp.item()

    # --------------------------
    # 7) Step in both optimizers
    # --------------------------
    optimizer_umps.step()
    optimizer_mpstpcp.step()

    # --------------------------
    # (3) After one step, check final parameters
    # --------------------------
    new_mps = umps_model.params[0].reshape(4, 4)
    new_tpcp = mpstpcp_model.kraus_ops[0]

    print("New uMPS parameters:\n", new_mps)
    print("New MPSTPCP parameters:\n", new_tpcp)
    # They should be on the same manifold, so check close
    assert new_mps.shape == new_tpcp.shape, "Updated parameters shape mismatch"
    # The transpose of the uMPS parameter accounts for the opposite direction of the unitary-MPS.
    assert torch.allclose(new_mps.T, new_tpcp, atol=1e-6), \
        "After one optimization step, parameters differ between uMPS and MPSTPCP"

    # --------------------------
    # 8) Forward pass after optimization
    # --------------------------
    outputs_umps_new = umps_model(input_for_umps)
    outputs_tpcp_new = mpstpcp_model(input_for_tpcp)

    # --------------------------
    # (5) Check outputs match after one step of optimization
    # --------------------------
    assert torch.allclose(outputs_umps_new, outputs_tpcp_new, atol=1e-6), \
        "Forward pass outputs differ between uMPS and MPSTPCP after one step of optimization"

    # --------------------------
    # 9) Compute new loss after optimization
    # --------------------------
    loss_umps_new = loss_batch(outputs_umps_new, target)
    loss_tpcp_new = loss_batch(outputs_tpcp_new, target)

    # --------------------------
    # (4) Check that loss has decreased
    # --------------------------
    assert loss_umps_new.item() < loss_before_umps, \
        "Loss did not decrease for uMPS after optimization"
    assert loss_tpcp_new.item() < loss_before_tpcp, \
        "Loss did not decrease for MPSTPCP after optimization"

    print(f"Loss before optimization (uMPS): {loss_before_umps}")
    print(f"Loss after optimization (uMPS): {loss_umps_new.item()}")
    print(f"Loss before optimization (MPSTPCP): {loss_before_tpcp}")
    print(f"Loss after optimization (MPSTPCP): {loss_tpcp_new.item()}")

    # --------------------------
    # 10) Perform 10 steps of optimization
    # --------------------------
    previous_loss_umps = float('inf')
    previous_loss_tpcp = float('inf')

    for _ in range(10):
        optimizer_umps.zero_grad()
        optimizer_mpstpcp.zero_grad()

        outputs_umps = umps_model(input_for_umps)
        outputs_tpcp = mpstpcp_model(input_for_tpcp)

        loss_umps = loss_batch(outputs_umps, target)
        loss_tpcp = loss_batch(outputs_tpcp, target)

        assert loss_umps.item() < previous_loss_umps, \
            "Loss did not decrease for uMPS in this epoch"
        assert loss_tpcp.item() < previous_loss_tpcp, \
            "Loss did not decrease for MPSTPCP in this epoch"

        previous_loss_umps = loss_umps.item()
        previous_loss_tpcp = loss_tpcp.item()

        loss_umps.backward()
        loss_tpcp.backward()

        optimizer_umps.step()
        optimizer_mpstpcp.step()

    # --------------------------
    # (6) Check outputs match after 10 steps of optimization
    # --------------------------
    outputs_umps_final = umps_model(input_for_umps)
    outputs_tpcp_final = mpstpcp_model(input_for_tpcp)

    assert torch.allclose(outputs_umps_final, outputs_tpcp_final, atol=1e-6), \
        "Forward pass outputs differ between uMPS and MPSTPCP after 10 steps of optimization"


@pytest.mark.integtest
@pytest.mark.parametrize("manifold_type", [ManifoldType.EXACT, ManifoldType.FROBENIUS, ManifoldType.CANONICAL])
def test_tpcp_largeN_integration(manifold_type):
    """
    Integration test for MPSTPCP on N = 16*16 and batch size=10.
    Checks:
      (1) Kraus ops remain on the specified manifold after one optimization step.
      (2) Loss decreases after one step.
      (3) The new Kraus ops are well-approximated by X - lr * rgrad (Euclidean step).
      (4) Loss decreases at each step over multiple iterations.
    """
    # --------------------------
    # 1) Fix random seeds
    # --------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------
    # 2) Create synthetic data
    # --------------------------
    bs = 10
    N = 16 * 16  # 256
    d = 2        # local dimension (e.g. qubit basis)
    K = 3

    # Create random data of shape (bs, N, d)
    # Normalizing data row-wise so sum(data[i, j, :]^2) ~ 1 for each (i,j)
    data = torch.randn(bs, N, d, dtype=torch.float64)
    data = data / torch.norm(data, dim=-1, keepdim=True)

    # Binary labels
    labels = torch.randint(0, 2, (bs,), dtype=torch.float64)

    # --------------------------
    # 3) Build a TPCP MPS model
    # --------------------------
    # We'll create MPSTPCP with K=1 Kraus operator per site for simplicity
    mpstpcp_model = MPSTPCP(
        N=N,
        K=K,
        d=d,
        with_identity=True,
        manifold=manifold_type,  # Use parameterized manifold type
    )
    mpstpcp_model.train()

    # --------------------------
    # 4) Set up Riemannian optimizer
    # --------------------------
    lr = 0.05
    optimizer = geoopt.optim.RiemannianSGD(mpstpcp_model.parameters(), lr=lr)

    # --------------------------
    # 5) Forward pass, compute loss
    # --------------------------
    outputs = mpstpcp_model(data)  # shape (bs,)
    loss_before = loss_batch(outputs, labels)
    loss_before_value = loss_before.item()

    # Backward pass
    optimizer.zero_grad()
    loss_before.backward()

    # -----------
    # Manifold checks: 
    # We examine the first Kraus op to show how one might check the 
    # "X - lr * rgrad" approximation. Summation for all Kraus ops is similarly possible.
    # -----------
    kraus_op_old = mpstpcp_model.kraus_ops[0].detach().clone()
    grad_old_euclid = mpstpcp_model.kraus_ops[0].grad.detach().clone()

    # Convert Euclidean grad -> Riemannian grad
    manifold = mpstpcp_model.kraus_ops[0].manifold
    rgrad_old = manifold.egrad2rgrad(kraus_op_old, grad_old_euclid)

    # --------------------------
    # 6) One optimization step
    # --------------------------
    optimizer.step()

    # Check manifold: after .step(), kraus_ops should remain on the specified manifold
    kraus_op_new = mpstpcp_model.kraus_ops[0].detach()
    check_eye = kraus_op_new.transpose(-1, -2) @ kraus_op_new
    identity_approx = torch.eye(check_eye.shape[-1], dtype=torch.float64)
    assert torch.allclose(check_eye, identity_approx, atol=1e-7), \
        "Kraus op not on Stiefel manifold after one step"

    # Approximate check: new param ~ old param - lr * rgrad
    approx_new = kraus_op_old - lr * rgrad_old
    assert torch.allclose(kraus_op_new, approx_new, atol=5e-2), \
        "Kraus operator is not close to a simple Euclidean step update. " \
        f"Max diff = {(kraus_op_new - approx_new).abs().max().item()}"

    # --------------------------
    # 7) Check that loss decreased
    # --------------------------
    with torch.no_grad():
        outputs_after = mpstpcp_model(data)
        loss_after = loss_batch(outputs_after, labels)
    assert loss_after.item() < loss_before_value, \
        "Loss did not decrease after first optimization step"

    print(f"Loss before first step: {loss_before_value}")
    print(f"Loss after first step:  {loss_after.item()}")

    # --------------------------
    # 8) Multiple steps: ensure the loss decreases at each step
    # --------------------------
    n_steps = 5
    previous_loss = loss_after.item()

    for step in range(1, n_steps+1):
        optimizer.zero_grad()
        outputs = mpstpcp_model(data)
        loss = loss_batch(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_val = loss.item()
            print(f"Step {step} => loss: {loss_val:.6f}")
            assert loss_val < previous_loss + 1e-12, \
                f"Loss did not decrease at step {step}"
            previous_loss = loss_val
