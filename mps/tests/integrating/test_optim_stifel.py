import pytest
import torch
import random
import numpy as np

# Import the optimizer that we want to use.
from mps.StiefelOptimizers import StiefelSGD

# Import the MPSTPCP model and the manifold type.
from mps.tpcp_mps import MPSTPCP, ManifoldType

###############################################################################
# Helper functions
###############################################################################

def loss_batch(outputs, labels):
    """
    Computes a simple negative-log-likelihood loss for binary labels.
    For each output, if the corresponding label==0, we take the output,
    otherwise (if label==1) we use (1 - output).
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        # Avoid log(0) by adding a small epsilon.
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss -= torch.log(prob + 1e-8)
    return loss

###############################################################################
# Test 1: Small-scale integration test using StiefelSGD.
###############################################################################

@pytest.mark.integtest
def test_mpstpcp_stiefel_sgd_integration():
    """
    Integration test for MPSTPCP using the StiefelSGD optimizer.
    This test:
      - Fixes random seeds.
      - Creates a synthetic (small) dataset.
      - Instantiates an MPSTPCP model with K=1 Kraus operator and manifold=EXACT.
      - Runs a forward pass and computes a binary cross entropy–like loss.
      - Performs one optimizer step with StiefelSGD.
      - Checks that:
          (a) The first Kraus operator remains on the Stiefel manifold (i.e. UᵀU≈I),
          (b) The updated Kraus operator is well approximated by a Euclidean step update
              (i.e. X_new ≈ X_old – lr * rgrad),
          (c) The loss decreases after the optimization step.
    """
    # --------------------------
    # 1) Fix random seeds for reproducibility
    # --------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------
    # 2) Create synthetic data.
    # Here, bs = batch size, N = number of sites/qubits, and d = local dimension.
    # --------------------------
    bs = 3
    N = 2
    d = 2
    data = torch.randn(bs, N, d, dtype=torch.float64)
    # Normalize the data along the last dimension.
    data = data / torch.norm(data, dim=-1, keepdim=True)
    # Binary labels.
    target = torch.randint(0, 2, (bs,), dtype=torch.float64)

    # --------------------------
    # 3) Instantiate the MPSTPCP model.
    # For simplicity we use K=1 Kraus operator per site and the EXACT manifold.
    # --------------------------
    mpstpcp_model = MPSTPCP(
        N=N,
        K=1,
        d=d,
        with_identity=True,
        manifold=ManifoldType.EXACT
    )
    mpstpcp_model.train()

    # --------------------------
    # 4) Set up the StiefelSGD optimizer.
    # (Here we use no momentum and a moderate learning rate.)
    # --------------------------
    lr = 0.01
    momentum = 0.0
    optimizer = StiefelSGD(mpstpcp_model.kraus_ops.parameters(), lr=lr, momentum=momentum)

    # --------------------------
    # 5) Forward pass and compute loss.
    # --------------------------
    outputs = mpstpcp_model(data)  # expected shape: (bs,)
    loss_before = loss_batch(outputs, target)
    loss_before_val = loss_before.item()

    optimizer.zero_grad()
    loss_before.backward()

    # --------------------------
    # 6) (Optional) Check the “approximate Euclidean step” for the first Kraus operator.
    # Record the old parameter and its Euclidean gradient.
    # --------------------------
    kraus_op_old = mpstpcp_model.kraus_ops[0].detach().clone()
    grad_euclid = mpstpcp_model.kraus_ops[0].grad.detach().clone()

    # Retrieve the manifold object from the parameter. It is assumed that the
    # parameter has an attribute 'manifold' that implements the egrad2rgrad method.
    manifold = mpstpcp_model.kraus_ops[0].manifold
    # Compute the Riemannian gradient.
    rgrad_old = manifold.egrad2rgrad(kraus_op_old, grad_euclid)

    # --------------------------
    # 7) Perform one optimization step.
    # --------------------------
    optimizer.step()

    # --------------------------
    # 8) Check that the updated Kraus operator is on the Stiefel manifold.
    # For a Stiefel matrix U, we require UᵀU ≈ I.
    # --------------------------
    kraus_op_new = mpstpcp_model.kraus_ops[0].detach()
    check_eye = kraus_op_new.transpose(-1, -2) @ kraus_op_new
    identity_approx = torch.eye(check_eye.shape[-1], dtype=torch.float64)
    assert torch.allclose(check_eye, identity_approx, atol=1e-7), \
        "Kraus operator is not on the Stiefel manifold after the optimizer step."

    # --------------------------
    # 9) Check that the new parameter is approximately equal to:
    #      X_new ≈ X_old – lr * rgrad_old
    # (Note: Since the optimizer performs a retraction, the match is approximate.)
    # --------------------------
    approx_new = kraus_op_old - lr * rgrad_old * 2 # To compensate for the retraction
    max_diff = (kraus_op_new - approx_new).abs().max().item()
    assert torch.allclose(kraus_op_new, approx_new, atol=5e-2), \
        f"Kraus operator update deviates from the Euclidean step approximation (max diff={max_diff})."

    # --------------------------
    # 10) Forward pass after the step and check that loss decreased.
    # --------------------------
    outputs_after = mpstpcp_model(data)
    loss_after = loss_batch(outputs_after, target)
    assert loss_after.item() < loss_before_val, \
        "Loss did not decrease after one optimization step with StiefelSGD."

    print(f"[Small-scale] Loss before: {loss_before_val:.6f}, after: {loss_after.item():.6f}")

###############################################################################
# Test 2: Large-scale test using StiefelSGD.
###############################################################################

@pytest.mark.integtest
def test_mpstpcp_largeN_stiefel_sgd_integration():
    """
    A larger-scale integration test for MPSTPCP using StiefelSGD.
    In this test we use a higher number of sites (N) and multiple Kraus operators.
    The test verifies that:
      - The Kraus operators remain on the Stiefel manifold after each optimization step.
      - The loss decreases monotonically over several steps.
      - The approximate update of one Kraus operator is consistent with the
        Euclidean step (i.e. X_new ≈ X_old – lr * rgrad).
    """
    # --------------------------
    # 1) Fix random seeds.
    # --------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------
    # 2) Create synthetic data.
    # Let bs = batch size, N = number of sites (e.g. 16x16), d = local dimension.
    # --------------------------
    bs = 10
    N = 16 * 16  # 256 sites
    d = 2
    K = 3  # Use K=3 Kraus operators per site.
    data = torch.randn(bs, N, d, dtype=torch.float64)
    data = data / torch.norm(data, dim=-1, keepdim=True)
    # Binary labels.
    labels = torch.randint(0, 2, (bs,), dtype=torch.float64)

    # --------------------------
    # 3) Instantiate the MPSTPCP model.
    # --------------------------
    mpstpcp_model = MPSTPCP(
        N=N,
        K=K,
        d=d,
        with_identity=True,
        manifold=ManifoldType.EXACT  # You can change this to another type if desired.
    )
    mpstpcp_model.train()

    # --------------------------
    # 4) Set up the StiefelSGD optimizer.
    # --------------------------
    lr = 0.05
    momentum = 0.0
    optimizer = StiefelSGD(mpstpcp_model.kraus_ops.parameters(), lr=lr, momentum=momentum)

    # --------------------------
    # 5) Initial forward pass and loss.
    # --------------------------
    outputs = mpstpcp_model(data)
    loss = loss_batch(outputs, labels)
    loss_before = loss.item()

    optimizer.zero_grad()
    loss.backward()

    # --- For one of the Kraus operators, check the Euclidean update approximation. ---
    kraus_op_old = mpstpcp_model.kraus_ops[0].detach().clone()
    grad_euclid = mpstpcp_model.kraus_ops[0].grad.detach().clone()
    manifold = mpstpcp_model.kraus_ops[0].manifold
    rgrad_old = manifold.egrad2rgrad(kraus_op_old, grad_euclid)

    # --------------------------
    # 6) Perform one optimization step.
    # --------------------------
    optimizer.step()

    # Check manifold property.
    kraus_op_new = mpstpcp_model.kraus_ops[0].detach()
    check_eye = kraus_op_new.transpose(-1, -2) @ kraus_op_new
    identity_approx = torch.eye(check_eye.shape[-1], dtype=torch.float64)
    assert torch.allclose(check_eye, identity_approx, atol=1e-7), \
        "After one step, Kraus op is not on the Stiefel manifold."

    # Approximate update check.
    approx_new = kraus_op_old - lr * rgrad_old
    max_diff = (kraus_op_new - approx_new).abs().max().item()
    assert torch.allclose(kraus_op_new, approx_new, atol=5e-2), \
        f"After one step, Kraus operator update deviates (max diff={max_diff})."

    # Check that the loss decreased.
    outputs_after = mpstpcp_model(data)
    loss_after = loss_batch(outputs_after, labels)
    assert loss_after.item() < loss_before, \
        "Loss did not decrease after one optimization step in the large-N test."

    print(f"[Large-scale, step 1] Loss before: {loss_before:.6f}, after: {loss_after.item():.6f}")

    # --------------------------
    # 7) Multiple optimization steps: ensure loss decreases monotonically.
    # --------------------------
    n_steps = 5
    previous_loss = loss_after.item()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        outputs = mpstpcp_model(data)
        loss = loss_batch(outputs, labels)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()

        # Check manifold for one Kraus op.
        kraus_op_new = mpstpcp_model.kraus_ops[0].detach()
        check_eye = kraus_op_new.transpose(-1, -2) @ kraus_op_new
        assert torch.allclose(check_eye, identity_approx, atol=1e-7), \
            f"Step {step}: Kraus op left the Stiefel manifold."

        # Assert monotonic decrease (with a small tolerance).
        assert loss_val < previous_loss + 1e-12, f"Step {step}: Loss did not decrease."
        previous_loss = loss_val
        print(f"[Large-scale] Step {step}: loss = {loss_val:.6f}")
