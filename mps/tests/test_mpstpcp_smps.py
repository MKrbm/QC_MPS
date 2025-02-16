#!/usr/bin/env python3
import torch
import pytest
import numpy as np

# Import the smps training function and the MPSTPCP model definitions.
from mps.simple_mps import SimpleMPS
from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps.trainer import smps_trainer

@pytest.mark.unit
def test_mpstpcp_smps_equivalence():
    """
    Test that MPSTPCP produces the same output as SimpleMPS when:
      - The MPSTPCP canonical MPS is set from the trained smps.
      - W is initialized as:
            W = torch.zeros(tpcp.L, 2, dtype=dtype, device=device)
            W[:, 0] = 1
            W[:, 1] = w_val
    This ensures that the canonical initialization makes both models behave equivalently.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    dtype = torch.float64

    # Define parameters for the test.
    bs = 30         # batch size
    N = 16          # number of sites (qubits)
    d = 2           # physical dimension
    l = 2           # some parameter for SMPS (e.g., bond dimension-related)

    # Create synthetic input data of shape (batch_size, N, d)
    data = torch.rand(bs, N, d, dtype=dtype, device=device)
    data = data / torch.sum(data, dim=-1, keepdim=True)

    # Create a dummy dataloader for smps_train (a list with one batch is enough for this test)
    target = torch.randint(0, l, (bs,), dtype=torch.int64, device=device)
    dummy_loader = [(data, target)]

    # Initialize (or "train") a SimpleMPS via the smps_train helper. We use 1 epoch for initialization.
    smps = smps_trainer.smps_train(
        dummy_loader,
        N = N,
        d = d,
        l = l,
        epochs = 100,
        lr = 0.01,
        log_steps = 1,
        dtype = dtype,
        device = device,
    )
    smps.to(device)
    smps.train()

    # Instantiate MPSTPCP with EXACT manifold mode.
    tpcp = MPSTPCP(
        N = N,
        K = 1,
        d = d,
        enable_r = False,
        with_identity = True,
        manifold = ManifoldType.EXACT,
    )
    tpcp.to(device)
    tpcp.train()

    # Use the initialized SimpleMPS to set the canonical MPS in MPSTPCP.
    tpcp.set_canonical_mps(smps)

    # Initialize W as specified.
    # Note: tpcp.L is expected to be the number of sites (or equivalently N) that determine W's shape.
    W = torch.zeros(tpcp.L, 2, dtype=dtype, device=device)
    W[:, 0] = 1
    W[:, 1] = 0
    tpcp.initialize_W(W)

    # Compute the forward pass outputs of both models.
    out_smps = smps(data.permute(1, 0, 2)).detach().cpu().numpy()
    out_tpcp = tpcp(data, return_probs=True).detach().cpu().numpy()

    res_smps = out_smps[:, 0] < out_smps[:, 1]
    res_tpcp = out_tpcp[:, 0] < out_tpcp[:, 1]


    assert np.allclose(res_smps.astype(float), target), "SimpleMPS failed to classify all samples correctly"
    assert np.allclose(res_smps, res_tpcp), "MPSTPCP and SimpleMPS produce different results"