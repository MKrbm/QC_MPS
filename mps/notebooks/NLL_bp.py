import opt_einsum as oe
import numpy as np
import torch
import sys
sys.path.append("../../")
import mps
from mps.trainer.data_utils import SyntheticDataset, SyntheticDatasetV2, SyntheticDatasetV3
from mps import tpcp_mps

import mps
from mps.trainer.utils import focal_loss
from mps.radam import RiemannianAdam
import geoopt
import random
import pandas as pd
import argparse


def grad_NLL(N, batch_size, dtype, lr, epochs):
    x0 = torch.randint(0, 2, (batch_size, N)).to(torch.float64)
    x0[:, 0] = 1
    x1 = x0.clone()
    x1[:, 0] = 0
    psi0 = torch.stack([x0, 1 - x0], dim=-1)
    psi1 = torch.stack([x1, 1 - x1], dim=-1)
    psi = torch.concat([psi0, psi1], dim=0)

    tpcp = tpcp_mps.MPSTPCP(
        N,
        K=1,
        d=2,
        enable_r=True,
        with_identity=False,
        manifold=tpcp_mps.ManifoldType.EXACT,
        dtype=dtype,
    )
    tpcp.train()

    optimizer = RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr)
    
    res = {
        "grad_first": [],
        "grad_last": [],
        "loss_list": [],
    }
    for _ in range(epochs):
        optimizer.zero_grad()
        out, reg = tpcp(psi.to(dtype), return_reg=True, return_probs=True)
        loss = focal_loss(out, psi[:, 0, 0])
        loss.backward()
        print("loss", loss.item(), "iter", _)
        res["loss_list"].append(loss.item())
        res["grad_first"].append(torch.norm(tpcp.kraus_ops.kraus_ops[0].grad).item())
        res["grad_last"].append(torch.norm(tpcp.kraus_ops.kraus_ops[-1].grad).item())
        optimizer.step()
    return res


def run_experiment(N, batch_size, lr, runs, epochs):
    res_list = []
    dtype = torch.complex128
    for _ in range(runs):
        seed = random.randint(0, 1000000)
        print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = grad_NLL(N, batch_size, dtype, lr, epochs)
        res_list.append(res)

    # Create dataframes: each column is one run's data and each row is an epoch.
    loss_df = pd.DataFrame({f'run_{i+1}': res['loss_list'] for i, res in enumerate(res_list)})
    grad_first_df = pd.DataFrame({f'run_{i+1}': res['grad_first'] for i, res in enumerate(res_list)})
    grad_last_df = pd.DataFrame({f'run_{i+1}': res['grad_last'] for i, res in enumerate(res_list)})

    # save the dataframes
    loss_df.to_csv(f"csv/loss_df_N{N}_bs{batch_size}_lr{lr}_runs{runs}_epochs{epochs}.csv", index=False)
    grad_first_df.to_csv(f"csv/grad_first_df_N{N}_bs{batch_size}_lr{lr}_runs{runs}_epochs{epochs}.csv", index=False)
    grad_last_df.to_csv(f"csv/grad_last_df_N{N}_bs{batch_size}_lr{lr}_runs{runs}_epochs{epochs}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLL experiment with specified parameters.")
    parser.add_argument('--N', type=int, default=10, help='Number of elements in the sequence.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate for the optimizer.')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs for the experiment.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for each run.')

    args = parser.parse_args()

    run_experiment(N=args.N, batch_size=args.batch_size, lr=args.lr, runs=args.runs, epochs=args.epochs)
