
import opt_einsum as oe
import numpy as np
import torch
import sys
import pandas as pd
import argparse
sys.path.append("../../")
import mps
from mps.trainer.data_utils import create_mnist_dataloader, SyntheticDataset, SyntheticDatasetV2, SyntheticDatasetV3
from mps import tpcp_mps
from mps.trainer.utils import calculate_accuracy, mean_risk, focal_loss
from mps.radam import RiemannianAdam

def run_experiment(N, epochs, lr, output_file, with_eps=True, with_ps=False):
    dataset = SyntheticDatasetV3(n=N, num_samples=2**15, seed=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2**10, shuffle=True)

    from mps.trainer.smps_trainer import smps_train
    from mps.simple_mps import SimpleMPS
    import copy

    chi = 2
    d = 2
    l = 2
    device = torch.device("cpu")
    dtype = torch.float64
    optimize = "greedy"
    if with_eps:
        eps = 1 / np.sqrt(N) * 0.1
    else:
        eps = 0

    smps = SimpleMPS(N, chi, d, l, layers=1, device=device, dtype=dtype, optimize=optimize, eps=eps)

    tpcp = tpcp_mps.MPSTPCP(
        N,
        K=1,
        d=2,
        enable_r=True,
        with_identity=False,
        manifold=tpcp_mps.ManifoldType.EXACT,
    )
    tpcp.train()
    tpcp.set_canonical_mps(smps)

    W = torch.zeros(tpcp.L, 2, dtype=torch.float64)
    W[:, 0] = 1
    W[:, 1] = 0 if with_ps else 1
    tpcp.initialize_W(W)

    lr = 0.0001
    optimizer = RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr)
    optimizer_weight = torch.optim.Adam([tpcp.r, tpcp.W], lr=lr)
    tpcp.W.requires_grad = False
    tpcp.r.requires_grad = True

    metrics_data = []

    # Define CSV column names
    column_names = "Epoch,Iter,Loss0,reg,Loss,Acc,SRPQ\n"

    # Check if the file exists and write column names if it doesn't
    try:
        with open(output_file, 'x') as f:
            f.write(column_names)
    except FileExistsError:
        pass

    for epoch in range(epochs):
        for iter_num, (data, target) in enumerate(dataloader, start=1):
            data_flipped = data.clone()
            data_flipped[:, 0] = 1 - data_flipped[:, 0]
            target_flipped = 1 - target

            data = torch.cat((data, data_flipped), dim=0)
            target = torch.cat((target, target_flipped), dim=0)

            optimizer.zero_grad()
            optimizer_weight.zero_grad()
            outputs, reg = tpcp(data, return_probs=True, return_reg=True)
            loss0 = focal_loss(outputs, target, gamma=0.0, alpha=0.5)

            loss = loss0

            loss0.backward()

            optimizer.step()
            optimizer_weight.step()

            tpcp.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)
            tpcp.normalize_w_and_r()

            acc = calculate_accuracy(outputs[:, 0], target)
            srpq = torch.exp(-reg)

            metrics_data.append({
                "Epoch": epoch + 1,
                "Iter": iter_num,
                "Loss0": loss0.item(),
                "reg": reg.item(),
                "Loss": loss.item(),
                "Acc": acc.item(),
                "SRPQ": srpq.item()
            })
            print(f"Epoch {epoch+1}, Iter {iter_num}, Loss0: {loss0.item():.6f}, reg: {reg.item():.6f}, Loss: {loss.item():.6f}, Acc: {acc.item():.2%}, SRPQ: {srpq.item():.6e}")

            with open(output_file, 'a') as f:
                f.write(f"{epoch + 1},{iter_num},{loss0.item()},{reg.item()},{loss.item()},{acc.item()},{srpq.item()}\n")
            
            if acc.item() > 0.99:
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPS experiment and save metrics to CSV.")
    parser.add_argument("--N", type=int, default=100, help="Number of features.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--with_eps", type=bool, default=True, help="Use epsilon in the experiment.")
    parser.add_argument("--with_ps", type=bool, default=False, help="with postselection in the experiment.")

    args = parser.parse_args()

    output_file = f"csv/bp_v3_si_N{args.N}_epochs{args.epochs}_lr{args.lr}_with_eps{args.with_eps}_with_ps{args.with_ps}.csv"

    run_experiment(args.N, args.epochs, args.lr, output_file, args.with_eps, args.with_ps)

