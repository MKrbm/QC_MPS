#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from mps.trainer.mpsae_trainer import mpsae_train
# from mps.trainer.data_utils import SyntheticDataset

# Synthetic dataset (same as in your original train_mpsae.py)
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, num_samples: int = 10000, seed: int | None = None):
        self.n = n
        self.num_samples = num_samples
        if seed is not None:
            np.random.seed(seed)
        self.labels = np.random.randint(0, 2, size=num_samples).astype(np.int64)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, index):
        l = self.labels[index]
        x = torch.zeros(self.n, dtype=torch.float64)
        x[0] = float(l)
        x_embedded = torch.stack([x, 1 - x], dim=-1)
        x_embedded = x_embedded / (x_embedded.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        return x_embedded, torch.tensor(l, dtype=torch.int64)

def main():
    parser = argparse.ArgumentParser(description="Train MPS Adiabatic Encoder (MPSAE) on synthetic data.")
    parser.add_argument("--n", type=int, default=256, help="Input vector dimension (default 256).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default 128).")
    parser.add_argument("--simple_epochs", type=int, default=1, help="Epochs for SimpleMPS training (default 1).")
    parser.add_argument("--mpsae_max_epochs", type=int, default=10, help="Max epochs for TPCP training (default 10).")
    parser.add_argument("--min_epochs", type=int, default=3, help="Min epochs before convergence (default 3).")
    parser.add_argument("--patience", type=int, default=2, help="Patience for convergence (default 2).")
    parser.add_argument("--conv_strategy", type=str, default="absolute", choices=["absolute", "relative"], help="Convergence strategy.")
    parser.add_argument("--conv_threshold", type=float, default=1e-4, help="Convergence threshold (default 1e-4).")
    parser.add_argument("--K", type=int, default=1, help="Number of Kraus operators per site (default 1).")
    parser.add_argument("--manifold", type=str, default="Exact", choices=["Exact", "Frobenius", "Canonical", "Cayley", "MatrixExp", "ForwardEuler", "Original"], help="Manifold type.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer (default adam).")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01).")
    parser.add_argument("--log_steps", type=int, default=10, help="Log steps (default 10).")
    parser.add_argument("--num_data", type=int, default=None, help="Number of samples (default 10000).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using fixed seed: {args.seed}")
    else:
        print("Using system-based randomness.")

    dataset = SyntheticDataset(n=args.n, num_samples=args.num_data or 10000, seed=args.seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _ = mpsae_train(
        dataloader=dataloader,
        device=device,
        N=args.n,
        d=2,
        l=2,
        layers=2,
        simple_epochs=args.simple_epochs,
        simple_lr=0.001,
        mps_optimize="greedy",
        manifold=args.manifold,
        optimizer_name=args.optimizer,
        K=args.K,
        max_epochs=args.mpsae_max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        conv_strategy=args.conv_strategy,
        conv_threshold=args.conv_threshold,
        lr=args.lr,
        log_steps=args.log_steps,
        weight_values=None,
        dtype=torch.float64,
    )

if __name__ == "__main__":
    main()
