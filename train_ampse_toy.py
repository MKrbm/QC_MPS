#!/usr/bin/env python3

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import geoopt

# If your mps/ package is local, ensure it’s on sys.path or installed in editable mode:
# sys.path.append("/path/to/your/project/root")

from mps.simple_mps import SimpleMPS
from mps.tpcp_mps import MPSTPCP, ManifoldType
from mps.StiefelOptimizers import StiefelAdam, StiefelSGD
from mps.radam import RiemannianAdam


###############################################################################
# Synthetic Dataset for n-dimensional vectors
###############################################################################
class SyntheticDataset(torch.utils.data.Dataset):
    """
    Each sample is an n-dimensional vector where the first element is either 0 or 1,
    and the remaining n-1 entries are 0. The label is the first element.
    We embed each scalar x to a 2-d vector [x, 1-x] and then L2-normalize.
    """
    def __init__(self, n: int, num_samples: int = 10000, seed: int | None = None):
        """
        Args:
            n: Dimension of the vector.
            num_samples: Number of samples in the dataset.
            seed: Random seed (optional).
        """
        self.n = n
        self.num_samples = num_samples
        if seed is not None:
            np.random.seed(seed)
        # Randomly assign labels (0 or 1) for each sample
        self.labels = np.random.randint(0, 2, size=num_samples).astype(np.int64)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # The label is either 0 or 1
        l = self.labels[index]
        # Create a vector of length n with first element l and rest zeros
        x = torch.zeros(self.n, dtype=torch.float64)
        x[0] = float(l)
        # Embed each scalar: map x -> [x, 1-x]
        x_embedded = torch.stack([x, 1 - x], dim=-1)  # shape: [n, 2]
        # Normalize each site so that the two entries sum to 1 (almost redundant here)
        # (If you prefer an L2 normalization across the 2 components, you can adjust this.)
        x_embedded = x_embedded / (x_embedded.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        # The target label is l
        return x_embedded, torch.tensor(l, dtype=torch.int64)


###############################################################################
# Loss & Accuracy (unchanged)
###############################################################################
def loss_batch(outputs, labels):
    """
    Binary cross-entropy style loss for outputs in [0, 1].
    For label=0 => use outputs[i], for label=1 => use 1 - outputs[i].
    """
    device = outputs.device
    loss_val = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss_val -= torch.log(prob + 1e-8)
        if torch.isnan(loss_val):
            print(f"NaN in loss at index {i}, prob={prob}, output={outputs[i]}, label={labels[i]}")
    return loss_val


def calculate_accuracy(outputs, labels):
    """
    Threshold outputs at 0.5 to assign label 0 or 1 and compare to true labels.
    """
    predictions = (outputs < 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / labels.numel()

def to_probs(outputs):
    outputs = outputs / outputs.sum(dim=-1).unsqueeze(-1)
    return outputs


###############################################################################
# Main Training Function (modified for synthetic dataset)
###############################################################################
def train_tpcp_synthetic(
    n: int = 256,
    optimizer_name: str = "adam",
    manifold: str = "Exact",
    K: int = 1,
    batch_size: int = 128,
    max_epochs: int = 10,
    conv_strategy: str = "absolute",
    conv_threshold: float = 1e-4,
    min_epochs: int = 3,
    patience: int = 2,
    num_data: int | None = None,
    lr: float = 0.01,
    log_steps: int = 10,
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
):
    """
    Train the TPCP MPS model on a synthetic dataset where each sample is an n-dimensional vector.
    The input is |l> ⊗ |0> ⊗ ... ⊗ |0> (after embedding) and the label is l (0 or 1).
    """
    # --------------------------
    # 1) Prepare Synthetic Dataset
    # --------------------------
    # Create the synthetic dataset with n sites
    synthetic_dataset = SyntheticDataset(n=n, num_samples=num_data or 10000)
    dataloader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)

    # --------------------------
    # 2) Train a SimpleMPS model (demo)
    # --------------------------
    # Here N = n (number of sites). Data input dimension d = 2 (after embedding) and label dimension l = 2.
    N = n
    d = l = 2  
    smps = SimpleMPS(
        N,
        2,
        d,
        l,
        layers=2,
        device=device,
        dtype=torch.float64,
        optimize="greedy",
    )
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    nnloss = torch.nn.NLLLoss(reduction="mean")
    opt_smps = torch.optim.Adam(smps.parameters(), lr=0.001)

    smps_losses = []
    smps.train()
    print("\n=== Training SimpleMPS for 1 epoch... ===")
    total_loss_smps = 0.0
    total_samples_smps = 0
    total_correct_smps = 0


    for epoch in range(10):
      total_loss_smps = 0.0
      total_samples_smps = 0
      total_correct_smps = 0
      for batch_idx, (data, target) in enumerate(dataloader):
          # data is of shape [batch_size, n, 2], we need to permute to [n, batch_size, 2]
          data, target = data.to(device), target.to(device)
          data = data.permute(1, 0, 2)
          opt_smps.zero_grad()
          outputs = smps(data)  # shape: [batch_size, 2]
          outputs = logsoftmax(outputs)
          loss = nnloss(outputs, target)
          loss.backward()
          opt_smps.step()

          bs = target.size(0)
          total_loss_smps += loss.item() * bs
          total_samples_smps += bs

          # Calculate batch accuracy
          preds = outputs.argmax(dim=-1)
          total_correct_smps += (preds == target).float().sum().item()

      # Calculate and print epoch loss and accuracy
      epoch_loss = total_loss_smps / total_samples_smps
      epoch_accuracy = total_correct_smps / total_samples_smps
      smps_losses.append(epoch_loss)
      print(f"[SimpleMPS] Epoch {epoch+1} | Loss: {epoch_loss:.6f} | Accuracy: {epoch_accuracy:.2%}")


    # --------------------------
    # 3) Build TPCP model with chosen manifold
    # --------------------------
    manifold_map = {
        "Exact": ManifoldType.EXACT,
        "Frobenius": ManifoldType.FROBENIUS,
        "Canonical": ManifoldType.CANONICAL,
        "Cayley": ManifoldType.EXACT,
        "MatrixExp": ManifoldType.EXACT,
        "ForwardEuler": ManifoldType.EXACT,
        "Original": ManifoldType.EXACT,  # use EXACT underlying type for "Original"
    }
    if manifold not in manifold_map:
        raise ValueError(f"Invalid manifold='{manifold}'. Valid options: {list(manifold_map.keys())}.")

    tpcp = MPSTPCP(N, K=K, d=2, with_probs=False, with_identity=True, manifold=manifold_map[manifold])
    tpcp.to(device)
    tpcp.train()
    # Optionally initialize TPCP from the SimpleMPS (canonical form)
    tpcp.set_canonical_mps(smps)

    # --------------------------
    # 4) Choose an optimizer for TPCP
    # --------------------------
    if manifold in ["Cayley", "MatrixExp", "ForwardEuler"]:
        if optimizer_name.lower() == "adam":
            opt_tpcp = StiefelAdam(tpcp.kraus_ops.parameters(), lr=lr, expm_method=manifold)
        elif optimizer_name.lower() == "sgd":
            opt_tpcp = StiefelSGD(tpcp.kraus_ops.parameters(), lr=lr, expm_method=manifold)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
    elif manifold in ["Exact", "Frobenius", "Canonical"]:
        if optimizer_name.lower() == "adam":
            opt_tpcp = geoopt.optim.RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            opt_tpcp = geoopt.optim.RiemannianSGD(tpcp.kraus_ops.parameters(), lr=lr)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
    elif manifold == "Original":
        if optimizer_name.lower() == "adam":
            opt_tpcp = RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            raise ValueError("SGD is not supported for the original update rule.")
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")

    # --------------------------
    # 5) Training TPCP MPS with convergence checking and metrics recording
    # --------------------------
    # We iterate over different weighting factors w.
    w_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tpcp_metrics_by_w = dict()

    for w_ in w_values:
        # Re-initialize weight parameter W for current w_
        W = torch.zeros(tpcp.L, 2, dtype=torch.float64, device=device)
        W[:, 0] = 1
        W[:, 1] = w_
        tpcp.initialize_W(W)

        print(f"\n=== Training TPCP with initial weight rate w={w_:.1f} (max_epochs={max_epochs}) ===")
        epoch = 0
        prev_epoch_loss = None
        conv_counter = 0  # counts consecutive epochs satisfying convergence condition
        metrics = {"loss": [], "accuracy": [], "weight_rate": []}

        while epoch < max_epochs:
            epoch_loss_sum = 0.0
            total_samples = 0
            epoch_acc_sum = 0.0
            total_acc_samples = 0
            t0 = time.time()
            for step, (data, target) in enumerate(dataloader):
                data = data.to(device)
                target = target.to(device)
                bs = target.size(0)

                opt_tpcp.zero_grad()
                outputs = tpcp(data)
                outputs = to_probs(outputs)[:, 0] #calculate the probability of label 0
                if torch.isnan(outputs).any():
                    print(f"NaN detected in outputs at step {step}, skipping batch.")
                    continue

                loss_val = loss_batch(outputs, target)
                if torch.isnan(loss_val):
                    print("NaN detected in batch loss, skipping batch.")
                    continue

                loss_val.backward()
                opt_tpcp.step()

                # Optionally project back to the manifold
                tpcp.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)

                epoch_loss_sum += loss_val.item() * bs
                total_samples += bs
                batch_acc = calculate_accuracy(outputs.detach(), target)
                epoch_acc_sum += batch_acc.item() * bs
                total_acc_samples += bs

                if (step + 1) % log_steps == 0:
                    print(f"[TPCP::w={w_:.1f}] Epoch {epoch+1}, Step {step+1}/{len(dataloader)} | Batch Loss: {loss_val.item():.6f} | Batch Accuracy: {batch_acc.item():.2%}")

            if total_samples == 0:
                print("No samples processed in this epoch, skipping.")
                continue

            avg_loss = epoch_loss_sum / total_samples
            avg_acc = epoch_acc_sum / total_acc_samples

            # Compute the current weight rate (average over sites)
            current_weight_rate = (tpcp.W[:, 1] / torch.sum(tpcp.W, dim=1)).mean().item() if hasattr(tpcp, "W") else w_

            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(avg_acc)
            metrics["weight_rate"].append(current_weight_rate)

            elapsed = time.time() - t0
            print(f"[TPCP::w={w_:.1f}] Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}, Avg Acc: {avg_acc:.2%}, Weight Rate: {current_weight_rate:.4f} | Time: {elapsed:.2f}s")

            # Check convergence only after min_epochs have been reached
            if epoch >= min_epochs:
                if prev_epoch_loss is not None:
                    if conv_strategy == "absolute":
                        condition = abs(avg_loss - prev_epoch_loss) < conv_threshold
                    elif conv_strategy == "relative":
                        condition = abs(avg_loss - prev_epoch_loss) / (abs(prev_epoch_loss) + 1e-8) < conv_threshold
                    else:
                        condition = False

                    if condition:
                        conv_counter += 1
                    else:
                        conv_counter = 0

                    if conv_counter >= patience:
                        print(f"Convergence reached: Condition met for {conv_counter} consecutive epochs.")
                        break
            prev_epoch_loss = avg_loss
            epoch += 1

        tpcp_metrics_by_w[w_] = metrics

    return {"simple_mps_losses": smps_losses, "tpcp_metrics_by_w": tpcp_metrics_by_w}


def main():
    parser = argparse.ArgumentParser(description="Train TPCP MPS on a synthetic dataset.")

    parser.add_argument(
        "--manifold",
        default="Exact",
        choices=["Exact", "Frobenius", "Canonical", "Cayley", "MatrixExp", "ForwardEuler", "Original"],
        help="Manifold/update rule for Kraus ops (default Exact).",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd"],
        help="Riemannian optimizer to use (default adam).",
    )
    parser.add_argument("--K", type=int, default=1, help="# of Kraus operators per site (default 1).")
    parser.add_argument("--n", type=int, default=256, help="Dimension of the input vector (default 256).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default 128).")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs for TPCP training (default 10).")
    parser.add_argument("--min_epochs", type=int, default=3, help="Minimum epochs before convergence is checked (default 3).")
    parser.add_argument("--patience", type=int, default=2, help="Number of consecutive epochs meeting convergence condition required to stop (default 2).")
    parser.add_argument("--conv_strategy", type=str, default="absolute", choices=["absolute", "relative"],
                        help="Convergence strategy for loss (absolute or relative).")
    parser.add_argument("--conv_threshold", type=float, default=1e-4, help="Threshold for loss convergence (default 1e-4).")
    parser.add_argument("--num_data", type=int, default=None, help="Limit dataset to N examples (default: 10000).")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01).")
    parser.add_argument("--log_steps", type=int, default=10, help="Steps between logging loss values (default 10).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: system-based).")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using fixed random seed: {args.seed}")
    else:
        print("Using system-based randomness.")

    results = train_tpcp_synthetic(
        n=args.n,
        manifold=args.manifold,
        optimizer_name=args.optimizer,
        K=args.K,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        conv_strategy=args.conv_strategy,
        conv_threshold=args.conv_threshold,
        num_data=args.num_data,
        lr=args.lr,
        log_steps=args.log_steps,
    )

    # --------------------------
    # Plotting: For each w value, plot loss, accuracy, and weight_rate on the same axes.
    # --------------------------
    tpcp_metrics_by_w = results["tpcp_metrics_by_w"]

    # Flatten metrics from different weight runs in the order of increasing w.
    all_loss = []
    all_accuracy = []
    all_weight_ratio = []
    for w_val in sorted(tpcp_metrics_by_w.keys()):
        metrics = tpcp_metrics_by_w[w_val]
        all_loss.extend(metrics["loss"])
        all_accuracy.extend(metrics["accuracy"])
        all_weight_ratio.extend(metrics["weight_rate"])

    # Create a cumulative x-axis: one unit per epoch across all weight values.
    x_axis = range(1, len(all_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    # Offset the right spine of ax3.
    ax3.spines['right'].set_position(('outward', 60))

    ax1.plot(x_axis, all_loss, label="Loss", marker="o", color='tab:blue')
    ax2.plot(x_axis, all_accuracy, label="Accuracy", marker="s", color='tab:orange')
    ax3.plot(x_axis, all_weight_ratio, label="Weight Ratio", marker="^", color='tab:green')

    ax1.set_xlabel("Epoch (Cumulative)")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax2.set_ylabel("Accuracy", color='tab:orange')
    ax3.set_ylabel("Weight Ratio", color='tab:green')

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    ax2.set_ylim(0, 1)  # Set accuracy range from 0 to 1

    fig.suptitle("Training Metrics (Loss, Accuracy, Weight Ratio) over Epochs")
    fig.tight_layout()

    # Save the figure
    plt.savefig("training_metrics.png")
    plt.show()


if __name__ == "__main__":
    main()
