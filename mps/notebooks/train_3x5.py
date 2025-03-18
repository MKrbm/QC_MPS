import opt_einsum as oe
import numpy as np
import torch
import sys
sys.path.append("../../")
import mps
from mps.trainer.data_utils import create_mnist_dataloader
from mps.trainer.smps_trainer import smps_train
from mps.simple_mps import SimpleMPS
from mps.trainer.smps_trainer import smps_train
from mps.trainer.utils import focal_loss
from mps import tpcp_mps
from mps.trainer.utils import calculate_accuracy, focal_loss, mean_risk
from mps.trainer.adaptive_mpsae_trainer import RiemannianAdam
from geoopt import optim
from mps.StiefelOptimizers import StiefelAdam, StiefelSGD
import torch
import pandas as pd
import os
import copy
import pandas as pd

import copy

img_size = 16
N = img_size * img_size
dataloader = create_mnist_dataloader(img_size=img_size, batch_size=2**12, allowed_digits=[3, 5])

# smps_params = torch.load("smps_2000.pth", weights_only=False)

chi = 2
d = 2
l = 2
device = torch.device("cpu")
dtype = torch.float64
optimize = "greedy"
eps = 1 / np.sqrt(N) * 0.1
smps = SimpleMPS(N, chi, d, l, layers=1, device=device, dtype=dtype, optimize=optimize, eps=eps)

epochs = 100
lr = 0.0001
softmax = torch.nn.Softmax(dim=-1)
nnloss = torch.nn.NLLLoss(reduction="mean")
opt_smps = torch.optim.Adam(smps.parameters(), lr=lr)
smps.train()
print(f"\n=== Training SimpleMPS for {epochs} epoch(s)... ===")

# Initialize a list to store epoch metrics
epoch_metrics = []

for epoch in range(epochs):
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        data = data.permute(1, 0, 2)  # [batch, N, 2] → [N, batch, 2]
        opt_smps.zero_grad()
        outputs = smps(data)
        outputs = torch.abs(outputs)
        outputs = softmax(outputs)
        loss = focal_loss(outputs[:, 0], target, gamma=0.0, alpha=0.5)
        loss.backward()
        
        # Print the norm of the gradients for 10 equally split indices
        params = list(smps.parameters())
        num_params = len(params)
        indices = [int(i * num_params / 10) for i in range(10)]
        grad_norms = [f"Gradient norm for parameter {idx}: {params[idx].grad.norm().item():.6f}" for idx in indices if params[idx].grad is not None]
        print(" | ".join(grad_norms))
        
        opt_smps.step()
        bs = target.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        preds = outputs.argmax(dim=-1)
        acc = (preds == target).float().sum().item()
        total_correct += acc
        print(f"[SimpleMPS] Epoch {epoch+1}, Step {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.6f} | Acc: {acc/bs:.2%}")
    
    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    epoch_metrics.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_acc})
    print(f"[SimpleMPS] Epoch {epoch+1} | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.2%}")

# Convert epoch metrics to a DataFrame and save to CSV
df_metrics = pd.DataFrame(epoch_metrics)
df_metrics.to_csv('csv/smps_3x5_epoch_metrics.csv', index=False)

# --- Step 2: Build and Prepare TPCP ---
tpcp = tpcp_mps.MPSTPCP(
    N,
    K=2,
    d=2,
    enable_r=True,
    with_identity=False,
    manifold=tpcp_mps.ManifoldType.EXACT,
)
tpcp.set_canonical_mps(smps)
tpcp.train()


# Predefined schedule for W[:, 1]
w_schedule = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512] + [0.6, 0.7, 0.8, 0.9] + [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
epoch_schedule = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30] + [30, 30, 30, 30] + [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
lr_schedule = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005] + [0.0005, 0.0005, 0.0005, 0.0005] + [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

assert len(w_schedule) == len(epoch_schedule) == len(lr_schedule)

# Define the file path for storing epoch metrics
epoch_metrics_file_path = "csv/tpcp_3x5_epoch_metrics.csv"
# Define the file path for storing iteration metrics
iter_metrics_file_path = "csv/tpcp_3x5_iter_metrics.csv"

# Initialize the epoch metrics file with headers if it exists
if os.path.exists(epoch_metrics_file_path):
    print("Epoch metrics file exists. Initializing it.")
    with open(epoch_metrics_file_path, 'w') as f:
        f.write("w_value,epoch,avg_loss,avg_loss_with_reg,avg_acc,avg_srpq,avg_reg\n")

# Initialize the iteration metrics file with headers if it exists
if os.path.exists(iter_metrics_file_path):
    print("Iteration metrics file exists. Initializing it.")
    with open(iter_metrics_file_path, 'w') as f:
        f.write("w_value,epoch,iter,loss0,reg,loss,acc,srpq\n")

for i in range(len(w_schedule)):
    w_value = w_schedule[i]
    epochs = epoch_schedule[i]
    lr = lr_schedule[i]
    print(f"Training TPCP with w_value: {w_value}, epochs: {epochs}, lr: {lr}")
    W = torch.zeros(tpcp.L, 2, dtype=torch.float64)
    W[:, 0] = 1 
    W[:, 1] = w_value
    tpcp.initialize_W(W)
    # W_now = tpcp.W.clone()
    # W_new = update_weights(W_now, 1)
    # tpcp.initialize_W(W_new)
    # torch.save(tpcp.state_dict(), f"pth/tpcp_state_dict_new_3x5_256_w_{w_value}.pth")
    # 
    # --- Step 3: Determine lambda_final Using the Initial Loss Value ---
    data_batch, target_batch = next(iter(dataloader))
    initial_probs, reg = tpcp(data_batch, return_probs=True, return_reg=True)
    initial_accuracy = calculate_accuracy(initial_probs[:, 0], target_batch)
    srpq = torch.exp(-reg)
    sr_sys = torch.exp(-reg*N)
    print("srpq: ", srpq)
    print("success rate : ", sr_sys)
    print(f"Initial accuracy: {initial_accuracy.item():.2%}")
    # loss = loss_batch(initial_probs[:, 0], target_batch)
    loss = focal_loss(initial_probs[:, 0], target_batch, gamma=0.0, alpha = 0.5)
    # ls = logsoftmax(initial_probs)
    # loss = nnloss(ls, target_batch)
    print(f"Initial loss: {loss.item()}")
    # optimizer = StiefelAdam(tpcp.kraus_ops.parameters(), lr=lr)
    optimizer = optim.RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr, betas=(0.99, 0.999))
    optimizer_weight = torch.optim.Adam([tpcp.r, tpcp.W], lr=lr * 0.1)
    tpcp.W.requires_grad = True
    tpcp.r.requires_grad = True


    clambda = 1

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        epoch_loss_with_reg_sum = 0.0
        epoch_srpq_sum = 0.0
        epoch_reg_sum = 0.0
        epoch_w = 0.0
        total_samples = 0

        for iter_num, (data, target) in enumerate(dataloader, start=1):
            optimizer.zero_grad()
            optimizer_weight.zero_grad()
            outputs, reg = tpcp(data, return_probs=True, return_reg=True)
            loss0 = focal_loss(outputs, target, gamma=0.0, alpha = 0.5)
            risk = mean_risk(outputs, target)
            # ls = logsoftmax(outputs)
            # loss0 = nnloss(ls, target)
            loss = loss0 + clambda * reg

            loss.backward()

            optimizer.step()
            optimizer_weight.step()

            tpcp.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)
            tpcp.normalize_w_and_r()

            acc = calculate_accuracy(outputs[:, 0], target)
            srpq = torch.exp(-reg)

            bs = target.size(0)
            epoch_loss_sum += loss0.item() * bs
            epoch_loss_with_reg_sum += loss.item() * bs
            total_samples += bs
            epoch_acc_sum += acc.item() * bs
            epoch_srpq_sum += srpq.item() * bs
            epoch_reg_sum += reg.item() * bs
            epoch_w += tpcp.W.data[:, 1].mean().item() * bs
            # Log iteration metrics
            print(f"Epoch {epoch+1}, Iter {iter_num}, Loss0: {loss0.item():.6f}, reg: {reg.item():.6f}, Loss: {loss.item():.6f}, Acc: {acc.item():.2%}, SRPQ: {srpq.item():.6e}, Risk: {risk.item():.6f}, W: {tpcp.W.data[:, 1].mean().item():.6f}")

            # Write iteration metrics to file
            with open(iter_metrics_file_path, 'a') as f:
                f.write(f"{w_value},{epoch+1},{iter_num},{loss0.item():.6f},{reg.item():.6f},{loss.item():.6f},{acc.item():.6f},{srpq.item():.6e}\n")

        # End of epoch: compute averages.
        avg_loss = epoch_loss_sum / total_samples
        avg_loss_with_reg = epoch_loss_with_reg_sum / total_samples
        avg_acc = epoch_acc_sum / total_samples
        avg_srpq = epoch_srpq_sum / total_samples
        avg_reg = epoch_reg_sum / total_samples
        avg_w = epoch_w / total_samples

        # Log epoch metrics
        print(f"Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.6f} | Avg Loss+Reg: {avg_loss_with_reg:.6f} | Acc: {avg_acc:.2%} | SRPQ: {avg_srpq:.6f} | Reg: {avg_reg:.6f} | W: {avg_w:.6f}")


        # Write epoch metrics to file
        with open(epoch_metrics_file_path, 'a') as f:
            f.write(f"{w_value},{epoch+1},{avg_loss:.6f},{avg_loss_with_reg:.6f},{avg_acc:.6f},{avg_srpq:.6f},{avg_reg:.6f},{avg_w:.6f}\n")
        
        W_now = tpcp.W.clone()
        # with torch.no_grad():
        #     torch.save(tpcp.state_dict(), f"pth/tpcp_state_dict_3x5_256_w_{w_value}.pth")

