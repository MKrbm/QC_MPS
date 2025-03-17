import opt_einsum as oe
import numpy as np
import torch
import sys

sys.path.append("../../")
import mps.simple_mps
import mps
from mps.trainer.data_utils import SyntheticDataset, SyntheticDatasetV2, SyntheticDatasetV3

N = 100
dataset = SyntheticDatasetV3(n=N, num_samples=(2**25), seed=42)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2**13, shuffle=True)

smps_params = torch.load("smps_2000_2.pth", weights_only=False)

N = 2000
dataset = SyntheticDatasetV3(n=N, num_samples=(2**25), seed=42)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2**10, shuffle=True)

chi = 2
d = 2
l = 2
device = torch.device("cpu")
dtype = torch.float64
optimize = "greedy"
eps = 1e-2
smps = mps.simple_mps.SimpleMPS(N, chi, d, l, layers=1, device=device, dtype=dtype, optimize=optimize, eps=eps)

smps.mps.set_params(smps_params)
smps.initialize_MPS()

from mps import tpcp_mps
from mps.trainer.utils import calculate_accuracy, focal_loss

# --- Step 2: Build and Prepare TPCP ---
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

logsoftmax = torch.nn.LogSoftmax(dim=-1)
nnloss = torch.nn.NLLLoss(reduction="mean")

# Initialize W: start with first column ones and second column small.

W = torch.zeros(tpcp.L, 2, dtype=torch.float64)
W[:, 0] = 1 
W[:, 1] = 0.0001
tpcp.initialize_W(W)
# W_now = tpcp.W.clone()
# W_new = update_weights(W_now, 0.001)
# tpcp.initialize_W(W_new)
# 
# --- Step 3: Determine lambda_final Using the Initial Loss Value ---
data_batch, target_batch = next(iter(dataloader))
initial_probs, reg = tpcp(data_batch, return_probs=True, return_reg=True)
# softmax_initial_probs = logsoftmax(initial_probs)
initial_accuracy = calculate_accuracy(initial_probs[:, 0], target_batch)
print(f"Initial accuracy: {initial_accuracy.item():.2%}")

loss = focal_loss(initial_probs[:, 0], target_batch)
print(f"Initial loss: {loss.item()}")

# from mps.trainer.adaptive_mpsae_trainer import RiemannianAdam
from geoopt import optim
from mps.StiefelOptimizers import StiefelAdam, StiefelSGD
import torch
lr = 0.00001
# optimizer = StiefelAdam(tpcp.kraus_ops.parameters(), lr=lr, expm_method='Cayley')
optimizer = optim.RiemannianAdam(tpcp.kraus_ops.parameters(), lr=lr, betas=(0.99, 0.999))
optimizer_weight = torch.optim.Adam([tpcp.r, tpcp.W], lr=lr)
tpcp.W.requires_grad = True
tpcp.r.requires_grad = True

clambda = 100
epochs = 1000
for _ in range(epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        optimizer_weight.zero_grad()
        outputs, reg = tpcp(data, return_probs=True, return_reg=True)
        # probs = logsoftmax(outputs)
        # loss0 = nnloss(probs, target)
        loss0 = focal_loss(outputs[:, 0], target)
        loss = loss0  + clambda * reg

        loss.backward()

        optimizer.step()
        optimizer_weight.step()

        tpcp.proj_stiefel(check_on_manifold=True, print_log=False, rtol=1e-3)
        tpcp.normalize_w_and_r()

        acc = calculate_accuracy(outputs[:, 0], target)
        srpq = torch.exp(-reg)
        print(f"Loss0 : {loss0.item():.6f}, reg: {reg.item():.6f}, Loss: {loss.item():.6f}, Acc: {acc.item():.2%}, SRPQ: {srpq.item():.6e}")
