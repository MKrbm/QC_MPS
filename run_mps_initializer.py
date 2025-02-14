
import opt_einsum as oe
import numpy as np
import torch
from mps import simple_mps, tpcp_mps
from mps.radam import RiemannianAdam
from mps.StiefelOptimizers import StiefelAdam
import geoopt


def filter_digits(dataset, allowed_digits=[0, 1]):
    """Return a subset of MNIST dataset containing only allowed_digits (0 or 1)."""
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in allowed_digits:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)


def filiter_single_channel(img: torch.Tensor) -> torch.Tensor:
    """
    MNIST is loaded as shape [C, H, W].
    Take only the first channel => shape [H, W].
    """
    return img[0, ...]


def embedding_pixel(batch, label: int = 0):
    """
    Flatten each image from shape [H, W] => [H*W],
    then embed x => [x, 1-x], and L2-normalize along last dim.
    """
    pixel_size = batch.shape[-1] * batch.shape[-2]
    x = batch.view(*batch.shape[:-2], pixel_size)
    x = torch.stack([x, 1 - x], dim=-1)
    x = x / torch.sum(x, dim=-1).unsqueeze(-1)
    return x

def loss_batch(outputs, labels):
    """
    Binary cross-entropy style loss for outputs in [0, 1].
    For label=0 => prob=outputs[i], else => 1 - outputs[i].
    """
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)
    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else (1 - outputs[i])
        loss -= torch.log(prob + 1e-8)
        # Start of Selection
        if torch.isnan(loss):
            print(f"Loss is NaN at i={i}")
            print(prob, outputs[i], labels[i])
    return loss


def calculate_accuracy(outputs, labels):
    """
    Threshold 0.5 => label 0 or 1. Compare to true labels.
    """
    predictions = (outputs < 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / labels.numel()

from torchvision import transforms
import torchvision

img_size = 16
transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(filiter_single_channel),
        transforms.Lambda(embedding_pixel),
        transforms.Lambda(lambda x: x.to(torch.float64)),  # double precision
    ]
)

trainset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
# Filter digits 0,1 only
trainset = filter_digits(trainset, allowed_digits=[0, 1])

batch_size = 128

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False
)


# ---------- Build MPS model ----------
N = img_size * img_size
d = l = 2 #data input dimension and class label dimension 
chi_umps = 2
chi_max = 2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
smps = simple_mps.SimpleMPS(
    N, 
    2,
    d, 
    l, 
    layers=2,
    device=device, 
    dtype=torch.float64, 
    optimize="greedy",
)

def accuracy(outputs, target):
    return (outputs.argmax(dim=-1) == target).float().mean()

losses = []
running_loss = 0
running_accuracy = 0
logsoftmax = torch.nn.LogSoftmax(dim=-1)
nnloss = torch.nn.NLLLoss(reduction="mean")
optimizer = torch.optim.Adam(smps.parameters(), lr=0.001)
n_samples = 0
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(trainloader):
        target = target.to(device).to(torch.int64)
        data = data.to(device).permute(1, 0, 2)
        optimizer.zero_grad()
        outputs = smps(data)
        outputs = logsoftmax(outputs)
        loss = nnloss(outputs, target)
        loss.backward()
        optimizer.step()

        data_size = data.shape[1]
        
        # Calculate accuracy
        # print(torch.exp(outputs[:10]), target[:10])
        
        running_loss += loss.item() * data_size
        n_samples += data_size
        
        if batch_idx % 1 == 0:
            avg_loss = running_loss / n_samples
            avg_accuracy = accuracy(outputs, target)
            losses.append(avg_loss)
            running_loss = 0
            running_accuracy = 0
            n_samples = 0
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.2f}%'.format(
                epoch, batch_idx * data_size, len(trainloader.dataset),
                100. * batch_idx / len(trainloader), avg_loss, avg_accuracy * 100))


tpcp = tpcp_mps.MPSTPCP(N, K=1, d=2, with_identity=True, manifold=tpcp_mps.ManifoldType.EXACT)

W = torch.zeros(tpcp.L, 2, dtype=torch.float64)

# optimizer = StiefelAdam(tpcp.parameters(), lr=0.0001, betas=(0.9, 0.999))
tpcp.set_canonical_mps(smps)
optimizer = RiemannianAdam(tpcp.parameters(), lr=0.0002, betas=(0.9, 0.999))

for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    W[:, 0] = 1
    W[:, 1] = w
    tpcp.initialize_W(W)
    epochs = 15
    for epoch in range(epochs):
        acc_tot = 0
        loss_tot = 0
        for batch_idx, (data, target) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = tpcp(data)
            loss = loss_batch(outputs, target)
            loss.backward()
            optimizer.step()
            acc = calculate_accuracy(outputs, target)
            acc_tot += acc
            loss_tot += loss.item()
            if batch_idx % 10 == 0:
                print("Loss: ", loss.item(), "Accuracy: ", acc)

        print(f"Epoch {epoch} / {epochs} / Loss: {loss_tot / len(trainloader)} / Accuracy: {acc_tot / len(trainloader)} / Weight Ratio: {w / (1 - w)}")
