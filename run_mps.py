import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data
import numpy as np

from mps import umps
from mps import unitary_optimizer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def embedding_pixel(batch, label: int = 0):
    pixel_size = batch.shape[-1] * batch.shape[-2]
    x = batch.view(*batch.shape[:-2], pixel_size)
    # x[:] = 0
    x = torch.stack([x, 1 - x], dim=-1)
    # x = x / torch.sum(x, dim=-1).unsqueeze(-1)
    x = x / torch.norm(x, dim=-1).unsqueeze(-1)
    return x


def embedding_label(labels: torch.Tensor):
    emb = torch.zeros(labels.shape[0], 2)
    emb[torch.arange(labels.shape[0]), labels] = 1
    return emb


def filiter_single_channel(batch):
    return batch[0, ...]


def filter_dataset(dataset, allowed_digits=[0, 1]):
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in allowed_digits:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)


img_size = 16
transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(filiter_single_channel),
        transforms.Lambda(embedding_pixel),
    ]
)

trainset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)

trainset = filter_dataset(trainset, allowed_digits=[0, 1])

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
)


def loss_batch(outputs, labels):
    device = outputs.device
    loss = torch.zeros(1, device=device, dtype=torch.float64)

    for i in range(len(outputs)):
        prob = outputs[i] if labels[i] == 0 else 1 - outputs[i]
        loss -= torch.log(prob + 1e-8)
    return loss


def calculate_accuracy(outputs, labels):
    predictions = (outputs < 0.5).float()
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy.item()


N = 16 * 16


umpsm = umps.uMPS(N=N, chi=2, d=2, l=2, layers=1, device="cpu")

umpsm_op = unitary_optimizer.Adam(umpsm, lr=0.01)

for epoch in range(100):
    acc = 0
    for data, target in trainloader:
        data = data.permute(1, 0, 2)
        umpsm_op.zero_grad()
        outputs = umpsm(data)
        loss = loss_batch(outputs, target)
        loss.backward()
        umpsm_op.step()

        # Calculate accuracy
        accuracy = calculate_accuracy(outputs, target)
        print(f"Accuracy: {accuracy:.4f}, loss: {loss.item():.4f}")

        acc += accuracy
    acc /= len(trainloader)
