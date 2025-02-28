import torch
import numpy as np

###############################################################################
# MNIST Dataset Creation Utility
###############################################################################
def create_mnist_dataloader(
    allowed_digits=[0, 1],
    img_size=16,
    root="data",
    train=True,
    download=True,
    batch_size=128,
    num_data=None
):
    """
    Create and return a DataLoader for the MNIST dataset.
    
    The function applies the following steps:
      - Resizes images to `img_size`.
      - Converts images to tensor.
      - Takes only the first channel.
      - Embeds each pixel as [x, 1-x] and L2-normalizes.
      - Filters the dataset to keep only the digits in allowed_digits.
      - Optionally, subsets the dataset to num_data samples.
      - Returns a DataLoader with the specified batch_size.
    
    Parameters:
      allowed_digits: list of digit labels to keep (default [0, 1]).
      img_size: size to which images will be resized (default 16).
      root: directory for dataset storage (default "data").
      train: if True, creates training set; else, test set.
      download: if True, downloads dataset if needed.
      batch_size: batch size for the DataLoader.
      num_data: if provided, limit dataset to this many samples.
    
    Returns:
      A torch.utils.data.DataLoader for MNIST.
    """
    from torchvision import transforms, datasets

    # Define the transforms.
    def filiter_single_channel(img: torch.Tensor) -> torch.Tensor:
        # MNIST images have shape [C, H, W]. Return the first channel.
        return img[0, ...]
    
    def embedding_pixel(batch, label: int = 0):
        # Flatten image from [H, W] to [H*W] and embed each scalar x as [x, 1-x].
        pixel_size = batch.shape[-1] * batch.shape[-2]
        x = batch.view(*batch.shape[:-2], pixel_size)
        x = torch.stack([x, 1 - x], dim=-1)
        # L2-normalize along last dimension.
        norm = torch.sum(x, dim=-1, keepdim=True).clamp(min=1e-8)
        return x / norm


    class FilteredDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, allowed_digits=[0,1]):
            self.allowed_digits = allowed_digits
            self.indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_digits]
            self.dataset = dataset

            # Mapping original labels to [0,1,...]
            self.label_to_index = {digit: idx for idx, digit in enumerate(allowed_digits)}

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data, original_label = self.dataset[self.indices[idx]]
            new_label = self.label_to_index[original_label]
            return data, new_label


    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float64),
        transforms.Lambda(filiter_single_channel),
        transforms.Lambda(embedding_pixel)
    ])

    # Load the MNIST dataset.
    dataset = datasets.MNIST(root=root, train=train, download=download, transform=transform)
    dataset = FilteredDataset(dataset, allowed_digits=allowed_digits)

    if num_data is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_data))

    # Create and return the DataLoader.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

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