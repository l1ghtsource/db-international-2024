from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from typing import Tuple, Optional, Callable
import torch
from dataset import RKNDataset


def get_rkn_dataloader(
    root_dir: str,
    transform: Optional[Callable] = None,
    shuffle: bool = True,
    batch_size: int = 32,
    val_split: float = 0.1,
    seed: int = 52
) -> Tuple[DataLoader, DataLoader]:
    '''
    Creates DataLoaders for training and validation splits of the RKNDataset.

    Args:
        root_dir (str): Path to the main directory containing subdirectories of images.
        transform (Optional[Callable]): Transformations to be applied to each image.
        shuffle (bool): Whether to shuffle the data at the DataLoader level.
        batch_size (int): Batch size for the DataLoaders.
        val_split (float): Proportion of the dataset to use for validation.
        seed (int): Random seed for splitting the dataset.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and validation sets.

    Example:
        root_dir = '/path/to/images/'
        transform = transforms.ToTensor()
        train_loader, val_loader = get_rkn_dataloader(root_dir, transform, shuffle=True, batch_size=32)
    '''
    dataset = RKNDataset(root_dir=root_dir, transform=transform)

    # determine lengths for train and validation splits
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # set random seed for reproducibility
    torch.manual_seed(seed)

    # split dataset into train and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create dataloaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
