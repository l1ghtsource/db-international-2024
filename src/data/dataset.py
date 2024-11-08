import os
import random
from PIL import Image
from typing import Tuple, Optional, Callable, Dict, List
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class RKNDataset(Dataset):
    '''
    A custom PyTorch Dataset class for generating triplets (anchor, positive, negative)
    based on image folders. Images within the same subfolder are considered similar 
    (class 1), while images in different subfolders are considered dissimilar (class 0).

    Args:
        root_dir (str): Path to the main directory containing subdirectories of images.
            Each subdirectory represents a class, and images within the same subdirectory
            are treated as similar.
        transform (Optional[Callable]): Optional transform to be applied on a sample 
            (anchor, positive, negative).

    Attributes:
        image_paths (List[Tuple[str, int]]): List containing tuples of image paths and their labels.
        labels (Dict[int, List[str]]): Dictionary where keys are labels, and values are lists of image paths
            corresponding to each label.

    Methods:
        __len__() -> int: Returns the length of the dataset.
        __getitem__(idx: int) -> Tuple[Tensor, Tensor, Tensor]: Generates a triplet (anchor, positive, negative) 
            for a given index.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing the transformed anchor, positive, 
            and negative images.

    Example:
        dataset = RKNDataset(root_dir='path/to/images', transform=transform)
        anchor, positive, negative = dataset[0]
    '''

    def __init__(
        self, root_dir: str,
        transform: Optional[Callable] = transforms.ToTensor()
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[Tuple[str, int]] = []
        self.labels: Dict[int, List[str]] = {}

        # collect all data
        for label, folder in enumerate(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.labels[label] = []
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    self.image_paths.append((image_path, label))
                    self.labels[label].append(image_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        anchor_path, anchor_label = self.image_paths[idx]

        # positive sample from the same directory
        positive_path = random.choice(self.labels[anchor_label])
        while positive_path == anchor_path:
            positive_path = random.choice(self.labels[anchor_label])

        # negative sample from a different directory
        negative_label = random.choice(list(self.labels.keys()))
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.labels.keys()))
        negative_path = random.choice(self.labels[negative_label])

        # load images
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        # apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
