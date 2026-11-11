from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from app.features import FEATURE_SIZE, extract_handcrafted

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
IMAGE_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def image_transform(training: bool = False):
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20)])
    ops.extend([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    return transforms.Compose(ops)


class HistologyDataset(Dataset):
    """ImageFolder dataset that returns normalized image, handcrafted vector, and label."""

    feature_dim = FEATURE_SIZE

    def __init__(self, root: str | Path, training: bool = False):
        self.base = datasets.ImageFolder(str(root))
        if self.base.classes != CLASS_NAMES:
            raise ValueError(f"Expected class directories in order {CLASS_NAMES}; found {self.base.classes}")
        self.samples = self.base.samples
        self.transform = image_transform(training)
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        # Augmentation is applied to the CNN view; descriptors are extracted from the same resized source.
        tensor = self.transform(image)
        feature = torch.from_numpy(extract_handcrafted(np.asarray(image)))
        return tensor, feature, target


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
