from pathlib import Path
from typing import Callable, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import config

class MNISTDataset(Dataset):
    def __init__(self, csv_file_path: str | Path, transform_fn: Callable):
        if transform_fn is None or not callable(transform_fn):
            raise ValueError(
                "A valid callable transform/preprocessing pipeline must be provided."
            )

        self.csv_path = Path(csv_file_path)
        self.transform_fn = transform_fn

        if not self.csv_path.exists():
            raise FileNotFoundError(f"MNIST CSV file not found: {self.csv_path.resolve()}")

        df = pd.read_csv(self.csv_path, header=None)

        self.labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        self.images_np = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype("uint8")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor | Image.Image, torch.Tensor]:
        img = Image.fromarray(self.images_np[idx], mode="L")
        label = self.labels[idx]

        img = self.transform_fn(img)

        return img, label

def load_mnist_train(transform_fn: Callable) -> MNISTDataset:
    csv_path = Path(config.MNIST.DIR) / "train.csv"
    return MNISTDataset(csv_path, transform_fn=transform_fn)

def load_mnist_val(transform_fn: Callable) -> MNISTDataset:
    csv_path = Path(config.MNIST.DIR) / "val.csv"
    return MNISTDataset(csv_path, transform_fn=transform_fn)

def load_mnist_test(transform_fn: Callable) -> MNISTDataset:
    csv_path = Path(config.MNIST.DIR) / "test.csv"
    return MNISTDataset(csv_path, transform_fn=transform_fn)