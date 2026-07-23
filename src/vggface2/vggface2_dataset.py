from pathlib import Path
from typing import Callable, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import config

class VGGFace2Dataset(Dataset):
    def __init__(self, split_dir_path: str | Path, transform_fn: Callable):
        if transform_fn is None or not callable(transform_fn):
            raise ValueError(
                "A valid callable transform/preprocessing pipeline must be provided."
            )

        self.split_dir = Path(split_dir_path)
        self.transform_fn = transform_fn

        if not self.split_dir.exists():
            raise FileNotFoundError(f"VGGFace2 split directory not found: {self.split_dir.resolve()}")

        self.image_paths = []
        self.labels = []

        subject_dirs = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])
        
        self.class_to_idx = {subj.name: idx for idx, subj in enumerate(subject_dirs)}

        for subj_dir in subject_dirs:
            class_label = self.class_to_idx[subj_dir.name]
            img_files = sorted(subj_dir.glob("*.jpg"))

            for img_path in img_files:
                self.image_paths.append(img_path)
                self.labels.append(class_label)

        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg images found in {self.split_dir.resolve()}")

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor | Image.Image, torch.Tensor]:
        img_path = self.image_paths[idx]

        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")

        label = self.labels[idx]

        transformed_img = self.transform_fn(pil_img)

        return transformed_img, label

def load_vggface2_train(transform_fn: Callable) -> VGGFace2Dataset:
    split_dir = Path(config.VGGFACE2.DIR) / "train"
    return VGGFace2Dataset(split_dir, transform_fn=transform_fn)

def load_vggface2_val(transform_fn: Callable) -> VGGFace2Dataset:
    split_dir = Path(config.VGGFACE2.DIR) / "val"
    return VGGFace2Dataset(split_dir, transform_fn=transform_fn)

def load_vggface2_test(transform_fn: Callable) -> VGGFace2Dataset:
    split_dir = Path(config.VGGFACE2.DIR) / "test"
    return VGGFace2Dataset(split_dir, transform_fn=transform_fn)