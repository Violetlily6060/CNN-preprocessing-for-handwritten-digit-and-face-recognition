import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def get_p1_mnist_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(config.MNIST.INPUT_SHAPE[1:], antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_p1_vggface2_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize(config.VGGFACE2.INPUT_SHAPE[1:], antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_p2_transform():
    return v2.Compose([
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.8, 0.8)),
    ])


class SobelEdgeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)

        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        is_3d = img_tensor.dim() == 3
        if is_3d:
            img_tensor = img_tensor.unsqueeze(0)

        if img_tensor.shape[1] > 1:
            img_tensor = img_tensor.mean(dim=1, keepdim=True)

        gx = F.conv2d(img_tensor, self.kernel_x, padding=1)
        gy = F.conv2d(img_tensor, self.kernel_y, padding=1)

        magnitude = torch.sqrt(gx**2 + gy**2)
        magnitude = torch.clamp(magnitude, 0.0, 1.0)

        return magnitude.squeeze(0) if is_3d else magnitude


sobel_transform = SobelEdgeDetector()


def get_p3_transform():
    return v2.Compose([
        sobel_transform,
    ])


def get_p4_transform():
    return v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomEqualize(p=1.0),
        v2.ToDtype(torch.float32, scale=True),
    ])


p1_mnist = get_p1_mnist_transform()
p1_vggface2 = get_p1_vggface2_transform()
p2 = get_p2_transform()
p3 = get_p3_transform()
p4 = get_p4_transform()