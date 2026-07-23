import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import config

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

        gx = F.conv2d(img_tensor, self.kernel_x, padding=1)
        gy = F.conv2d(img_tensor, self.kernel_y, padding=1)

        magnitude = torch.sqrt(gx**2 + gy**2)

        max_val = magnitude.max()
        if max_val > 0:
            magnitude = magnitude / max_val

        return magnitude.squeeze(0) if is_3d else magnitude

sobel_transform = SobelEdgeDetector()

def get_p3_mnist_transform():
    return v2.Compose([
        v2.Resize(config.MNIST.INPUT_SHAPE[1:], antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        sobel_transform,
    ])

def get_p3_vggface2_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize(config.VGGFACE2.INPUT_SHAPE[1:], antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        sobel_transform,
    ])

p3_mnist = get_p3_mnist_transform()
p3_vggface2 = get_p3_vggface2_transform()