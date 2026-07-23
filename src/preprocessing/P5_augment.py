import torch
from torchvision.transforms import v2
import config

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

gaussian_noise = AddGaussianNoise(mean=0.0, std=0.05)

def get_p5_mnist_transform():
    return v2.Compose([
        v2.Resize(config.MNIST.INPUT_SHAPE[1:], antialias=True),
        v2.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        gaussian_noise,
    ])

def get_p5_vggface2_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize(config.VGGFACE2.INPUT_SHAPE[1:], antialias=True),
        v2.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        gaussian_noise,
    ])

p5_mnist = get_p5_mnist_transform()
p5_vggface2 = get_p5_vggface2_transform()