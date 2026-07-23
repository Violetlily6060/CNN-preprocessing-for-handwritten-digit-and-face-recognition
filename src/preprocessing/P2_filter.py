import torch
from torchvision.transforms import v2
import config

def get_p2_mnist_transform():
    return v2.Compose([
        v2.Resize(config.MNIST.INPUT_SHAPE[1:], antialias=True),
        v2.GaussianBlur(kernel_size=(3, 3)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

def get_p2_vggface2_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize(config.VGGFACE2.INPUT_SHAPE[1:], antialias=True),
        v2.GaussianBlur(kernel_size=(3, 3)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

p2_mnist = get_p2_mnist_transform()
p2_vggface2 = get_p2_vggface2_transform()