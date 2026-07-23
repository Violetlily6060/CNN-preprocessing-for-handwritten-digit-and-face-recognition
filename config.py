from types import SimpleNamespace
import os
import torch
import torch.nn as nn
import torch.optim as optim

# use device GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5

# loss Function and optimizer
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER_CLASS = optim.Adam

# mnist parameter
MNIST = SimpleNamespace(
    INPUT_SHAPE=(1, 28, 28),
    NUM_CLASS=10,
    DIR=os.path.join("dataset", "mnist_28x28"),
)

# vggface2 parameter
VGGFACE2 = SimpleNamespace(
    INPUT_SHAPE=(1, 112, 112),
    NUM_CLASS=100,
    DIR=os.path.join("dataset", "vggface2_112x112"),
)