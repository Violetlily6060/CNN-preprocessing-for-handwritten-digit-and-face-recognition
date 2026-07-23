import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layers
        self.fc = nn.LazyLinear(out_features=128)                       # dense layer with 128 neurons
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)                # regularization using dropout rate
        self.out = nn.Linear(in_features=128, out_features=num_class)   # output for raw scores

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1) # flatten spatial feature maps into 1D vector

        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.out(x)

        return x