"""PyTeen Network Structure"""

import torch
from torch import nn, optim


class PyTeen(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential layers (fully connected/linear layers)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, 5, padding=2), #(num channels, out channels, size of kernels)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(1920, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        # Loss/Cost func
        self.loss = nn.CrossEntropyLoss()
        # Optimizer - for adjusting weights
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, inputs):
        # Forward pass of network
        return self.layers(inputs)

    def predict(self, feature):
        # Don't update gradients
        with torch.no_grad():
            # Make and return prediction
            pred = self.forward(feature)
            return torch.argmax(pred, axis=-1)

    def train_net(self, feature, label):
        # Zero out all gradients at each step
        self.optimizer.zero_grad()
        # Make prediction using the NN
        pred = self.forward(feature) # Feature = input
        # Calcuate the loss using our defined loss func
        loss = self.loss(pred, label) # Label = actual
        # Put gradients in the .grad props
        loss.backward()
        # Update gradients
        self.optimizer.step()
        return loss.item() # Don't NEED to return loss - nice for training loop
    