import torch
from torch import nn, optim
from torchvision import transforms


class PyTeen(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(2480, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, input):
        return self.layers(input)

    def predict(self, input):
        with torch.no_grad():
            pred = self.forward(input)
            return torch.argmax(pred, axis=-1)


def predict(img):
    network = PyTeen()

    network.load_state_dict(torch.load("py_teen_augmented.pth"))

    img_tf = transforms.ToTensor()
    input_img = img_tf(img).unsqueeze(0)
    return (network.predict(input_img)).item()
