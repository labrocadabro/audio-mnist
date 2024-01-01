"""Training PyTeen Network"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pyteen


BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform PIL image to tensor of image dimensions
cust_transform = transforms.ToTensor()

# Load training/testing data
training_data = datasets.ImageFolder(
    root="./PyTeen_CNN/data/train", # Define where to store/look for data
    transform=cust_transform) # Use custom transform

# Run this puppy!
pyteen = pyteen.PyTeen()
pyteen.to(DEVICE)

# Data loaders for train/test data
train_loader = DataLoader(
    training_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

# Training loop
print("-----------------Training Network-----------------")

for i in range(EPOCHS):
    total_loss = 0
    # Call train repeatedly with traiing data pairs (feature, label)
    for feature,label in tqdm(train_loader):
        feature = feature.to(DEVICE)
        label = label.to(DEVICE)
        total_loss += pyteen.train_net(feature, label)
    print(f"Total loss for Epoch {i+1}: {total_loss/len(train_loader)}")

# Save trained network weights
torch.save(pyteen.state_dict(), "./PyTeen_CNN/py_teen.pth")

print("\033[1;32mTrained state saved.")
