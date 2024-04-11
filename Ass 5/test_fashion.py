from stanford40 import create_stanford40_splits
from HMDB51 import create_hmdb51_splits
import torch.nn as nn
import torch.optim as optim
from train import initialize_model, train_and_validate, plot_metrics, plot_learning_rate
from torch.optim.lr_scheduler import CyclicLR, StepLR
import settings
import matplotlib.pyplot as plt
from models import Stanford40_model, HMDB51_model, HMDB51_OF_model, Fashion_model
from datasets import CustomStandford40Dataset, VideoFrameDataset, OpticalFlowDataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
import os

from torchvision import datasets

# Import train and test data from the Fashion-MNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Split the training data into training and validation sets
val_size = int(0.2 * len(training_data))
train_size = len(training_data) - val_size
train_data, val_data = random_split(training_data, [train_size, val_size])

# Define loaders for training, validation, and test sets
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
validation_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

scheduler_i = 'dynamic'

# THIRD MODEL
# Initialize model, loss function, and optimizer
model = Fashion_model()
model_name = "MNIST_model"
criteria = nn.CrossEntropyLoss()

# set learning rate scheduler that decreases the learning rate by a factor of 0.5 every 5 epochs or cyclitic learning rate
if (scheduler_i) == 'dynamic':
    optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.0001)
    # Use StepLR for dynamic learning rate adjustments
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
elif (scheduler_i) == 'cyclic':
    optimizer = optim.SGD(model.parameters(), lr=0.0000001, weight_decay=0.0001)
    # Use CyclicLR for cyclic learning rate adjustments
    scheduler = CyclicLR(optimizer, base_lr=0.0000001, max_lr=0.000001, step_size_up=15)
else:
    raise ValueError("Invalid learning rate schedule type specified.")

# Train the model
train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=10)

# Plot the training and validation losses and accuracies
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, scheduler_i)