import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
import os

from models import LeNet5, LeNet5Variant1, LeNet5Variant2
import settings
from train import train_model
from evaluate import plot_losses_accuracies


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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
val_size = int(settings.VALIDATION_SIZE * len(training_data))
train_size = len(training_data) - val_size
train_data, val_data = random_split(training_data, [train_size, val_size])

# Define loaders for training, validation, and test sets
train_loader = DataLoader(train_data, batch_size=settings.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=settings.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=settings.BATCH_SIZE, shuffle=False)

print("Data loaded successfully")

# BASELINE MODEL
if not os.path.exists("plots/LeNet5_losses_accuracies.png"):

    print("Training baseline model")

    # Initialize model, loss function, and optimizer
    baseline = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(baseline.parameters(), lr=0.001)

    # Train the model
    train_losses, val_losses, train_accs, val_accs = train_model(baseline, "LeNet5", criterion, optimizer, train_loader, val_loader, dynamic_lr=False)

    print("Baseline model trained successfully")

    # Plot the training and validation losses and accuracies
    plot_losses_accuracies(train_losses, val_losses, train_accs, val_accs, "LeNet5")

    print("Baseline model losses and accuracies plotted successfully")

variants = [LeNet5Variant1(), LeNet5Variant2()]
variants_names = ["LeNet5Variant1", "LeNet5Variant2"]

for i in range(len(variants)):
    if not os.path.exists(f"plots/{variants_names[i]}_losses_accuracies.png"):

        print(f"Training {variants_names[i]} model")

        # Initialize model, loss function, and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(variants[i].parameters(), lr=0.001)

        # Train the model
        train_losses, val_losses, train_accs, val_accs = train_model(variants[i], variants_names[i], criterion, optimizer, train_loader, val_loader, dynamic_lr=True)

        print(f"{variants_names[i]} model trained successfully")

        # Plot the training and validation losses and accuracies
        plot_losses_accuracies(train_losses, val_losses, train_accs, val_accs, variants_names[i])

        print(f"{variants_names[i]} model losses and accuracies plotted successfully")