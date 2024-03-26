import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
import os
import random

from models import LeNet5, LeNet5Variant1, LeNet5Variant2, LeNet5Variant3, LeNet5Variant4
import settings
from train import train_model, k_fold_train_and_validate
from evaluate import plot_losses_accuracies, test_model, plot_feature_maps

# Define the device to be used
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

# BASELINE MODEL - Train and plot the losses and accuracies
if not os.path.exists("models/LeNet5_fold_0.pth"):

    print("Training baseline model")

    # Initialize model, loss function, and optimizer
    baseline = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(baseline.parameters(), lr=0.001)

    # Train the model
    train_losses, val_losses, train_accs, val_accs, _ = train_model(baseline, "LeNet5", criterion, optimizer, train_loader, val_loader, dynamic_lr=False)

    print("Baseline model trained successfully")

    # Plot the training and validation losses and accuracies
    plot_losses_accuracies(train_losses, val_losses, train_accs, val_accs, "LeNet5")

    print("Baseline model losses and accuracies plotted successfully")

# Define all the model variants
variants = [LeNet5Variant1(), LeNet5Variant2(), LeNet5Variant3(), LeNet5Variant4()]
variants_names = ["LeNet5Variant1", "LeNet5Variant2", "LeNet5Variant3", "LeNet5Variant4"]

# Train and plot the losses and accuracies for each model variant
for i in range(len(variants)):
    if not os.path.exists(f"models/{variants_names[i]}_fold_0.pth"):

        print(f"Training {variants_names[i]} model")

        # Initialize model, loss function, and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(variants[i].parameters(), lr=0.001)

        # Train the model
        train_losses, val_losses, train_accs, val_accs, _ = train_model(variants[i], variants_names[i], criterion, optimizer, train_loader, val_loader, dynamic_lr=(i >= 2))

        print(f"{variants_names[i]} model trained successfully")

        # Plot the training and validation losses and accuracies
        plot_losses_accuracies(train_losses, val_losses, train_accs, val_accs, variants_names[i])

        print(f"{variants_names[i]} model losses and accuracies plotted successfully")

# Perform the choice tasks on the best model
j = len(variants) - 1
print(f"{variants_names[j]} model - Choice tasks")

# Test the accuracy of the model on the test set and plot the t-SNE visualization
test_accuracy = test_model(variants_names[j], test_loader, IS_TSNE = True)

print(f"Test accuracy for {variants_names[j]} model: {test_accuracy}")

# CROSS VALIDATION

if not os.path.exists(f"models/{variants_names[j]}_fold_5.pth"):

    print("Cross validation...")

    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(variants[j].parameters(), lr=0.001)

    # Train the model
    train_losses, val_losses, train_accs, val_accs = k_fold_train_and_validate(variants[j], variants_names[j], criterion, training_data, num_folds=5, dynamic_lr=False)

    print(f"{variants_names[j]} model trained successfully")

    # Plot the training and validation losses and accuracies
    plot_losses_accuracies(train_losses, val_losses, train_accs, val_accs, variants_names[j])

    print(f"{variants_names[j]} model losses and accuracies plotted successfully")

# INTERMEDIATE OUTPUT LAYERS

print("Testing final model with intermediate outputs")

# Choose two items from the test set to be anaylised
# Choose two classes to retrieve
classes_to_retrieve = [0, 9]
# Choose the index of the item to retrieve for each class
index_of_class = [3, 9]
for i in range(len(classes_to_retrieve)):
    # Define the class to retrieve
    class_to_retrieve = classes_to_retrieve[i] 

    # Select all the indexes of the images with the class to retrieve
    index_image = [i for i, label in enumerate(test_data.targets) if label == class_to_retrieve]

    # Get the image to be analysed
    image = test_data[index_image[index_of_class[i]]][0].unsqueeze(0)

    # Load the model
    model = LeNet5Variant4()
    model.load_state_dict(torch.load(f'models/LeNet5Variant4_fold_0.pth'))

    # Set the model to evaluation mode
    model.eval()
    _, _, intermediates = model(image)

    # Define the class names
    class_names = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Plot the feature maps of the intermediate layers
    plot_feature_maps(intermediates, class_names[class_to_retrieve])

# TRAINING ON TRAIN AND VALIDATION SET

if not os.path.exists(f"models/Train + Validation Set {variants_names[j]}_fold_0.pth"):
    print("Training on train and validation set...")

    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(variants[j].parameters(), lr=0.001)

    # Combine the training and validation sets
    train_val_loader = DataLoader(training_data, batch_size=settings.BATCH_SIZE, shuffle=True)

    # Train the model
    train_losses, val_losses, train_accs, val_accs, _ = train_model(variants[j], f"Train + Validation Set {variants_names[j]}", criterion, optimizer, train_val_loader, _, dynamic_lr=True, use_validation=False)

    print(f"{variants_names[j]} model trained successfully")

    # Plot the training and validation losses and accuracies
    plot_losses_accuracies(train_losses, val_losses, train_accs, val_accs, f"Train + Validation Set {variants_names[j]}", use_validation=False)

    print(f"{variants_names[j]} model losses and accuracies plotted successfully")

    test_accuracy = test_model(f"Train + Validation Set {variants_names[j]}", test_loader)

    print(f"Test accuracy for {variants_names[j]} model trained on train+val: {test_accuracy}")