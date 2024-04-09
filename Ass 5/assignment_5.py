from stanford40 import create_stanford40_splits
from HMDB51 import create_hmdb51_splits
import torch.nn as nn
import torch.optim as optim
from train import initialize_model, train_and_validate, plot_metrics, plot_confusion_matrix
from torch.optim.lr_scheduler import CyclicLR, StepLR
import settings
import matplotlib.pyplot as plt
from models import Stanford40_model, HMDB51_model
from datasets import CustomStandford40Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor


# Create Stanford40 dataset
train_files, train_labels, test_files, test_labels = create_stanford40_splits()

# Read the files from the augmented_files.txt file
with open('augmented_files.txt', 'r') as f:
    augmented_files = f.readlines()
    augmented_files = [x.strip() for x in augmented_files]
train_files = train_files + augmented_files

# Read the labels from the augmented_labels.txt file
with open('augmented_labels.txt', 'r') as f:
    augmented_labels = f.readlines()
    augmented_labels = [int(x.strip()) for x in augmented_labels]
train_labels = train_labels + augmented_labels

# Create custom dataset
# Initialize the datasets
train_dataset = CustomStandford40Dataset(img_dir='photo_dataset/train', file_paths=train_files, labels=train_labels, transform=ToTensor())
test_dataset = CustomStandford40Dataset(img_dir='photo_dataset/test', file_paths=test_files, labels=test_labels, transform=ToTensor())

# Split the training dataset into training and validation
validation_split = 0.15
num_train = len(train_dataset)
num_validation = int(num_train * validation_split)
num_train -= num_validation
train_data, validation_data = random_split(train_dataset, [num_train, num_validation])

# Setup the DataLoader for each split
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print("Data loaders created successfully.")


models = [Stanford40_model(), HMDB51_model()]

for i in range(len(models)):
    # Initialize model, loss function, and optimizer
    model = initialize_model(models[i])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criteria = nn.CrossEntropyLoss()

    # set learning rate scheduler that decreases the learning rate by a factor of 0.5 every 5 epochs or cyclitic learning rate
    if (settings.LR_SCHEDULER_TYPE) == 'dynamic':
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE) == 'cyclic':
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=25)
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=5)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model.__class__.__name__)

'''# Train and test files for model 2, 3 and 4
keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
            "run", "shoot_bow", "smoke", "throw", "wave"]
train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)'''