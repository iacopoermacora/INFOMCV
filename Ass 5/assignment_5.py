from stanford40 import create_stanford40_splits
from HMDB51 import create_hmdb51_splits
import torch.nn as nn
import torch.optim as optim
from train import initialize_model, train_and_validate, plot_metrics, plot_learning_rate
from torch.optim.lr_scheduler import CyclicLR, StepLR
import settings
import matplotlib.pyplot as plt
from models import Stanford40_model, HMDB51_model, HMDB51_OF_model, HMDB51_OF_VGG_model, ActionRecognitionModel
from datasets import CustomStandford40Dataset, VideoFrameDataset, OpticalFlowDataset, OpticalFlowDataset_PY
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
import os
from evaluate import test_model
import torch

'''# STANFORD40 - FRAMES
if not os.path.exists(f'Stanford40_model_{settings.LR_SCHEDULER_TYPE}.pth'):
    # Create Stanford40 dataset
    train_files, train_labels, test_files, test_labels = create_stanford40_splits()

    # Read the files from the augmented_files.txt file
    with open('augmented_files.txt', 'r') as f:
        augmented_files = [line.strip() for line in f]
    train_files = train_files + augmented_files

    # Read the labels from the augmented_labels.txt file
    with open('augmented_labels.txt', 'r') as f:
        augmented_labels = [line.strip() for line in f]
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
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Data loaders created successfully.")

    # FIRST MODEL
    # Initialize model, loss function, and optimizer
    model = Stanford40_model()
    model_name = "Stanford40_model"
    criteria = nn.CrossEntropyLoss()

    # set learning rate scheduler that decreases the learning rate by a factor of 0.5 every 5 epochs or cyclitic learning rate
    if (settings.LR_SCHEDULER_TYPE) == 'dynamic':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE) == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=15, mode='triangular2')
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=20)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, settings.LR_SCHEDULER_TYPE)

# Test the model
#if confusion_matrix_Stanford40_model.png does not exist
if not os.path.exists(f'plots/confusion_matrix_Stanford40_model_{settings.LR_SCHEDULER_TYPE}.png'):
    if os.path.exists(f'Stanford40_model_{settings.LR_SCHEDULER_TYPE}.pth'):
        # Create Stanford40 test dataset
        train_files, train_labels, test_files, test_labels = create_stanford40_splits()
        test_dataset = CustomStandford40Dataset(img_dir='photo_dataset/test', file_paths=test_files, labels=test_labels, transform=ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = Stanford40_model()
        model.load_state_dict(torch.load(f'Stanford40_model_dynamic.pth'))
        test_accuracy, confusion_matrix = test_model(model, test_loader, device='cpu')
        print(f'Test accuracy: {test_accuracy:.2f}% for Stanford40_model_dynamic.pth')

# HMDB51 - FRAMES
if not os.path.exists(f'HMDB51_model_{settings.LR_SCHEDULER_TYPE}.pth'):
    keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                "run", "shoot_bow", "smoke", "throw", "wave"]
    train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)

    # Choose frame number between 0, 25, 50, 75, 100
    frame_number = 50
    train_dataset = VideoFrameDataset(train_files, train_labels, "video_image_dataset", frame_number, transform=ToTensor())
    test_dataset = VideoFrameDataset(test_files, test_labels, "video_image_dataset", frame_number, transform=ToTensor())

    # Split the training dataset into training and validation
    validation_split = 0.15
    num_train = len(train_dataset)
    num_validation = int(num_train * validation_split)
    num_train -= num_validation
    train_data, validation_data = random_split(train_dataset, [num_train, num_validation])

    # Setup the DataLoader for each split
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Data loaders created successfully.")

    # SECOND MODEL
    # Initialize model, loss function, and optimizer
    model = HMDB51_model()
    model_name = "HMDB51_model"
    criteria = nn.CrossEntropyLoss()

    # set learning rate scheduler that decreases the learning rate by a factor of 0.5 every 5 epochs or cyclitic learning rate
    if (settings.LR_SCHEDULER_TYPE) == 'dynamic':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE) == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, step_size_up=15, mode='triangular')
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=20)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, settings.LR_SCHEDULER_TYPE)

# Test the model
if not os.path.exists(f'plots/confusion_matrix_HMDB51_model_{settings.LR_SCHEDULER_TYPE}.png'):
    if os.path.exists(f'HMDB51_model_{settings.LR_SCHEDULER_TYPE}.pth'):
        # HMDB51 - Frames
        keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                    "run", "shoot_bow", "smoke", "throw", "wave"]
        train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)
        frame_number = 50
        test_dataset = VideoFrameDataset(test_files, test_labels, "video_image_dataset", frame_number, transform=ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = HMDB51_model()
        model.load_state_dict(torch.load(f'HMDB51_model_cyclic.pth'))
        test_accuracy, confusion_matrix = test_model(model, test_loader, device='cpu')
        print(f'Test accuracy: {test_accuracy:.2f}% for HMDB51_model_cyclic.pth')'''
 


# HMDB51 - OPTICAL FLOW
if not os.path.exists(f'HMDB51_OF_model_{settings.LR_SCHEDULER_TYPE}.pth'):
    # HMDB51 - Optical Flow
    keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                "run", "shoot_bow", "smoke", "throw", "wave"]
    train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)

    # Create custom dataset
    train_dataset = OpticalFlowDataset_PY(train_files, train_labels, root_dir="video_OF_py_dataset", transform=ToTensor())
    test_dataset = OpticalFlowDataset_PY(test_files, test_labels, root_dir="video_OF_py_dataset", transform=ToTensor())

    # Split the training dataset into training and validation
    validation_split = 0.15
    num_train = len(train_dataset)
    num_validation = int(num_train * validation_split)
    num_train -= num_validation
    train_data, validation_data = random_split(train_dataset, [num_train, num_validation])

    # Setup the DataLoader for each split
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Data loaders created successfully.")

    # THIRD MODEL
    # Initialize model, loss function, and optimizer
    model = ActionRecognitionModel()
    model_name = "ActionRecognitionModel"
    criteria = nn.CrossEntropyLoss()

    # set learning rate scheduler that decreases the learning rate by a factor of 0.5 every 5 epochs or cyclitic learning rate
    if (settings.LR_SCHEDULER_TYPE) == 'dynamic':
        optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.0001)
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE) == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=0.0000001, weight_decay=0.0001)
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.0000001, max_lr=0.000001, step_size_up=15)
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=10)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, settings.LR_SCHEDULER_TYPE)