from stanford40 import create_stanford40_splits
from HMDB51 import create_hmdb51_splits
import torch.nn as nn
import torch.optim as optim
from train import train_and_validate, plot_metrics
from torch.optim.lr_scheduler import CyclicLR, StepLR
import settings
import matplotlib.pyplot as plt
from models import Stanford40_model, HMDB51_Frame_Model, HMDB51_OF_Model, HMDB51_Fusion_Model
from datasets import Stanford40_Dataset, HMDB51_Frame_Dataset, HMDB51_OF_Dataset, HMDB51_Fusion_Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
import os
from evaluate import test_model
import torch
from sklearn.model_selection import train_test_split

# STANFORD40 - FRAMES
if not os.path.exists(f'Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}.pth'):
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
    train_dataset = Stanford40_Dataset(img_dir='photo_dataset/train', file_paths=train_files, labels=train_labels, transform=ToTensor())

    # Split the training dataset into training and validation
    validation_split = 0.15
    num_train = len(train_dataset)
    num_validation = int(num_train * validation_split)
    num_train -= num_validation
    train_data, validation_data = random_split(train_dataset, [num_train, num_validation])

    # Setup the DataLoader for each split
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    print("Data loaders created successfully.")

    # FIRST MODEL
    # Initialize model, loss function, and optimizer
    model = Stanford40_model()
    model_name = "Stanford40_model"
    criteria = nn.CrossEntropyLoss()

    # Set the learning rate scheduler (or fixed) based on the settings
    if (settings.LR_SCHEDULER_TYPE["Stanford40"]) == 'dynamic':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE["Stanford40"]) == 'fixed':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    elif (settings.LR_SCHEDULER_TYPE["Stanford40"]) == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=15, mode='triangular')
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=20)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, settings.LR_SCHEDULER_TYPE["Stanford40"])

# Test the model
if not os.path.exists(f'plots/Stanford40_model/CM/Test_CM_Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}.png'):
    print(f'plots/Stanford40_model/CM/Test_CM_Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}.png')
    if os.path.exists(f'Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}.pth'):
        # Create Stanford40 test dataset
        train_files, train_labels, test_files, test_labels = create_stanford40_splits()
        test_dataset = Stanford40_Dataset(img_dir='photo_dataset/test', file_paths=test_files, labels=test_labels, transform=ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = Stanford40_model()
        model.load_state_dict(torch.load(f'Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}.pth'))
        test_accuracy, confusion_matrix = test_model(model, test_loader, settings.LR_SCHEDULER_TYPE["Stanford40"], device='cpu')
        # Save the test accuracy on a txt file
        with open('plots/tests/Test_accuracies.txt', 'w') as f:
            f.write(f'Test accuracy: {test_accuracy:.2f}% for Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}')
        print(f'Test accuracy: {test_accuracy:.2f}% for Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}')

###########################################################################################################################################################################################

# HMDB51 - FRAMES
if not os.path.exists(f'HMDB51_Frame_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]}.pth'):
    keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                "run", "shoot_bow", "smoke", "throw", "wave"]
    train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)

    # Choose frame number between 0, 25, 50, 75, 100
    frame_number = 50
    train_dataset = HMDB51_Frame_Dataset(train_files, train_labels, "video_image_dataset", frame_number, transform=ToTensor())

    # Split the training dataset into training and validation
    validation_split = 0.15
    num_train = len(train_dataset)
    num_validation = int(num_train * validation_split)
    num_train -= num_validation
    train_data, validation_data = random_split(train_dataset, [num_train, num_validation])

    # Setup the DataLoader for each split
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    print("Data loaders created successfully.")

    # SECOND MODEL
    # Initialize model, loss function, and optimizer
    model = HMDB51_Frame_Model()
    model_name = "HMDB51_Frame_Model"
    criteria = nn.CrossEntropyLoss()

    # Set the learning rate scheduler (or fixed) based on the settings
    if (settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]) == 'dynamic':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]) == 'fixed':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = None
    elif (settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]) == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=15, mode='triangular')
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=20)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, settings.LR_SCHEDULER_TYPE["HMDB51_Frames"])

# Test the model
if not os.path.exists(f'plots/HMDB51_Frame_Model/CM/Test_CM_HMDB51_Frame_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]}.png'):
    if os.path.exists(f'HMDB51_Frame_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]}.pth'):
        # HMDB51 - Frames
        keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                    "run", "shoot_bow", "smoke", "throw", "wave"]
        train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)
        frame_number = 50
        test_dataset = HMDB51_Frame_Dataset(test_files, test_labels, "video_image_dataset", frame_number, transform=ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = HMDB51_Frame_Model()
        model.load_state_dict(torch.load(f'HMDB51_Frame_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]}.pth'))
        test_accuracy, confusion_matrix = test_model(model, test_loader, settings.LR_SCHEDULER_TYPE["HMDB51_Frames"], device='cpu')
        # Save the test accuracy on a txt file
        with open('plots/tests/Test_accuracies.txt', 'w') as f:
            f.write(f'Test accuracy: {test_accuracy:.2f}% for HMDB51_Frame_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]}.pth')
        print(f'Test accuracy: {test_accuracy:.2f}% for HMDB51_Frame_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]}.pth')
 

###########################################################################################################################################################################################

# HMDB51 - OPTICAL FLOW
if not os.path.exists(f'HMDB51_OF_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_OF"]}.pth'):
    # HMDB51 - Optical Flow
    keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                "run", "shoot_bow", "smoke", "throw", "wave"]
    all_train_files, all_train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)
    
    train_files, val_files, train_labels, val_labels = train_test_split(all_train_files, all_train_labels, test_size=0.1, random_state=0, stratify=all_train_labels)

    # Create custom dataset
    train_dataset = HMDB51_OF_Dataset(train_files, train_labels, root_dir="video_OF_dataset")
    val_dataset = HMDB51_OF_Dataset(val_files, val_labels, root_dir="video_OF_dataset")

    # Setup the DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print("Data loaders created successfully.")

    # THIRD MODEL
    # Initialize model, loss function, and optimizer
    model = HMDB51_OF_Model()
    model_name = "HMDB51_OF_Model"
    criteria = nn.CrossEntropyLoss()

    # Set the learning rate scheduler (or fixed) based on the settings
    if (settings.LR_SCHEDULER_TYPE["HMDB51_OF"]) == 'dynamic':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE["HMDB51_OF"]) == 'fixed':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = None
    elif (settings.LR_SCHEDULER_TYPE["HMDB51_OF"]) == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=0.0000001, weight_decay=0.0001)
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=15, mode='triangular')
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=50)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, 'Fixed LR')

if not os.path.exists(f'plots/HMDB51_OF_Model/CM/Test_CM_HMDB51_OF_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_OF"]}.png'):
    if os.path.exists(f'HMDB51_OF_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_OF"]}.pth'):
        print("Testing the model...")
        # HMDB51 - Frames
        keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                    "run", "shoot_bow", "smoke", "throw", "wave"]
        train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)
        test_dataset = HMDB51_OF_Dataset(test_files, test_labels, root_dir="video_OF_dataset")
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = HMDB51_OF_Model()
        model.load_state_dict(torch.load(f'HMDB51_OF_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_OF"]}.pth'))
        test_accuracy, confusion_matrix = test_model(model, test_loader, settings.LR_SCHEDULER_TYPE["HMDB51_OF"], device='cpu')
        # Save the test accuracy on a txt file
        with open('plots/tests/Test_accuracies.txt', 'w') as f:
            f.write(f'Test accuracy: {test_accuracy:.2f}% for HMDB51_OF_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_OF"]}.pth')
        print(f'Test accuracy: {test_accuracy:.2f}% for HMDB51_OF_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_OF"]}.pth')

###########################################################################################################################################################################################

# HMDB51 - FUSION
if not os.path.exists(f'HMDB51_Fusion_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]}.pth'):
    # HMDB51 - FUSION
    keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                "run", "shoot_bow", "smoke", "throw", "wave"]
    all_train_files, all_train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)
    
    train_files, val_files, train_labels, val_labels = train_test_split(all_train_files, all_train_labels, test_size=0.1, random_state=0, stratify=all_train_labels)

    # Create custom dataset
    train_dataset = HMDB51_Fusion_Dataset(train_files, train_labels, frames_root_dir="video_image_dataset", frame_number=50, optical_flow_root_dir="video_OF_dataset", transform=ToTensor())
    val_dataset = HMDB51_Fusion_Dataset(val_files, val_labels, frames_root_dir="video_image_dataset", frame_number=50, optical_flow_root_dir="video_OF_dataset", transform=ToTensor())

    # Setup the DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print("Data loaders created successfully.")

    # FOURTH MODEL
    # Initialize model, loss function, and optimizer
    model = HMDB51_Fusion_Model()
    model_name = "HMDB51_Fusion_Model"
    criteria = nn.CrossEntropyLoss()

    # Set the learning rate scheduler (or fixed) based on the settings
    if (settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]) == 'dynamic':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        # Use StepLR for dynamic learning rate adjustments
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    elif (settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]) == 'fixed':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = None
    elif (settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]) == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)
        # Use CyclicLR for cyclic learning rate adjustments
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=15)
    else:
        raise ValueError("Invalid learning rate schedule type specified.")

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=10)

    # Plot the training and validation losses and accuracies
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, 'Fixed LR')

if not os.path.exists(f'plots/HMDB51_Fusion_Model/CM/Test_CM_HMDB51_Fusion_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]}.png'):
    if os.path.exists(f'HMDB51_Fusion_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]}.pth'):
        print("Testing the model...")
        # HMDB51 - Optical Flow
        keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                    "run", "shoot_bow", "smoke", "throw", "wave"]
        all_train_files, all_train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)
        
        train_files, val_files, train_labels, val_labels = train_test_split(all_train_files, all_train_labels, test_size=0.1, random_state=0, stratify=all_train_labels)

        # Create custom dataset
        test_dataset = HMDB51_Fusion_Dataset(test_files, test_labels, frames_root_dir="video_image_dataset", frame_number=50, optical_flow_root_dir="video_OF_dataset", transform=ToTensor())

        # Setup the DataLoader for each split
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        print("Data loaders created successfully.")

        # Initialize model, loss function, and optimizer
        model = HMDB51_Fusion_Model()
        model.load_state_dict(torch.load(f'HMDB51_Fusion_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]}.pth'))
        test_accuracy, confusion_matrix = test_model(model, test_loader, settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"], device='cpu')
        # Save the test accuracy on a txt file
        with open('plots/tests/Test_accuracies.txt', 'w') as f:
            f.write(f'Test accuracy: {test_accuracy:.2f}% for HMDB51_Fusion_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]}.pth')
        print(f'Test accuracy: {test_accuracy:.2f}% for HMDB51_Fusion_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Fusion"]}.pth')