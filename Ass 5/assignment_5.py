from stanford40 import create_stanford40_splits
from HMDB51 import create_hmdb51_splits
import torch.nn as nn
import torch.optim as optim
from train import initialize_model, train_and_validate, plot_metrics, plot_confusion_matrix
from torch.optim.lr_scheduler import CyclicLR, StepLR
import settings


# Train and test files for model 1
train_files, train_labels, test_files, test_labels = create_stanford40_splits()

# Initialize model, loss function, and optimizer
model = initialize_model(model_class)
optimizer = optim.Adam(model.parameters(), lr=0.001) # TODO: learnig rate
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
train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs=15)

# Plot the training and validation losses and accuracies
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)



# TODO: Read the files and labels from the augmented dataset and append them to these






# Train and test files for model 2, 3 and 4
keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
            "run", "shoot_bow", "smoke", "throw", "wave"]
train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)