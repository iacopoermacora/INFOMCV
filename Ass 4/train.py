import settings
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader, Subset
from  tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model_name, y_true, y_pred, classes):
    '''
    Plot the confusion matrix

    param: model_name: str: Name of the model
    param: y_true: list: True labels
    param: y_pred: list: Predicted labels
    param: classes: list: List of class names

    return: None
    '''
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrices/' + model_name + '_confusion_matrix.png')

def plot_lr_evolution(learning_rates):
    '''
    Plot the evolution of learning rate with respect to the epoch

    param: learning_rates: list: List of learning rates

    return: None
    '''
    # Plot the evolution of learning rate with respect to the epoch
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, settings.NUM_EPOCHS + 1), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Evolution')
    plt.grid(True)
    plt.savefig('plots/learning_rate_evolution.png')

def initialize_model(model_class):
    '''
    Initialize the model

    param: model_class: class: Model class

    return: model: Model instance
    '''
    model = model_class()
    return model

# Train and validate the model using k-fold cross-validation
def k_fold_train_and_validate(model_constr, model_name, criterion, training_data, num_folds=5, dynamic_lr=False):
    '''
    Implement k-fold cross-validation

    param: model_constr: class: Model class
    param: model_name: str: Name of the model
    param: criterion: loss: Loss function
    param: training_data: Dataset: Training dataset
    param: num_folds: int: Number of folds
    param: dynamic_lr: bool: Whether to use dynamic learning rate

    return: results: dict: Dictionary of training and validation losses and accuracies
    '''
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # Initialize results dictionary
    results = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': []
    }
    learning_rates_per_fold = []

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(training_data)))):
        print(f'Fold {fold+1}/{num_folds}')

        # Creating train and validation subsets using PyTorch Subset
        train_subsampler = Subset(training_data, train_idx)
        val_subsampler = Subset(training_data, val_idx)

        print(f'Training on {len(train_subsampler)} samples, validating on {len(val_subsampler)} samples')

        # Creating data loaders for each fold
        train_loader = DataLoader(train_subsampler, batch_size=settings.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=settings.BATCH_SIZE, shuffle=False)

        # Initialize the model and optimizer for each fold
        model = initialize_model(type(model_constr))
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        # Call your existing training function
        train_losses, val_losses, train_accs, val_accs, learning_rates = train_model(model, model_name, criterion, optimizer, train_loader, val_loader, fold_num=fold+1, dynamic_lr=dynamic_lr)

        # Print after collecting results from each fold
        print(f'Completed fold {fold+1} with validation accuracy: {val_accs[-1]}')

        # Collect and average results from each fold
        results['train_losses'].append(train_losses)
        results['val_losses'].append(val_losses)
        results['train_accs'].append(train_accs)
        results['val_accs'].append(val_accs)
        learning_rates_per_fold.append(learning_rates)  # Collect learning rates for plotting

    # After all folds are completed, you can calculate and print the average loss and accuracy across all folds
    print(f"Average training loss: {np.mean(results['train_losses'])}")
    print(f"Average validation loss: {np.mean(results['val_losses'])}")
    print(f"Average training accuracy: {np.mean(results['train_accs'])}")
    print(f"Average validation accuracy: {np.mean(results['val_accs'])}")

    print('K-fold cross-validation completed.')

    return results

def train_model(model, model_name, criterion, optimizer, train_loader, val_loader, fold_num=0, dynamic_lr=False, use_validation = True):
    '''
    Train the model

    param: model: Model: Model instance
    param: model_name: str: Name of the model
    param: criterion: loss: Loss function
    param: optimizer: optimizer: Optimizer
    param: train_loader: DataLoader: Training data loader
    param: val_loader: DataLoader: Validation data loader
    param: fold_num: int: Fold number
    param: dynamic_lr: bool: Whether to use dynamic learning rate
    param: use_validation: bool: Whether to use validation

    return: train_losses: list: List of training losses
    '''
    print("Dinamic lr: ", dynamic_lr)
    # Check if fold_num is specified to identify a k-fold training scenario
    if fold_num > 0:  
        print(f"Training Fold {fold_num}")
    
    # Initialize learning rates list for plotting
    learning_rates = []
    if dynamic_lr:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 5))


    # Training loop with validation
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Iterate over each epoch
    for epoch in tqdm(range(settings.NUM_EPOCHS), desc=f'Training {model_name} model'):
        # Set the model to training mode
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Iterate over each batch
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, _, _ = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        # Calculate average training loss and accuracy
        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Step and save learning rate for plotting  
        if dynamic_lr:
            scheduler.step()
            learning_rates.append(scheduler.get_last_lr()[0])
        
        if use_validation:
            # Calculate validation loss and accuracy
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            predicted_labels = []
            true_labels = []
            with torch.no_grad():
                for data, targets in val_loader:
                    outputs, _, _ = model(data)
                    val_loss = criterion(outputs, targets)
                    running_val_loss += val_loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_labels.extend(predicted.tolist())
                    true_labels.extend(targets.tolist())
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()

            # Calculate average validation loss and accuracy
            avg_val_loss = running_val_loss / len(val_loader)
            val_accuracy = correct_val / total_val

            # Save txt file with the model's train and validation losses and accuracies
            with open(f'plots/{model.__class__.__name__}_fold_{fold_num}_losses_accuracies.txt', 'a') as f:
                f.write(f'Epoch [{epoch+1}/{settings.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}\n')
        else:
            # Save txt file with the model's train losses and accuracies
            with open(f'plots/{model.__class__.__name__}_fold_{fold_num}_losses_accuracies_train+val.txt', 'a') as f:
                f.write(f'Epoch [{epoch+1}/{settings.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}\n')
        # Save losses and accuracies for plotting
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        if use_validation:
            val_losses.append(avg_val_loss)
            val_accs.append(val_accuracy)
        else:
            val_losses.append(0)
            val_accs.append(0)
        
    # Save the model
    torch.save(model.state_dict(), f'models/{model_name}_fold_{fold_num}.pth')
    
    # Plot the learning rate evolution
    if (not os.path.exists("plots/learning_rate_evolution.png")) and dynamic_lr:
        plot_lr_evolution(learning_rates)

    # Plot the confusion matrix
    if use_validation:
        plot_confusion_matrix(model_name, true_labels, predicted_labels, classes=["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])

    return train_losses, val_losses, train_accs, val_accs, learning_rates