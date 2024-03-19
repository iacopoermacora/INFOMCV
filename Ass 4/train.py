import settings
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader, Subset

def initialize_model(model_class):
    model = model_class()
    return model

# Train and validate the model using k-fold cross-validation
def k_fold_train_and_validate(model_constr, model_name, criterion, training_data, num_folds=5, dynamic_lr=False):
    kfold = KFold(n_splits=num_folds, shuffle=True)

    results = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': []
    }
    learning_rates_per_fold = []

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

    # After completing all folds, plotting the learning rate evolution across folds
    if dynamic_lr:
        plt.figure(figsize=(10, 6))
        for i, lr in enumerate(learning_rates_per_fold):
            plt.plot(range(1, len(lr) + 1), lr, label=f'Fold {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Evolution Across Folds')
        plt.legend()
        plt.grid(True)
        plt.show()

    return results


def train_model(model, model_name, criterion, optimizer, train_loader, val_loader, fold_num=0, dynamic_lr=False):
    # Check if fold_num is specified to identify a k-fold training scenario
    if fold_num > 0:  
        print(f"Training Fold {fold_num}")
    
    learning_rates = []
    if dynamic_lr:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 5))


    # Training loop with validation
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(settings.NUM_EPOCHS):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, targets)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        # Calculate average training loss and accuracy
        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        if dynamic_lr:
            scheduler.step()
            learning_rates.append(scheduler.get_last_lr()[0])
        
        # Calculate validation loss and accuracy
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        # Calculate average validation loss and accuracy
        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        print(f'Epoch [{epoch+1}/{settings.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        # Save txt file with the model's train and validation losses and accuracies
        with open(f'plots/{model.__class__.__name__}_fold_{fold_num}_losses_accuracies.txt', 'a') as f:
            f.write(f'Epoch [{epoch+1}/{settings.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}\n')

        # Save losses and accuracies for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    # Save the model
    torch.save(model.state_dict(), f'models/{model_name}_fold_{fold_num}.pth')
    
    if (not os.path.exists("plots/learning_rate_evolution.png")) and dynamic_lr:
        if fold_num == 0:
            # Plot the evolution of learning rate with respect to the epoch
            plt.plot(range(1, settings.NUM_EPOCHS + 1), learning_rates)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Evolution')
            plt.grid(True)
            plt.savefig('plots/learning_rate_evolution.png')      

    return train_losses, val_losses, train_accs, val_accs, learning_rates