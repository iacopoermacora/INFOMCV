import settings
import torch
import torch.optim.lr_scheduler as lr_scheduler
import os
import matplotlib.pyplot as plt

def train_model(model, model_name, criterion, optimizer, train_loader, val_loader, dynamic_lr=False):
    if dynamic_lr:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 5))
        learning_rates = []

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
        with open('plots/' + model.__class__.__name__ + '_losses_accuracies.txt', 'a') as f:
            f.write(f'Epoch [{epoch+1}/{settings.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}\n')

        # Save losses and accuracies for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    # Save the model
    torch.save(model, f'models/{model_name}.pth')
    
    if (not os.path.exists("plots/learning_rate_evolution.png")) and dynamic_lr:
        # Plot the evolution of learning rate with respect to the epoch
        plt.plot(range(1, settings.NUM_EPOCHS + 1), learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Evolution')
        plt.grid(True)
        plt.savefig('plots/learning_rate_evolution.png')

    return train_losses, val_losses, train_accs, val_accs