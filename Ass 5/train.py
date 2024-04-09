import torch
from torch.optim.lr_scheduler import CyclicLR, StepLR
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


def initialize_model(model_class):
    '''
    Initialize the model

    param: model_class: class: Model class

    return: model: Model instance
    '''
    model = model_class()
    return model

def train_and_validate(model, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs):
    device = torch.device("cpu")
    model.to("cpu")
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_train_correct, total_train_samples = 0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

            # Update learning rate after each batch for CyclicLR
            if isinstance(scheduler, CyclicLR):
                scheduler.step()
        
        # Update learning rate after each epoch for other schedulers
        if not isinstance(scheduler, CyclicLR):
            scheduler.step()
        
        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = total_train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criteria(outputs, labels)
                
                total_val_loss += val_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

                # Collect all predictions and true labels
                all_preds.extend(predicted.view(-1).cpu().numpy())
                all_true.extend(labels.view(-1).cpu().numpy())

        avg_val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
                
        print(f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}, '
                f'Train Acc: {train_accuracy:.4f}, '
                f'Val Acc: {val_accuracy:.4f}')
    
    # Compute confusion matrix
    classes = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
            "run", "shoot_bow", "smoke", "throw", "wave"]
    cm = confusion_matrix(all_true, all_preds)
    plot_confusion_matrix(cm, classes, title='Confusion Matrix')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    # save txt file with the model's train and validation losses and accuracies for each epoch
    with open(f'plots/{model.__class__.__name__}_losses_accuracies.txt', 'a') as f:
        for epoch in range(num_epochs):
            f.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[epoch]:.4f}, Val Loss: {val_losses[epoch]:.4f}, Train Acc: {train_accuracies[epoch]:.4f}, Val Acc: {val_accuracies[epoch]:.4f}\n')
    
    # save confusion matrix plot
    plt.savefig(f'plots/{model.__class__.__name__}_confusion_matrix.png')

    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss')
    plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'o-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'metrics_plot_{model_name}.png')