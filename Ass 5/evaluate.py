import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import settings
import itertools

def test_model(model, test_loader, scheduler_type, device='cpu'):
    '''
    Test the model on the test set and calculate the accuracy

    model: model to test
    test_loader: DataLoader object containing the test data
    scheduler_type: type of scheduler used during training
    device: device to run the model on

    return: accuracy of the model on the test set
    '''
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Lists to store the predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        # Iterate over the test data
        for data in tqdm(test_loader):
            # Check if the data contains data for the fusion model and extract the data accordingly
            if len(data) == 3:
                photos, flows, labels = data
                outputs = model(photos, flows)
            else:
                inputs, labels = data
                outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append the predictions and labels to the lists
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    # Calculate and plot the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(f'plots/{model.__class__.__name__}/CM/Test_CM_{model.__class__.__name__}_{scheduler_type}.png')
    plt.show()

    return accuracy, cm

def plot_confusion_matrix(model_name, cm, classes, scheduler_type, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Plot the confusion matrix and save it

    model_name: name of the model
    cm: confusion matrix
    classes: list of class names
    scheduler_type: type of scheduler used during training
    title: title of the plot
    cmap: color map to use for the plot

    return: None
    '''
    plt.figure(figsize=(10, 10))
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
    # save confusion matrix plot in the plots folder
    plt.savefig(f'plots/{model_name}/CM/Val_CM_{model_name}_{scheduler_type}_{title}.png')
    plt.close()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, scheduler_type):
    '''
    Plot the training and validation losses and accuracies and save the plot

    train_losses: list of training losses
    val_losses: list of validation losses
    train_accuracies: list of training accuracies
    val_accuracies: list of validation accuracies
    model_name: name of the model
    scheduler_type: type of scheduler used during training

    return: None
    '''
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
    # Save the plot inside the plots folder 
    plt.savefig(f'plots/{model_name}/{model_name}_{scheduler_type}_metrics.png')
    plt.close()

def plot_learning_rate(learning_rates, scheduler_type, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Batch' if scheduler_type == 'cyclic' else 'Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Evolution ({scheduler_type} scheduler)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot inside the plots folder with the name of the model and the scheduler type
    plt.savefig(f'plots/{model_name}/{model_name}_{scheduler_type}_learning_rate.png')
    plt.close()
