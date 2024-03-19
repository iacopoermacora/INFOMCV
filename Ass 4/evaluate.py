import torch
import settings
import matplotlib.pyplot as plt

def test_model(model, test_loader):
    # Testing
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_accuracy = correct / total

    return test_accuracy

def see_model_outputs(model, data):
    # Testing
    model.eval()  # Set the model to evaluation mode
    _, feature_maps_conv1= model(data)

    # Visualize feature maps
    plt.figure(figsize=(10, 4))

    # Visualize feature maps after the first convolutional layer
    for i in range(12):  # Assuming there are 12 feature maps in the first convolutional layer
        plt.subplot(2, 6, i + 1)
        plt.imshow(feature_maps_conv1[0, i, :, :], cmap='gray')
        plt.axis('off')
        plt.title(f'Conv1 Feature Map {i+1}')

    plt.tight_layout()
    plt.savefig('plots/feature_maps_conv1.png')

def plot_losses_accuracies(train_losses, val_losses, train_accs, val_accs, title="Model"):
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, settings.NUM_EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, settings.NUM_EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - ' + title)
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, settings.NUM_EPOCHS + 1), train_accs, label='Training Accuracy')
    plt.plot(range(1, settings.NUM_EPOCHS + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy - ' + title)
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/' + title + '_losses_accuracies.png')