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