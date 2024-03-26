import torch
import settings
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import LeNet5Variant4
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def test_model(test_loader, fold_num=0):
    # Load the model
    model = LeNet5Variant4()
    model.load_state_dict(torch.load(f'models/LeNet5Variant4_fold_{fold_num}.pth'))
    # Testing
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Lists to store the embeddings and labels
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            outputs, embedding = model(data)
            # Extract embeddings from the fully connected layer (before softmax)
            embeddings.append(embedding)
            labels.append(targets)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Concatenate embeddings and labels
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    # Convert embeddings to numpy array
    embeddings_np = embeddings.cpu().numpy()

    
    
    if settings.TSNE_COMPONENTS == 2:
        print("Creating TSNE...")
        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings_np)
        print("TNSE created!")
        # Class names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # Plot the t-SNE visualization
        plt.figure(figsize=(10, 8))
        for i in range(10):  # Assuming you have 10 classes
            indices = labels == i
            plt.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], label=class_names[i])
        plt.legend()
        plt.title('t-SNE Visualization of Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.savefig('plots/tsne_visualization.png')
    elif settings.TSNE_COMPONENTS:
        print("Creating TSNE...")
        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings_np)
        print("TNSE created!")
        # Class names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # Plot the t-SNE visualization with points in 3D
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(10):  # Assuming you have 10 classes
            indices = labels == i
            ax.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], embeddings_tsne[indices, 2], label=class_names[i])
        ax.legend()
        ax.set_title('t-SNE Visualization of Embeddings (3D)')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        plt.show()

    test_accuracy = correct / total

    return test_accuracy

def see_model_outputs(model, data):
    # Testing
    model.eval()  # Set the model to evaluation mode
    _, _, feature_maps_conv1= model(data)

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

def plot_feature_maps(intermediates, class_name):
    for layer_idx, (feature_maps, filters) in enumerate(intermediates):
        num_feature_maps = feature_maps.size(1)
        num_filters = filters.size(0)

        if layer_idx == 0:
            subplots_per_column = num_feature_maps
            subplots_per_row = 2
        else:
            subplots_per_column = int(num_feature_maps/4)
            subplots_per_row = 4
        # Create 6 subplots
        fig, axs = plt.subplots(subplots_per_column, subplots_per_row, figsize=(subplots_per_row*4, subplots_per_column*5))

        # Visualize each feature map with its corresponding filter
        for i in range(num_feature_maps):
            feature_map = feature_maps[:, i, :, :]
            filter = filters[i, :, :, :]

            if subplots_per_row == 2:
                axs[i, 0].imshow(filter.squeeze().detach().numpy(), cmap='gray')
                axs[i, 0].set_title(f"Filter {i+1}")
                axs[i, 0].axis('off')

                axs[i, 1].imshow(feature_map.squeeze().detach().numpy(), cmap='gray')
                axs[i, 1].set_title(f"Feature Map {i+1}")
                axs[i, 1].axis('off')
            else:
                row = int(i/subplots_per_row)
                col = int(i % subplots_per_row)

                axs[row, col].imshow(feature_map.squeeze().detach().numpy(), cmap='gray')
                axs[row, col].set_title(f"Feature Map {i+1}")
                axs[row, col].axis('off')

            fig.suptitle(f'Visualization of Feature Maps for Conv Layer {layer_idx} - {class_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'plots/featureMaps_conv{layer_idx}_{class_name}.png')