import os
import cv2
import numpy as np
from HMDB51 import train_files, train_labels, test_files, test_labels
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from stanford40 import create_stanford40_splits

# DATASETS
class CustomStandford40Dataset(Dataset):
    def __init__(self, img_dir, file_paths, labels, transform=None):
        """
        Args:
            img_dir (string): Path to the image directory.
            file_paths (list): List of file paths relative to img_dir.
            labels (list): List of labels corresponding to each file path.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        # Map each file path to its label
        self.file_label_map = {file_path: label for file_path, label in zip(file_paths, labels)}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Use the idx to get the file path and then find its corresponding label from the map
        file_path = self.file_paths[idx]
        label = self.file_label_map[file_path]
        img_path = os.path.join(self.img_dir, file_path)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


class OpticalFlowDataset(Dataset):
    def __init__(self, file_paths, labels, root_dir, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        flow_stack = self.load_optical_flow_stack(idx)
        label = self.labels[idx]

        if self.transform:
            flow_stack = self.transform(flow_stack)

        return flow_stack, label

    def load_optical_flow_stack(self, idx):
        flow_stack = []

        flow_folder_path = os.path.join(self.root_dir, self.labels[idx])
        print(self.file_paths[idx])
        image_files = [file for file in os.listdir(flow_folder_path) if file.startswith(os.path.splitext(self.file_paths[idx])[0])]
                       
        for image in image_files:
            flow_img_path = os.path.join(flow_folder_path, image)
            flow_img = cv2.imread(flow_img_path, cv2.IMREAD_GRAYSCALE)
            flow_stack.append(flow_img)

        return np.stack(flow_stack)


# STANFORD40 DATASET

# Create Stanford40 dataset
stanford40_splits = create_stanford40_splits()
train_files, train_labels, test_files, test_labels = stanford40_splits

# Create custom dataset
# Initialize the datasets
train_dataset = CustomStandford40Dataset(img_dir='photo_dataset/train', file_paths=train_files, labels=train_labels, transform=ToTensor())
test_dataset = CustomStandford40Dataset(img_dir='photo_dataset/test', file_paths=test_files, labels=test_labels, transform=ToTensor())

# Split the training dataset into training and validation
validation_split = 0.15
num_train = len(train_dataset)
num_validation = int(num_train * validation_split)
num_train -= num_validation
train_data, validation_data = random_split(train_dataset, [num_train, num_validation])

# Setup the DataLoader for each split
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print("Data loaders created successfully.")

# HMDB51 DATASET (FRAMES)

# HMDB51 DATASET (OPTICAL FLOW)

'''# Create custom dataset
train_dataset = OpticalFlowDataset(train_files, train_labels, root_dir="optical_flow_images")
test_dataset = OpticalFlowDataset(test_files, test_labels, root_dir="optical_flow_images")
print(test_dataset.__getitem__(0))
# Example usage of DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)'''

