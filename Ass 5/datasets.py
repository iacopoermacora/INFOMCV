import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from HMDB51 import create_hmdb51_splits
from tqdm import tqdm

# DATASETS
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

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
            "run", "shoot_bow", "smoke", "throw", "wave"]
train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)

# Create custom dataset
train_dataset = OpticalFlowDataset(train_files, train_labels, root_dir="optical_flow_images")
test_dataset = OpticalFlowDataset(test_files, test_labels, root_dir="optical_flow_images")
print(test_dataset.__getitem__(0))
# Example usage of DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
