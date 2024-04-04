import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from HMDB51 import train_files, train_labels, test_files, test_labels

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
        video_path = os.path.join(self.root_dir, self.labels[idx], self.file_paths[idx])
        optical_flow_folder = os.path.join(self.root_dir, "optical_flow_images")
        flow_stack = []

        for i in range(1, 16):
            flow_img_path = os.path.join(optical_flow_folder, f"{idx}_{i}.png")
            flow_img = cv2.imread(flow_img_path, cv2.IMREAD_GRAYSCALE)
            flow_stack.append(flow_img)

        return np.stack(flow_stack)

# FUNCTIONS

def extract_optical_flow_and_save(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, frame_count, 16):  # Extract 16 evenly spaced frames
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        if not ret:
            break

        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate magnitude and angle
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize magnitude to range [0, 255] and save as grayscale image
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{i}.png"), mag)

    cap.release()

# Create optical flow images for training set
for i, video_file, video_label in enumerate(zip(train_files, train_labels)):
    video_path = os.path.join("video_data", video_label, video_file)
    output_folder = os.path.join("optical_flow_images", video_label)
    extract_optical_flow_and_save(video_path, output_folder)

# Create optical flow images for test set
for i, video_file, video_label in enumerate(zip(test_files, test_labels)):
    video_path = os.path.join("video_data", video_label, video_file)
    output_folder = os.path.join("optical_flow_images", video_label)
    extract_optical_flow_and_save(video_path, output_folder)

'''# Create custom dataset
train_dataset = OpticalFlowDataset(range(len(train_files)), train_labels, root_dir="video_data")
test_dataset = OpticalFlowDataset(range(len(test_files)), test_labels, root_dir="video_data")

# Example usage of DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)'''
