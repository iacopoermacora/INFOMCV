import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

# DATASETS

# Custom dataset class for Stanford40 dataset
class Stanford40_Dataset(Dataset):
    def __init__(self, img_dir, file_paths, labels, transform=None):
        '''
        img_dir: directory containing the images
        file_paths: list of file paths for the images
        labels: list of labels for the images
        transform: torchvision.transforms object to apply transformations to the images
        '''
        self.img_dir = img_dir
        self.file_paths = file_paths
        self.labels = labels
        self.label_map = {
            "applauding": 0,
            "climbing": 1,
            "drinking": 2,
            "jumping": 3,
            "pouring_liquid": 4,
            "riding_a_bike": 5,
            "riding_a_horse": 6,
            "running": 7,
            "shooting_an_arrow": 8,
            "smoking": 9,
            "throwing_frisby": 10,
            "waving_hands": 11
        }
        self.transform = transform
        # Map each file path to its label
        self.file_label_map = {file_path: label for file_path, label in zip(file_paths, labels)}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Use the idx to get the file path and then find its corresponding label from the map
        file_path = self.file_paths[idx]
        label = self.file_label_map[file_path]
        label_idx = self.label_map[label]
        img_path = os.path.join(self.img_dir, file_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor

# Custom dataset class for HMDB51 dataset (images)
class HMDB51_Frame_Dataset(Dataset):
    def __init__(self, files, labels, root_dir, frame_number, transform=None):
        '''
        files: list of video filenames
        labels: list of labels for the videos
        root_dir: root directory containing the video frames
        frame_number: frame number to extract from each video
        transform: torchvision.transforms object to apply transformations to the images
        '''
        self.files = files
        self.labels = labels
        self.label_map = {
            "clap": 0,
            "climb": 1,
            "drink": 2,
            "jump": 3,
            "pour": 4,
            "ride_bike": 5,
            "ride_horse": 6,
            "run": 7,
            "shoot_bow": 8,
            "smoke": 9,
            "throw": 10,
            "wave": 11
        }
        self.root_dir = root_dir
        self.transform = transform
        self.frame_number = frame_number

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_filename = self.files[idx]
        label = self.labels[idx]
        label_idx = self.label_map[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # Assuming video filename format is "name.avi"
        video_name = os.path.splitext(video_filename)[0]
        
        # Constructing the image filename with frame_number suffix
        image_filename = f"{video_name}_{self.frame_number}.png"
        
        # Constructing the full path to the image file
        image_path = os.path.join(self.root_dir, label, image_filename)
        
        # Loading the image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_tensor


# Custom dataset class for HMDB51 dataset (optical flow)
class HMDB51_OF_Dataset(Dataset):
    def __init__(self, file_paths, labels, root_dir):
        '''
        file_paths: list of optical flow file paths
        labels: list of labels for the optical flow files
        root_dir: root directory containing the optical flow files
        '''
        self.file_paths = file_paths
        self.labels = labels
        self.label_map = {
            "clap": 0,
            "climb": 1,
            "drink": 2,
            "jump": 3,
            "pour": 4,
            "ride_bike": 5,
            "ride_horse": 6,
            "run": 7,
            "shoot_bow": 8,
            "smoke": 9,
            "throw": 10,
            "wave": 11
        }
        self.root_dir = root_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        label_idx = self.label_map[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # Retrieve the numpy array of the optical flow stack
        flow_stack = np.load(os.path.join(self.root_dir, self.labels[idx], os.path.splitext(self.file_paths[idx])[0]+'_flow.npy'))
        flow_stack = torch.tensor(flow_stack).float()

        return flow_stack, label_tensor

class HMDB51_Fusion_Dataset(Dataset):
    def __init__(self, files, labels, frames_root_dir, frame_number, optical_flow_root_dir, transform=None):
        '''
        files: list of video filenames
        labels: list of labels for the videos
        frames_root_dir: root directory containing the video frames
        frame_number: frame number to extract from each video
        optical_flow_root_dir: root directory containing the optical flow data
        transform: torchvision.transforms object to apply transformations to the images
        '''
        self.files = files
        self.labels = labels
        self.label_map = {
            "clap": 0,
            "climb": 1,
            "drink": 2,
            "jump": 3,
            "pour": 4,
            "ride_bike": 5,
            "ride_horse": 6,
            "run": 7,
            "shoot_bow": 8,
            "smoke": 9,
            "throw": 10,
            "wave": 11
        }
        self.frames_root_dir = frames_root_dir
        self.transform = transform
        self.frame_number = frame_number
        self.optical_flow_root_dir = optical_flow_root_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_filename = self.files[idx]
        label = self.labels[idx]
        label_idx = self.label_map[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        # Assuming video filename format is "name.avi"
        video_name = os.path.splitext(video_filename)[0]
        
        # Constructing the image filename with frame_number suffix
        image_filename = f"{video_name}_{self.frame_number}.png"
        
        # Constructing the full path to the image file
        image_path = os.path.join(self.frames_root_dir, label, image_filename)
        
        # Loading the image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Loading optical flow data
        optical_flow_path = os.path.join(self.optical_flow_root_dir, label, f"{video_name}_flow.npy")
        optical_flow = torch.tensor(np.load(optical_flow_path)).float()
        
        return image, optical_flow, label_tensor
