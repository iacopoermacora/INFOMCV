import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn

# Implement the LeNet5 model
# NOTE: In the original architecture they use tanh and not ReLU, I am pretty sure
# https://readmedium.com/en/https:/medium.com/@siddheshb008/lenet-5-architecture-explained-3b559cb2d52b

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4x4 image dimension after 2x2 pooling twice TODO: Check if it is fine changing to 4x4 instead than 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes for Fashion-MNIST

        # Initialize weights using Kaiming Uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # First convolutional layer with ReLU activation and 2x2 max pooling
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(F.max_pool2d(x, 2, stride=2))
        # Second convolutional layer with ReLU activation and 2x2 max pooling
        x = self.conv2(x)
        x = torch.tanh(F.max_pool2d(x, 2, stride=2))
        # Flatten the output for fully connected layers
        x = x.view(-1, self.num_flat_features(x)) # NOTE: Not completely sure what is happening here
        # Fully connected layers with ReLU activation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # Output layer with softmax activation
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # Exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4x4 image dimension after 2x2 pooling twice
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes for Fashion-MNIST

        # Initialize weights using Kaiming Uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # First convolutional layer with ReLU activation and 2x2 max pooling
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(F.max_pool2d(x, 2, stride=2))
        # Second convolutional layer with ReLU activation and 2x2 max pooling
        x = self.conv2(x)
        x = torch.tanh(F.max_pool2d(x, 2, stride=2))
        # Flatten the output for fully connected layers
        x = x.view(-1, self.num_flat_features(x)) # NOTE: Not completely sure what is happening here
        # Fully connected layers with ReLU activation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # Output layer with softmax activation
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # Exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
# First variant model, with ReLU activation
class LeNet5Variant1(nn.Module):
    def __init__(self):
        super(LeNet5Variant1, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4x4 image dimension after 2x2 pooling twice
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes for Fashion-MNIST

        # Initialize weights using Kaiming Uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # First convolutional layer with ReLU activation and 2x2 max pooling
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 2, stride=2))
        # Second convolutional layer with ReLU activation and 2x2 max pooling
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2, stride=2))
        # Flatten the output for fully connected layers
        x = x.view(-1, self.num_flat_features(x)) # NOTE: Not completely sure what is happening here
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer with softmax activation
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # Exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features