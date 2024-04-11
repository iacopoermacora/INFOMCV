import torch
from torchsummary import summary
import torch.nn as nn
import torchvision.models as models

# 1. Stanford 40 – Frames: Create a CNN and train it on the images in Stanford 40. Naturally, you will have 12 output classes.

def Stanford40_model(num_classes=12, dropout_prob=0.5):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    
    # Modify the output layer to have num_classes classes
    num_ftrs = model.fc.in_features
    # Create a new Sequential model for the classifier
    # It includes a Dropout layer followed by the final Linear layer
    model.fc = nn.Sequential(
        nn.Dropout(dropout_prob),  # Add dropout with a probability of dropout_prob
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

# 2. HMDB51 – Frames (transfer learning): Use your pretrained CNN (same architecture/weights) and fine-tune it on the middle 
#    frame of videos of the HMDB51 dataset. You can use a different learning rate than for the Stanford 40 network training.

def HMDB51_model(num_classes=12, dropout_prob=0.5):
    # Load the pre-trained stanford40 model from the standford40.pth file
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')  # Change to the correct model architecture

    # Modify the output layer to have num_classes classes
    num_ftrs = model.fc.in_features
    # Create a new Sequential model for the classifier
    # It includes a Dropout layer followed by the final Linear layer
    model.fc = nn.Sequential(
        nn.Dropout(dropout_prob),  # Add dropout with a probability of dropout_prob
        nn.Linear(num_ftrs, num_classes)
    )

    # Load the pre-trained Stanford40 model state dictionary
    state_dict = torch.load('Stanford40_model_dynamic.pth')
    model.load_state_dict(state_dict)

    freeze = True
    for name, param in model.named_parameters():
        if freeze and 'layer4' not in name:  # Change 'bottleneck.162' accordingly
            param.requires_grad = False
        else:
            freeze = False
    
    # Check which layers are frozen
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    
    return model

# 3. HMDB51 – Optical flow: Create a new CNN and train it on the optical flow of videos in HMBD51. You can use the middle frame
#    (max 5 points) or stack a fixed number (e.g., 16) of optical flow frames together (max 10 points).

class HMDB51_OF_model_old(nn.Module):
    def __init__(self):
        super(HMDB51_OF_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=7, stride=2), # TODO: Change in_channels accordingly
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.full6 = nn.Sequential(
            nn.Linear(512*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.full7 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.linear = nn.Linear(2048, 12)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv5 for FC layers
        x = self.full6(x)
        x = self.full7(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x

def HMDB51_OF_model(num_classes=12, dropout_prob=0.5):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    
    # Change the first layer to accept 32 channels instead of 3
    model.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify the output layer to have num_classes classes
    num_ftrs = model.fc.in_features
    # Create a new Sequential model for the classifier
    # It includes a Dropout layer followed by the final Linear layer
    model.fc = nn.Sequential(
        nn.Dropout(dropout_prob),  # Add dropout with a probability of dropout_prob
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

def Fashion_model(num_classes=10, dropout_prob=0.5):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    
    # Change the first layer to accept 32 channels instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # Modify the output layer to have num_classes classes
    num_ftrs = model.fc.in_features
    # Create a new Sequential model for the classifier
    # It includes a Dropout layer followed by the final Linear layer
    model.fc = nn.Sequential(
        nn.Dropout(dropout_prob),  # Add dropout with a probability of dropout_prob
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

def HMDB51_OF_VGG_model(num_classes=12, dropout_prob=0.5):
    # Load the pre-trained VGG-16 model
    model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    
    # Change the first layer to accept 32 channels instead of 3
    model.features[0] = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    # Modify the output layer to have num_classes classes
    num_ftrs = model.classifier[6].in_features
    # Create a new Sequential model for the classifier
    # It includes a Dropout layer followed by the final Linear layer
    model.classifier[6] = nn.Sequential(
        nn.Dropout(dropout_prob),  # Add dropout with a probability of dropout_prob
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=12):
        super(ActionRecognitionModel, self).__init__()
        
        self.conv1 = nn.Conv3d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=3)
        
        self.fc1 = nn.Linear(512*6*6*1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        
        x = x.view(-1, 512*1*6*6)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# 4. HMDB51 – Two-stream: Finally, create a two-stream CNN with one stream for the frames and one stream for the optical flow. 
#    Use your pre-trained CNNs to initialize the weights of the two branches. Think about how to fuse the two streams and motivate 
#    this in your report. Look at the Q&A at the end of this assignment. Fine-tune the network.

