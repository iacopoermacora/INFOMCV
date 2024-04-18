import torch
from torchsummary import summary
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import settings

# 1. Stanford 40 – Frames: Create a CNN and train it on the images in Stanford 40. Naturally, you will have 12 output classes.

'''def Stanford40_model(num_classes=12, dropout_prob=0.5): # TODO: Change this to a class
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
    
    return model'''

class Stanford40_model(nn.Module):
    def __init__(self, num_classes=12, dropout_prob=0.5):
        super(Stanford40_model, self).__init__()
        self.resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        x = self.resnet50(x)
        return x

# 2. HMDB51 – Frames (transfer learning): Use your pretrained CNN (same architecture/weights) and fine-tune it on the middle 
#    frame of videos of the HMDB51 dataset. You can use a different learning rate than for the Stanford 40 network training.

'''def HMDB51_Frame_Model(num_classes=12, dropout_prob=0.5): # TODO: Change this to a class
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
    
    return model'''

class HMDB51_Frame_Model(nn.Module):
    def __init__(self, num_classes=12, dropout_prob=0.5):
        super(HMDB51_Frame_Model, self).__init__()
        self.resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_ftrs, num_classes)
        )

        # Load the pre-trained Stanford40 model state dictionary
        state_dict = torch.load(f'Stanford40_model_{settings.LR_SCHEDULER_TYPE["Stanford40"]}.pth')
        self.resnet50.load_state_dict(state_dict)

        freeze = True
        for name, param in self.resnet50.named_parameters():
            if freeze and 'layer4' not in name:  # Change 'bottleneck.162' accordingly
                param.requires_grad = False
            else:
                freeze = False
    
    def forward(self, x):
        x = self.resnet50(x)
        return x

# 3. HMDB51 – Optical flow: Create a new CNN and train it on the optical flow of videos in HMBD51. You can use the middle frame
#    (max 5 points) or stack a fixed number (e.g., 16) of optical flow frames together (max 10 points).

class Conv2Plus1D(nn.Module): # TODO: CHange names and reorder
    def __init__(self, in_channels, filters, kernel_size, padding):
        super(Conv2Plus1D, self).__init__()
        self.spatial_conv = nn.Conv3d(in_channels=in_channels,
                                      out_channels=filters,
                                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                                      padding=padding)
        self.temporal_conv = nn.Conv3d(in_channels=filters,
                                       out_channels=filters,
                                       kernel_size=(kernel_size[0], 1, 1),
                                       padding=padding)
    
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(ResidualBlock, self).__init__()
        self.seq = nn.Sequential(
            Conv2Plus1D(in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            nn.ReLU(),
            Conv2Plus1D(in_channels=filters,
                        filters=filters,
                        kernel_size=kernel_size,
                        padding='same')
        )
        self.downsample = None
        if in_channels != filters:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, filters, kernel_size=1, bias=False),
            )
    
    def forward(self, x):
        res = x
        x = self.seq(x)
        if self.downsample is not None:
            res = self.downsample(res)

        return x + res

class HMDB51_OF_Model(nn.Module):
    def __init__(self):
        super(HMDB51_OF_Model, self).__init__()
        self.initial_conv = Conv2Plus1D(in_channels=2, filters=8, kernel_size=(3, 3, 3), padding='same')
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(8, 16, (3, 3, 3)),
            ResidualBlock(16, 32, (3, 3, 3)),
            ResidualBlock(32, 64, (3, 3, 3)),
            ResidualBlock(64, 128, (3, 3, 3))
        ])
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.35)
        self.final_dense = nn.Linear(128, 12)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.relu(x)
        for block in self.res_blocks:
            x = block(x)
            # Add max pooling after each residual block except the last one
            if block != self.res_blocks[-1]:
                x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final_dense(x)
        return x

# 4. HMDB51 – Two-stream: Finally, create a two-stream CNN with one stream for the frames and one stream for the optical flow. 
#    Use your pre-trained CNNs to initialize the weights of the two branches. Think about how to fuse the two streams and motivate 
#    this in your report. Look at the Q&A at the end of this assignment. Fine-tune the network.

class HMDB51_Fusion_Model(nn.Module):
    def __init__(self):
        super(HMDB51_Fusion_Model, self).__init__()
        self.frame_model = HMDB51_Frame_Fusion()
        
        self.of_model = HMDB51_OF_Fusion()

        # 1X1 convolutions
        self.conv1x1 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(2, 1, 1))
        self.convFusion = nn.Conv2d(in_channels=2048+128, out_channels=128, kernel_size=(1, 1))
        self.fc1 = nn.Linear(128*7*7, 128)
        self.fc2 = nn.Linear(128, 12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.35)

    def forward(self, frame_x, of_x):
        frame_x = self.frame_model(frame_x)
        of_x = self.of_model(of_x)
        of_x = self.conv1x1(of_x)
        # Squeeze the 1st dimension
        of_x = torch.squeeze(of_x, dim=2)
        x = torch.cat((frame_x, of_x), dim=1)
        # Convolution
        x = self.convFusion(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HMDB51_Frame_Fusion(nn.Module):
    def __init__(self, num_classes=12, dropout_prob=0.5):
        super(HMDB51_Frame_Fusion, self).__init__()

        self.resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')  # Change to the correct model architecture

        # Modify the output layer to have num_classes classes
        num_ftrs = self.resnet50.fc.in_features
        # Create a new Sequential model for the classifier
        # It includes a Dropout layer followed by the final Linear layer
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(dropout_prob),  # Add dropout with a probability of dropout_prob
            nn.Linear(num_ftrs, num_classes)
        )

        # Load the pre-trained HMDB51 model state dictionary
        state_dict = torch.load(f'HMDB51_Frame_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_Frames"]}.pth')
        self.resnet50.load_state_dict(state_dict)

        # Freeze layers
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        return x
    
class HMDB51_OF_Fusion(nn.Module):
    def __init__(self):
        super(HMDB51_OF_Fusion, self).__init__()
        self.HMDB51_OF = HMDB51_OF_Model()
        state_dict = torch.load(f'HMDB51_OF_Model_{settings.LR_SCHEDULER_TYPE["HMDB51_OF"]}.pth')
        self.HMDB51_OF.load_state_dict(state_dict)

        # Freeze layers
        self.freeze_all_layers(self.HMDB51_OF)

        # Convolution layer to get output from (128, 2, 9, 9) to (128, 2, 7, 7)
        self.final_conv = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding='valid')

    
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def freeze_all_layers(self, model):
        self.freeze_model(model)

        # Recursively freeze parameters in submodules
        for module in model.modules():
            if isinstance(module, nn.Module):
                self.freeze_model(module)

    def forward(self, x):
        x = self.HMDB51_OF.initial_conv(x)
        x = self.HMDB51_OF.relu(x)
        for block in self.HMDB51_OF.res_blocks:
            x = block(x)
            if block != self.HMDB51_OF.res_blocks[-1]:
                x = self.HMDB51_OF.max_pool(x)
        x = self.final_conv(x)
        return x