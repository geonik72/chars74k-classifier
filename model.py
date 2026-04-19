import torch
import torch.nn as nn
import torch.nn.functional as F


input_height = 128 # Height of the input image for now this is for a square image so width is the same as height
output_size = 62 # 10 digits + 26 uppercase + 26 lowercase = 62 classes

class CharacterClassification(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Convolutional Block 1: 1@128x128 -> 32@64x64
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 2: 32@64x64 -> 64@32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 3: 64@32x32 -> 128@16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 4: 128@16x16 -> 256@8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            # Fully Connected Layers: 256*8*8 -> 512 -> 62
            nn.Flatten(), # Flattens tensor from (batch_size, 256, 8, 8) to (batch_size, 256*8*8)
            nn.Linear(256 * (input_height // 16) ** 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x