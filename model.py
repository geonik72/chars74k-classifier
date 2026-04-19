import torch
import torch.nn as nn
import torch.nn.functional as F


input_height = 128 # Height of the input image for now this is for a square image so width is the same as height
output_size = 62 # 10 digits + 26 uppercase + 26 lowercase = 62 classes

class CharacterClassification(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the CNN layers with Batch Normalization
        # First convolutional layer: input channels = 1, output channels = 32, kernel = 3x3 with batch normalization
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32) # Batch normalization for the first convolutional layer
        # Second convolutional layer: input channels = 32, output channels = 64, kernel = 3x3 with batch normalization
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Third convolutional layer: input channels = 64, output channels = 128, kernel = 3x3 with batch normalization
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fourth convolutional layer: input channels = 128, output channels = 256, kernel = 3x3 with batch normalization
        self.cnn4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dimensions by a factor of 2 each time it's applied
        
        # Dropout layer to prevent overfitting 
        self.dropout = nn.Dropout(0.5)
       
        # Fully connected layers
        self.fc1 = nn.Linear(256 * (input_height // 16) ** 2, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        # Pass the input (1@128x128) through the CNN layers. Convolution -> BatchNorm -> ReLU -> Pooling
        x = self.pool(F.relu(self.bn1(self.cnn1(x)))) #32@128x128 -> 32@64x64

        x = self.pool(F.relu(self.bn2(self.cnn2(x)))) #64@64x64 -> 64@32x32

        x = self.pool(F.relu(self.bn3(self.cnn3(x)))) #128@32x32 -> 128@16x16
      
        x = self.pool(F.relu(self.bn4(self.cnn4(x)))) #256@16x16 -> 256@8x8

        x = x.view(x.size(0), -1) # Flatten tensor to (batch_size, 256*8*8)

        x = F.relu(self.fc1(x)) # First fully connected layer with ReLU activation
        x = self.dropout(x) # Apply dropout after the first fully connected layer

        x = self.fc2(x) # Output layer (logits for each class)
        

        return x