import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Calculate the shape of the features after convolution and pooling
        self.features_shape = self._get_features_shape()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.features_shape, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.features_shape)  # Flatten the feature maps
        x = self.classifier(x)
        return x

    def _get_features_shape(self):
        # Pass a random tensor through the convolutional layers to get the shape of the features
        with torch.no_grad():
            x = torch.zeros(1, 1, 360, 560)
            features = self.features(x)
            return features.view(1, -1).shape[1]