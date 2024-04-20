import torch
import torch.nn as nn

class AdvancedCNNFeatures(nn.Module):
    def __init__(self, input_shape=(1, 360, 560), num_classes=7):
        super(AdvancedCNNFeatures, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.input_shape = input_shape
        self.features_shape = self._get_features_shape(input_shape)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.features_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _get_features_shape(self, input_shape):
        with torch.no_grad():
            X = torch.zeros((1, *input_shape))
            features = self.features(X)
            return features.view(1, -1).shape[1]
        
    def forward(self, X, create_feature_maps=False):
        feature_maps = []
        for layer in self.features:
            X = layer(X)
            if create_feature_maps and isinstance(layer, nn.MaxPool2d):
                feature_maps.append(X)
                
        X = X.view(-1, self.features_shape)
        X = self.classifier(X)

        return (X, feature_maps) if create_feature_maps else X