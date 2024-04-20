import torch
import torch.nn as nn
import numpy as np

class SimpleCNNFeatures(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNNFeatures, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.features_shape = self._get_features_shape()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.features_shape, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )



    def _get_features_shape(self):
        with torch.no_grad():
            X = torch.zeros(1, 1, 360, 560)
            features = self.features(X)
            return features.view(1, -1).shape[1]
        
        
    def forward(self, X, create_feature_maps=False):
        
        if create_feature_maps:
            feature_maps = []
    
        for layer in self.features:
            X = layer(X)
            
            if create_feature_maps:
                
                if isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                    feature_maps.append(X)
                
            
        X = X.view(-1, self.features_shape)
        
        X = self.classifier(X)

        if create_feature_maps:
            return X, feature_maps
        
        return X