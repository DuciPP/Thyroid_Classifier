import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, RandomSampler
from custom_dataset import CustomDataset, shuffle  # Assuming CustomDataset is your dataset class
from neural_networks.simple_model import SimpleCNN
from neural_networks.simple_model_features import SimpleCNNFeatures
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time
from matplotlib import cm


writer = SummaryWriter(log_dir=f"./logs/TE{time.time()}SSS")


model = SimpleCNNFeatures(num_classes=7)

image = Image.open("./data/images_labels/3.jpg")
            
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.RandomHorizontalFlip(),              
                                transforms.RandomRotation(30),                  
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(140),
                                transforms.ToTensor()])

batch = np.empty((32, 1, 360, 560))  # Initialize an empty array with the desired shape

# Create a batch of tensors by applying transformations to the image
for i in range(32):
    transformed_image = transform(image)
    batch[i] = transformed_image  # Add the transformed image tensor to the batch

tensor_batch = torch.Tensor(batch)


# for layer_idx, batch in enumerate(tensor_batch):
#     # Normalize the feature map for better visualization
#     fmap_grid = make_grid(batch)
#     print(type(fmap_grid))
#     print(fmap_grid.shape)

#     writer.add_image(f'FeatureMapsTrai', make_grid(tensor_batch), 0)


# print(type(tensor_batch), tensor_batch.shape)

x, feature_maps = model(tensor_batch, create_feature_maps=True)

import torch.nn.functional as F

""" # DIE 16 FEATURE CHANNELS PRO BILD ZUSAMMEN
for layer_idx, layer in enumerate(feature_maps):
    # Iterate over the individual feature maps in the batch
    for batch_idx, batch in enumerate(layer):
        # Concatenate channels along a new dimension
        merged_fmap = torch.cat([batch], dim=0)  # Concatenate along channel dimension
        merged_fmap = torch.sum(merged_fmap, dim=0, keepdim=True)  # Sum along channel dimension to get single-channel image
        
        # Normalize merged feature map to range [0, 1] for visualization
        merged_fmap_normalized = (merged_fmap - merged_fmap.min()) / (merged_fmap.max() - merged_fmap.min())
        #print(merged_fmap_normalized.shape)
        
        # Log merged feature map to TensorBoard
        writer.add_image(f'FeatureMaps/Batch_{layer_idx}/MergedFeatureMap_{batch_idx}', merged_fmap_normalized, global_step=layer_idx)

# Close SummaryWriter
writer.close() """

""" # FUNKTIONIERT SPLITTED 16 CHANNELS
for layer_idx, layer in enumerate(feature_maps):
    # Iterate over the individual feature maps in the batch
    for batch_idx, batch in enumerate(layer):
        # Iterate over each channel in the feature map
        for channel_idx in range(batch.size(0)):
            # Extract the channel
            channel = batch[channel_idx, :, :]
            
            # Normalize channel to range [0, 1] for visualization
            channel_normalized = (channel - channel.min()) / (channel.max() - channel.min())
            
            # Convert channel to image format (2D tensor)
            channel_image = channel_normalized.unsqueeze(0)  # Add batch and channel dimensions
            
            # Log channel image to TensorBoard
            writer.add_image(f'FeatureMaps/Batch_{layer_idx}/FeatureMap_{batch_idx}/Channel_{channel_idx}', channel.unsqueeze(0), global_step=layer_idx)

# Close SummaryWriter
writer.close()
 """

# HEATMAP SPLITTED 16 CHANNELS WORKS
# type:list, feature_maps.shape: [[B, C, H, W], [B, C, H, W], [B, C, H, W], [B, C, H, W], [B, C, H, W], [B, C, H, W]] len(feature_maps) => 6

for layer_idx, layer in enumerate(feature_maps):
    # Iterate over the individual feature maps in the batch
    for batch_idx, batch in enumerate(layer):
        # Iterate over each channel in the feature map
        for channel_idx in range(batch.size(0)):
            # Extract the channel
            channel = batch[channel_idx, :, :]
            
            # Normalize channel to range [0, 1] for visualization
            channel_normalized = (channel - channel.min()) / (channel.max() - channel.min())
            
            # Convert channel to heatmap using a colormap
            heatmap = cm.inferno(channel_normalized.detach().numpy())  # Apply colormap to channel
            heatmap_rgb = (heatmap[:, :, :3] * 255).astype('uint8')  # Convert to RGB format
            
            permuted_rgb = np.transpose(heatmap_rgb, (2, 0, 1))
            
            # Log heatmap image to TensorBoard
            writer.add_image(f'FeatureMaps/Batch_{layer_idx}/FeatureMap_{batch_idx}/Channel_{channel_idx}_Heatmap', permuted_rgb, global_step=layer_idx)
            print("Writing successful")

# Close SummaryWriter
writer.close()



""" # VERSUCHT GRID
# for layer_idx, layer in enumerate(feature_maps):
#     # Iterate over the individual feature maps in the batch
#     for batch_idx, batch in enumerate(layer):
#         # Normalize feature map to range [0, 1] for visualization
#         fmap_normalized = (batch - batch.min()) / (batch.max() - batch.min())
        
#         # Iterate over each channel in the feature map
#         for channel_idx in range(fmap_normalized.shape[0]):
#             # Extract the channel
#             channel = fmap_normalized[channel_idx, :, :]
            
#             # Convert channel to PIL Image
#             channel_image = Image.fromarray((channel * 255).detach().numpy().astype('uint8'))
            
#             # Log channel image to TensorBoard
#             writer.add_image(f'FeatureMaps/Batch_{layer_idx}/FeatureMap_{batch_idx}/Channel_{channel_idx}', torch.tensor(channel_image).permute(2, 0, 1), global_step=layer_idx)

# # Close SummaryWriter
# writer.close() """
