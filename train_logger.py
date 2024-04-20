import numpy as np
from matplotlib import cm


def log_all_feature_maps_with_original(writer, epoch, feature_maps, original_batches, n_batches=1, set=""):
    
    for original_batch_idx, original_sample in enumerate(original_batches[:n_batches]):
        
        original_normalized = (original_sample - original_sample.min()) / (original_sample.max() - original_sample.min())
        writer.add_image(tag=f'Epoch_{epoch}_Original/{set}', img_tensor=original_normalized, global_step=original_batch_idx)
        
        # Iterate through every neural network layer output
        for layer_idx, layer in enumerate(feature_maps):
            
            # To see every single channel heatmap
            for channel_idx in range(layer.size(1)):
                # Extract the channel
                channel = layer[original_batch_idx, channel_idx, :, :]
                
                # Normalize channel to range [0, 1] for visualization
                channel_normalized = (channel - channel.min()) / (channel.max() - channel.min())
                
                # Convert channel to heatmap using a colormap
                heatmap = cm.inferno(channel_normalized.detach().numpy())  # Apply colormap to channel
                heatmap_rgb = (heatmap[:, :, :3] * 255).astype('uint8')  # Convert to RGB format
                
                permuted_rgb = np.transpose(heatmap_rgb, (2, 0, 1))
                
                # Log heatmap image to TensorBoard
                writer.add_image(tag=f'Epoch_{epoch}_FeatureMap/{set}/MaxPool2d_{layer_idx}/Channel_{channel_idx}_Heatmap', img_tensor=permuted_rgb, global_step=layer_idx)

    # Close SummaryWriter
    # writer.close()