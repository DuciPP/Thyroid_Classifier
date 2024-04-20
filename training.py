import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomDataset, shuffle
import torch.nn as nn
from neural_networks.simple_model import SimpleCNN
from neural_networks.simple_model_features import SimpleCNNFeatures
from neural_networks.advanced_model_features import AdvancedCNNFeatures
from train_logger import log_all_feature_maps_with_original

writer = SummaryWriter(f'./logs/training_loop_advanced_1')

dataset = shuffle(CustomDataset(data="./data/path_list/data.csv"))

# Define the sizes of the train and test subsets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the shuffled dataset into train and test subsets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Instantiate your model
model = AdvancedCNNFeatures()
writer.add_graph(model, torch.randn(1, 1, 360, 560))
model.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

EPOCHS = 100
BATCH_SIZE = 32

# Create DataLoader objects for the training and testing sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


for epoch in range(EPOCHS):
    
    model.train()
    train_loss = 0
    train_acc = 0
    
    for batchidx, (X, y) in enumerate(train_loader):
        last_batch_train_X = X
        
        X, y = X.to(device), y.to(device)
        
        y_pred, feature_maps_train = model(X, create_feature_maps=True)
        
        loss = loss_fn(y_pred, y)
        
        train_loss += loss
        
        right = torch.argmax(y_pred, dim=1) == y
        n_right = len(right[right])
        train_acc += n_right
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    train_loss /= train_size
    train_acc /= train_size
    
    
    model.eval()
    test_loss = 0
    test_acc = 0
    
    with torch.inference_mode():
        
        for batchidx, (X, y) in enumerate(test_loader):
            last_batch_test_X = X
            
            X, y = X.to(device), y.to(device)
            
            y_pred, feature_maps_test = model(X, create_feature_maps=True)
            
            loss = loss_fn(y_pred, y)
            
            test_loss += loss
            
            right = torch.argmax(y_pred, dim=1) == y
            n_right = len(right[right])
            test_acc += n_right
            
        test_loss /= test_size
        test_acc /= test_size
        
        
        print(f"Epoch: {epoch+1} | Training loss: {train_loss} | Training accuracy: {train_acc*100:.2f}%")
        print(f"Epoch: {epoch+1} | Test loss: {test_loss} | Test accuracy: {test_acc*100:.2f}%")


        writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)
        writer.add_scalars('Learning Rate', {'Change': optimizer.param_groups[0]['lr']}, epoch)
        log_all_feature_maps_with_original(writer=writer, epoch=epoch, feature_maps=feature_maps_train, original_batches=last_batch_train_X, n_batches=1, set="Train")
        log_all_feature_maps_with_original(writer=writer, epoch=epoch, feature_maps=feature_maps_test, original_batches=last_batch_test_X, n_batches=1, set="Test")
        
        
writer.close()