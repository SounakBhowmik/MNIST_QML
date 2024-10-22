#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:41:03 2024

@author: sounakbhowmik
"""
train_data_file = "../Datasets/MNIST/mnist_train.csv"
test_data_file = "../Datasets/MNIST/mnist_test.csv"
#----------------------------------------------------------------------------------------------------
import torch
from torchsummary import summary
import torch.nn as nn
import torch
import torch.nn as nn
#import torch.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Models import QConv2D_AE, Q_linear
#%%
# Hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 100

# Load the dataset
def load_mnist_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values.astype(np.float32)
    return images, labels

# Data preparation
class MNISTDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data
train_images, train_labels = load_mnist_from_csv(train_data_file)
test_images, test_labels = load_mnist_from_csv(train_data_file)

# Split train data into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = MNISTDataset(train_images, train_labels, transform=transform)
val_dataset = MNISTDataset(val_images, val_labels, transform=transform)
test_dataset = MNISTDataset(test_images, test_labels, transform=transform)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=8, kernel_size = 2) #8, 27, 27
        self.pool1 = nn.MaxPool2d(2, 2)  #8, 13, 13
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=14, kernel_size = 4, stride = 2) #15, 5, 5
        self.pool2 = nn.MaxPool2d(2, 1)  #16, 4, 4
        self.fc1 = nn.Linear(14 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 14 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)
        return x

#%%
classical_model = CNN()
classical_model.load_state_dict(torch.load('mnist_cnn.pth'))
classical_model.eval()

#print(summary(classical_model, (1,28,28)))

#%%
'''
In this section we shall develop a function that will take,
    1) the classical model
    2) dataloaders
as INPUT
It will output,
    1) A dict
        --> {'layer_i_op': tensor} for i in n_layers
'''

from tqdm import tqdm

def get_layerwise_op(train_loader, classical_model):
    op = {'input': torch.empty(0), 
          'conv1_op':torch.empty(0), 
          'conv2_op': torch.empty(0), 
          'flattened_ip': torch.empty(0), 
          'preds': torch.empty(0)}
    for images, labels in tqdm(train_loader):
        if(len(op['input']) == 0):
            op['input'] = images
        else:
            op['input'] = torch.cat((op['input'], images), dim=0)
            
            
        x = classical_model.conv1(images)
        
        
        if(len(op['conv1_op']) == 0):
            op['conv1_op'] = x
        else:
            op['conv1_op'] = torch.cat((op['conv1_op'], x), dim=0)
            
            
        x = classical_model.conv2(classical_model.pool1(torch.relu(x)))
        
        
        if(len(op['conv2_op']) == 0):
            op['conv2_op'] = x
        else:
            op['conv2_op'] = torch.cat((op['conv2_op'], x), dim=0)
            
            
        x = classical_model.pool2(torch.relu(x)).view(-1, 14 * 4 * 4)
        
        
        if(len(op['flattened_ip']) == 0):
            op['flattened_ip'] = x
        else:
            op['flattened_ip'] = torch.cat((op['flattened_ip'], x), dim=0)
            
        
        if(len(op['preds']) == 0):
            op['preds'] = labels
        else:
            op['preds'] = torch.cat((op['preds'], labels), dim=0)
            
    return op

output_dir = get_layerwise_op(train_loader, classical_model)

#%% Define the modules of the quantum convolutional model


from Models import QConv2D_MF
qc1 = QConv2D_MF(1, 2, 2, 1, 4)
qc2 = QConv2D_MF(8, 4, 2, 2, 2)


#%%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the loss function and optimizer globally
loss_function = nn.MSELoss()
optimizer = None  # Will be initialized with the model parameters later

def train_model(model, X, y, num_iterations=50, learning_rate=0.0001, batch_size=32):
    """
    Trains a PyTorch model using MSE loss and Adam optimizer with batch processing.

    Parameters:
    model (torch.nn.Module): The model to train.
    X (torch.Tensor): Input features.
    y (torch.Tensor): Target values.
    num_iterations (int): Number of training iterations.
    learning_rate (float): Learning rate for the optimizer.
    batch_size (int): Size of each batch during training.

    Returns:
    list: A list of loss values during training.
    """

    global optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a DataLoader for batch processing
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # List to store loss values during training
    loss_values = []

    # Training loop
    for i in range(num_iterations):
        print(f'Iteration number #{i}')
        running_loss = 0.0
        for batch_X, batch_y in tqdm(dataloader):
            # Zero the gradients before running the backward pass
            optimizer.zero_grad()

            # Forward pass: Compute predicted y by passing batch_X to the model
            y_pred = model(batch_X)

            # Compute and print loss
            loss = loss_function(y_pred, batch_y)
            running_loss += loss.item()

            # Backward pass: Compute gradient of the loss with respect to all the learnable parameters
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        loss_values.append(avg_loss)

        # Print the loss 
        print(f"Iteration {i+1}/{num_iterations}, Loss: {avg_loss}")

    return loss_values

# Example usage:
# Assuming `X` and `y` are torch Tensors and `model` is an instance of a torch.nn.Module subclass
# X = torch.randn(100, 10)  # Example input
# y = torch.randn(100, 1)   # Example target
# model = YourModel()       # Replace with your actual model

model = qc1
X = output_dir['input'].clone().detach()
y = output_dir['conv1_op'].clone().detach()

train_model(model, X, y, num_iterations=50, learning_rate=0.0001, batch_size=16)
