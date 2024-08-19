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
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=8, kernel_size=5) #8, 24, 24
        self.pool1 = nn.MaxPool2d(2, 2)  #8, 12, 12
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)#16, 8, 8
        self.pool2 = nn.MaxPool2d(2, 2)  #16, 4, 4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
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
    op = {'conv1_op':torch.empty(0), 
          'conv2_op': torch.empty(0), 
          'flattened_ip': torch.empty(0), 
          'preds': torch.empty(0)}
    for images, labels in tqdm(train_loader):
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
            
        x = classical_model.pool2(torch.relu(x)).view(-1, 16 * 4 * 4)
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


            
        
        










