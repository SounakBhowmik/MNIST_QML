#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:41:03 2024

@author: sounakbhowmik
"""

train_data_file = "../Datasets/MNIST/mnist_train.csv"
test_data_file = "../Datasets/MNIST/mnist_test.csv"

import pandas as pd
train = pd.read_csv(test_data_file)
print(train.head())


#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Models import QConv2D_AE, Q_linear

# Hyperparameters
batch_size = 64
learning_rate = 0.001
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

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.qconv1 = QConv2D_AE(1, 4, 1, 1) # 4, 25, 25
        self.qconv2 = QConv2D_AE(4, 4, 1, 1) # 6, 9, 9
        self.qconv3 = QConv2D_AE(6, 3, 1, 1) # 6, 2, 2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.clinear = nn.Linear(6*2*2, 10)
        self.qlinear = Q_linear(10, 2)
        
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        '''
    def forward(self,x):
        x = self.pool(torch.relu(self.qconv1(x))) # 4, 12, 12
        x = self.pool(torch.relu(self.qconv2(x))) # 6, 4, 4
        x = torch.relu(self.qconv3(x))            # 6, 2, 2
        x = x.view(-1, 6 * 2 * 2)
        x = torch.relu(self.clinear(x))
        x = torch.log_softmax(x, dim=1)
        return x
        
    def forward_(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Evaluating the model
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# Train and evaluate the model
train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Test the model
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), 'mnist_cnn.pth')
#%%
# Train more
model = CNNModel()
model.load_state_dict(torch.load('mnist_cnn.pth'))
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)
# Test the model
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), 'mnist_cnn_1.pth')

#%%