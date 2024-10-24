# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F

DEBUG = False

DATA_SHAPE = (1,28,28)
DATA_RESHAPE = (17, 17)

# Function to reshape the features
def data_reshape(X, shape):
    return X.reshape((-1, )+ shape)




# Function to load data from CSV and preprocess it
def load_data(npz_path):
    '''
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values  # Assuming 'label' is the target column
    y = LabelEncoder().fit_transform(df['label'].values)
    return X, y
    '''
    data = np.load(npz_path)
    return data['x'], data['y']

# Function to get DataLoader from the dataset
def get_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class BinaryMNISTClassifier(nn.Module):
    def __init__(self):
        super(BinaryMNISTClassifier, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
        
        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers with batch norm, ReLU, and dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.log_softmax(self.fc2(x), 1)  # Output layer with sigmoid activation
        
        return x
    
    
    
# Simple feedforward neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 4, 2) # 8,13,13
        
        
        self.conv2 = nn.Conv2d(8, 16, 8, 2) # 16,3,3
        
        #op_d = (input_shape[2] - kernel_size) // stride  + 1
        self.fc1 = nn.Linear(16 * 3 * 3,  64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        #self.fc2 = nn.Linear(32, num_classes)
        #self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        if(DEBUG):
            print(f'shape after conv1 is {x.shape}')
        x = torch.flatten(x, start_dim=1)
        if(DEBUG):
            print(f'shape after flatten is {x.shape}')
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.log_softmax(self.fc2(x), dim=1)
        #x = torch.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

# Function to train the model and evaluate it on validation set
def train_model(train_loader, val_loader, test_loader, model, criterion, optimizer, num_epochs, device, num_classes):
    logs = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        # Training loop
        for inputs, labels in tqdm(train_loader):
            #inputs, labels = inputs.to(device), labels.to(device) # Turn on while using GPU
            if(DEBUG):
                print(f'input shape={inputs.shape}, labels shape = {labels.shape}')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.eye(num_classes)[labels.int()])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            if(DEBUG):
                print(f'predictions = {predicted}')
                #break
            total += labels.size(0)
            if(DEBUG):
                print(f'total += {labels.size(0)}')
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation loop
        _, val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(val_loader, model, criterion, device, num_classes)

        # Logging the epoch results
        logs['epoch'].append(epoch + 1)
        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['train_acc'].append(train_acc)
        logs['val_acc'].append(val_acc)
        logs['val_precision'].append(val_precision)
        logs['val_recall'].append(val_recall)
        logs['val_f1'].append(val_f1)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}')
        #_, test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(test_loader, model, criterion, device, num_classes)

        #print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        #test_predictions = get_test_predictions(test_loader, model, device)
        #test_metrics = calculate_test_metrics(y_test, test_predictions)
        
    return logs

# Function to evaluate the model on the validation set
def evaluate_model(data_loader, model, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, torch.eye(num_classes)[labels.int()])
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return all_preds, loss, accuracy, precision, recall, f1

# Function to get test predictions
def get_test_predictions(test_loader, model, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:  # Labels are not required for test set
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    return all_preds

# Function to save logs and predictions to an Excel file
def save_logs_to_excel(logs, test_predictions, output_dir):
    # Convert logs to a DataFrame
    df_logs = pd.DataFrame(logs)
    
    # Save the logs and test predictions to an Excel file
    excel_path = os.path.join(output_dir, 'training_logs_and_predictions.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        df_logs.to_excel(writer, sheet_name='Training Logs', index=False)
        pd.DataFrame({'Test Predictions': test_predictions}).to_excel(writer, sheet_name='Test Predictions', index=False)

    print(f"Logs and test predictions saved to {excel_path}.")

# Function to calculate test metrics (accuracy, precision, recall, f1) using test predictions
def calculate_test_metrics(test_loader,  model, criterion, device, num_classes):
    
    test_predictions, test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(test_loader, model, criterion, device, num_classes)
    

    print(f'test_accuracy: {test_acc}')
    return test_predictions, {
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1 Score': test_f1
    }



# Function to save test metrics (accuracy, precision, recall, f1) to an Excel file
def save_test_metrics_to_excel(test_metrics, output_dir, model_name):
    # Add the model name to the test metrics dictionary
    test_metrics['Model Name'] = model_name

    # Create a DataFrame for the test metrics
    df_test_metrics = pd.DataFrame([test_metrics])

    # Define the path for the test results file
    excel_path = os.path.join(output_dir, 'test_results.xlsx')

    # Check if the file already exists
    if os.path.exists(excel_path):
        # Load the existing sheet to check if it's empty
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            sheet = writer.sheets.get('Test Metrics')
            if sheet is not None:
                # Get the current number of rows
                startrow = sheet.max_row
            else:
                # If the sheet does not exist, start from the first row
                startrow = 0

            # Write the DataFrame to the Excel file, add header only if writing to the first row
            df_test_metrics.to_excel(writer, sheet_name='Test Metrics', index=False, header=startrow == 0, startrow=startrow)

    else:
        # If the file doesn't exist, create it and write the DataFrame with headers
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            df_test_metrics.to_excel(writer, sheet_name='Test Metrics', index=False)

    print(f"Test metrics saved to {excel_path} for model '{model_name}'.")
    
    
    
    
    