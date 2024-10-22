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
    dataset = TensorDataset(torch.tensor(X.unsqueeze(1), dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Simple feedforward neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleNN, self).__init__()
        kernel_size = 4
        stride = 2
        op_channels = 10
        self.conv1 = nn.Conv2d(input_shape[1], op_channels, kernel_size, stride)
        op_d = (input_shape[2] - kernel_size) // stride  + 1
        self.fc1 = nn.Linear( op_d * op_d * op_channels,  64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        if(DEBUG):
            print(f'shape after conv1 is {x.shape}')
        x = torch.flatten(x, start_dim=1)
        if(DEBUG):
            print(f'shape after flatten is {x.shape}')
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model and evaluate it on validation set
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    logs = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        # Training loop
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)
            if(DEBUG):
                print(f'input shape={inputs.shape}')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation loop
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(val_loader, model, criterion, device)

        # Logging the epoch results
        logs['epoch'].append(epoch + 1)
        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['train_acc'].append(train_acc)
        logs['val_acc'].append(val_acc)
        logs['val_precision'].append(val_precision)
        logs['val_recall'].append(val_recall)
        logs['val_f1'].append(val_f1)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

    return logs

# Function to evaluate the model on the validation set
def evaluate_model(data_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
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

    return loss, accuracy, precision, recall, f1

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
def calculate_test_metrics(test_labels, test_predictions):
    accuracy = accuracy_score(test_labels, test_predictions) * 100
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    return {
        'Test Accuracy': accuracy,
        'Test Precision': precision,
        'Test Recall': recall,
        'Test F1 Score': f1
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
    
    
    
    
    