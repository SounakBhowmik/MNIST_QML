import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os
from train import load_data, get_dataloader, SimpleNN, train_model, evaluate_model, get_test_predictions, save_logs_to_excel, calculate_test_metrics, save_test_metrics_to_excel, data_reshape         
import numpy as np
from data_visualize import visualize
from QNNModels import QCONV_011

DEBUG = False

train_set_path = '../Datasets/MNIST/small_MNIST/train.npz'
val_set_path = '../Datasets/MNIST/small_MNIST/val.npz'
test_set_path = '../Datasets/MNIST/small_MNIST/test.npz'

model_name = "QCONV_011"


MAX_X_VAL = 255.0
BATCH_SIZE = 25

# Main function to execute the entire workflow
def main():
    # Paths to your dataset
    output_dir = 'results'

    # Load data
    X_train, y_train = load_data(os.path.join(train_set_path))
    
    X_val, y_val = load_data(os.path.join(val_set_path))
    X_test, y_test = load_data(os.path.join(test_set_path))
    
    
    # Pull the values within [0,1]
    X_train = X_train/MAX_X_VAL
    X_val = X_val/MAX_X_VAL
    X_test = X_test/MAX_X_VAL

    '''
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    '''
    
    # Create DataLoaders
    train_loader = get_dataloader(X_train, y_train, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(X_val, y_val, batch_size=BATCH_SIZE)
    test_loader = get_dataloader(X_test, y_test, batch_size=BATCH_SIZE)

    # Device configuration (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model, loss function, and optimizer
    data_iter = iter(train_loader)  
    data_batch, labels_batch = next(data_iter)
    input_shape = data_batch.shape
    visualize(data_batch[10])
    return 
    
    
    num_classes = len(set(y_train))
    #model = SimpleNN(input_shape, num_classes).to(device)
    model = QCONV_011()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model and log the training process
    num_epochs = 20
    logs = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device)

    # Get test predictions
    test_predictions = get_test_predictions(test_loader, model, device)

    # Save logs and test predictions to an Excel file
    save_logs_to_excel(logs, test_predictions, output_dir)
    
    # Calculate test metrics using test predictions and true labels
    test_metrics = calculate_test_metrics(y_test, test_predictions)
    
    # Save test metrics (accuracy, precision, recall, f1) to a separate Excel file
    save_test_metrics_to_excel(test_metrics, output_dir, model_name)

if __name__ == '__main__':
    main()
    