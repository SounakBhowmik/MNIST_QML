# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.datasets import mnist

def downsize_and_save_mnist(save_dir, samples_per_class_train, samples_per_class_test):
    """
    Downsize the MNIST dataset, split into train/val/test sets, and save them to files.
    
    Parameters:
        save_dir (str): The directory where the downsized datasets will be saved.
        samples_per_class_train (int): The number of samples to keep per class for the training set.
        samples_per_class_test (int): The number of samples to keep per class for the test set.
    """
    def downsize_mnist(x, y, samples_per_class):
        """
        Downsize MNIST dataset to have an equal number of samples per class.
        
        Parameters:
            x (numpy array): The input images.
            y (numpy array): The labels.
            samples_per_class (int): The number of samples to keep per class.
            
        Returns:
            x_downsized (numpy array): The downsized images.
            y_downsized (numpy array): The downsized labels.
        """
        x_downsized = []
        y_downsized = []

        for digit in range(10):
            # Get all images of the current digit
            digit_indices = np.where(y == digit)[0]
            
            # Randomly select the specified number of samples
            selected_indices = np.random.choice(digit_indices, samples_per_class, replace=False)
            
            # Append to the downsized dataset
            x_downsized.append(x[selected_indices])
            y_downsized.append(y[selected_indices])
        
        # Concatenate the results and shuffle
        x_downsized = np.concatenate(x_downsized, axis=0)
        y_downsized = np.concatenate(y_downsized, axis=0)
        
        # Shuffle the downsized dataset
        indices = np.arange(len(x_downsized))
        np.random.shuffle(indices)
        
        return x_downsized[indices], y_downsized[indices]

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Downsize training and test datasets
    x_train_small, y_train_small = downsize_mnist(x_train, y_train, samples_per_class_train)
    x_test_small, y_test_small = downsize_mnist(x_test, y_test, samples_per_class_test)

    # Split training data into training and validation sets (85% train, 15% validation)
    split_idx = int(0.85 * len(x_train_small))
    x_train_final, y_train_final = x_train_small[:split_idx], y_train_small[:split_idx]
    x_val, y_val = x_train_small[split_idx:], y_train_small[split_idx:]

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the datasets
    np.savez_compressed(os.path.join(save_dir, "train.npz"), x=x_train_final, y=y_train_final)
    np.savez_compressed(os.path.join(save_dir, "val.npz"), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(save_dir, "test.npz"), x=x_test_small, y=y_test_small)

    print(f"Datasets saved successfully in {save_dir}!")

#%%
# Example usage:
save_directory = "../Datasets/MNIST/small_MNIST"  # Replace with the desired path
samples_per_class_train = 1000  # Number of samples per class for training
samples_per_class_test = 100   # Number of samples per class for testing

downsize_and_save_mnist(save_directory, samples_per_class_train, samples_per_class_test)


#%%



def shuffle_and_split_datasets(base_dataset_dir, train_size=0.8, random_state=42):
    """
    Reads train.csv and test.csv, shuffles and splits train.csv into train and validation sets,
    shuffles test.csv, and saves them to separate CSV files.
    """
    # Step 1: Load train.csv and test.csv
    train_path = os.path.join(base_dataset_dir, 'mnist_train.csv')
    test_path = os.path.join(base_dataset_dir, 'mnist_test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Step 2: Shuffle and split train.csv into train and validation sets
    train_df_shuffled, val_df = train_test_split(train_df, test_size=1 - train_size, shuffle=True, random_state=random_state)

    # Step 3: Shuffle test.csv
    test_df_shuffled = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Step 4: Create a directory to save the split datasets if it doesn't exist
    output_dir = os.path.join(base_dataset_dir, 'split_datasets')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 5: Save the shuffled and split datasets to new CSV files
    train_df_shuffled.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df_shuffled.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Datasets have been split and saved in {output_dir} as 'train.csv', 'val.csv', and 'test.csv'.")

#%% Usage example
base_dataset_dir = '../Datasets/MNIST'
shuffle_and_split_datasets(base_dataset_dir)

#%% Dataset print
dr = pd.read_csv(os.path.join('Datasets/MNIST/split_datasets/train.csv'))
print(dr.head())