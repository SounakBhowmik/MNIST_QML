# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.datasets import mnist

#%%
def prepare_and_save_mnist(save_dir):
    """
    Load the MNIST dataset, shuffle and split the train set into train and val (85:15 ratio),
    shuffle the test set, and save all datasets in a specified location.
    
    Parameters:
        save_dir (str): The directory where the shuffled datasets will be saved.
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Split train set into train and val (85:15)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=42, shuffle=True)

    # Shuffle test set
    test_indices = np.arange(len(x_test))
    np.random.shuffle(test_indices)
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the datasets
    np.savez_compressed(os.path.join(save_dir, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(save_dir, "val.npz"), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(save_dir, "test.npz"), x=x_test, y=y_test)

    print(f"Datasets saved successfully in {save_dir}")

# Example usage:
save_directory = "../Datasets/MNIST/multi_class"  # Replace with your desired directory
prepare_and_save_mnist(save_directory)




#%% One vs all binary set prepare

def create_one_vs_all_mnist(save_dir):
    """
    Create binary classification datasets from MNIST where each dataset corresponds to
    one class vs. all other classes. Saves each dataset in separate folders: bin_0, bin_1, ..., bin_9.
    Each folder will contain train, val, and test datasets.
    
    Parameters:
        save_dir (str): The directory where the binary classification datasets will be saved.
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Iterate over each class (0 to 9)
    for target_class in range(10):
        # Create a directory name like bin_0, bin_1, ..., bin_9
        folder_name = f"bin_{target_class}"
        folder_path = os.path.join(save_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Filter training data for the target class and other classes
        target_train_indices = np.where(y_train == target_class)[0]
        other_train_indices = np.where(y_train != target_class)[0]

        # Balance the binary classes by sampling an equal number of "other" class samples
        num_target_samples = len(target_train_indices)
        other_train_indices = np.random.choice(other_train_indices, num_target_samples, replace=False)

        # Prepare training data and labels
        x_train_binary = np.concatenate((x_train[target_train_indices], x_train[other_train_indices]), axis=0)
        y_train_binary = np.concatenate((np.ones(num_target_samples), np.zeros(num_target_samples)), axis=0)

        # Shuffle and split the train dataset into train and val (85:15)
        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
            x_train_binary, y_train_binary, test_size=0.15, random_state=42, shuffle=True)

        # Filter test data for the target class and other classes
        target_test_indices = np.where(y_test == target_class)[0]
        other_test_indices = np.where(y_test != target_class)[0]

        # Balance the binary classes by sampling an equal number of "other" class samples
        num_target_samples_test = len(target_test_indices)
        other_test_indices = np.random.choice(other_test_indices, num_target_samples_test, replace=False)

        # Prepare test data and labels
        x_test_binary = np.concatenate((x_test[target_test_indices], x_test[other_test_indices]), axis=0)
        y_test_binary = np.concatenate((np.ones(num_target_samples_test), np.zeros(num_target_samples_test)), axis=0)

        # Shuffle the test set
        test_indices = np.arange(len(x_test_binary))
        np.random.shuffle(test_indices)
        x_test_binary = x_test_binary[test_indices]
        y_test_binary = y_test_binary[test_indices]

        # Save the datasets in the appropriate folder
        np.savez_compressed(os.path.join(folder_path, "train.npz"), x=x_train_split, y=y_train_split)
        np.savez_compressed(os.path.join(folder_path, "val.npz"), x=x_val_split, y=y_val_split)
        np.savez_compressed(os.path.join(folder_path, "test.npz"), x=x_test_binary, y=y_test_binary)

        print(f"Saved binary dataset for class {target_class} vs all in {folder_path}")

# Example usage:
save_directory = "../Datasets/MNIST/binary_one_vs_all/"  # Replace with your desired directory
create_one_vs_all_mnist(save_directory)





#%% Binary dataset preparation
from itertools import combinations


def create_binary_classification_mnist(save_dir):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Iterate over all combinations of 10 classes (10C2)
    for (class1, class2) in combinations(range(10), 2):
        # Create a directory name like bin_01, bin_02, ..., bin_89
        folder_name = f"bin_{class1}{class2}"
        folder_path = os.path.join(save_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Filter training data for the two classes
        train_indices = np.where((y_train == class1) | (y_train == class2))[0]
        x_train_binary = x_train[train_indices]
        y_train_binary = y_train[train_indices]
        
        # Convert labels to binary (0 or 1)
        y_train_binary = np.where(y_train_binary == class1, 0, 1)
        
        # Shuffle and split the train dataset into train and val (85:15)
        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
            x_train_binary, y_train_binary, test_size=0.15, random_state=42, shuffle=True)
        
        # Filter test data for the two classes
        test_indices = np.where((y_test == class1) | (y_test == class2))[0]
        x_test_binary = x_test[test_indices]
        y_test_binary = y_test[test_indices]
        
        # Convert test labels to binary (0 or 1)
        y_test_binary = np.where(y_test_binary == class1, 0, 1)
        
        # Shuffle the test set
        test_indices = np.arange(len(x_test_binary))
        np.random.shuffle(test_indices)
        x_test_binary = x_test_binary[test_indices]
        y_test_binary = y_test_binary[test_indices]
        
        # Save the datasets in the appropriate folder
        np.savez_compressed(os.path.join(folder_path, "train.npz"), x=x_train_split, y=y_train_split)
        np.savez_compressed(os.path.join(folder_path, "val.npz"), x=x_val_split, y=y_val_split)
        np.savez_compressed(os.path.join(folder_path, "test.npz"), x=x_test_binary, y=y_test_binary)
        
        print(f"Saved binary dataset for classes {class1} and {class2} in {folder_path}")

# Example usage:
save_directory = "../Datasets/MNIST/binary/"  # Replace with your desired directory
create_binary_classification_mnist(save_directory)



#%%
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
save_directory = "../Datasets/MNIST/small_MNIST_50000"  # Replace with the desired path
samples_per_class_train = 5000  # Number of samples per class for training
samples_per_class_test = 1000   # Number of samples per class for testing

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