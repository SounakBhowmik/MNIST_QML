# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

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