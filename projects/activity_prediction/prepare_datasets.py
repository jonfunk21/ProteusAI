import pandas as pd
import os
import numpy as np

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
datasets_dir = os.path.join(script_path, 'datasets')

dataset_files = os.listdir(datasets_dir)
dataset_files.sort()

dfs = [pd.read_csv(os.path.join(datasets_dir, f)) for f in dataset_files if f.endswith('.csv')]
names = [f.split('.')[0] for f in dataset_files if f.endswith('.csv')]

# Folders to save the split datasets
train_dir = os.path.join(script_path, 'datasets/train')
val_dir = os.path.join(script_path, 'datasets/validate')
test_dir = os.path.join(script_path, 'datasets/test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Random seed for reproducibility
np.random.seed(0)

for i, df in enumerate(dfs):
    # Define the dataset name
    name = names[i]

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=0)

    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(df))  # 80% for training
    val_size = int(0.1 * len(df))  # 10% for validation
    test_size = len(df) - train_size - val_size  # remaining 10% for test

    train_df = df[:train_size]
    val_df = df[train_size : train_size + val_size]
    test_df = df[train_size + val_size :]

    # Save the datasets
    train_df.to_csv(os.path.join(train_dir, name + '.csv'), index=False)
    val_df.to_csv(os.path.join(val_dir, name + '.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, name + '.csv'), index=False)
