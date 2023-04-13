import sys
sys.path.append('../')
import pandas as pd
import torch
from activity_predictor import FFNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.pytorchtools import CustomDataset, train
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Hyper parameters (later all argparse)
epochs = 10
batch_size = 26
rep_layer = 33
input_size = 147 * 1280
hidden_layers = [1280]
output_size = 1
patience = 10
save_path = 'checkpoints'
data_path = '../example_data/directed_evolution/GB1/GB1.csv'
train_log='train_log'
n_folds = 5

if not os.path.exists(save_path):
    os.mkdir(save_path)

# pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data and split into train and val. first naive split, normalize activity
df = pd.read_csv(data_path)
df['Data_normalized'] = (df['Data'] - df['Data'].min()) / (df['Data'].max() - df['Data'].min())

# Perform clustering based on sequence similarity (using Hamming distance)
def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

# Create a distance matrix using the Hamming distance
distance_matrix = pd.DataFrame([[hamming_distance(seq1, seq2) for seq1 in df['Sequence']] for seq2 in df['Sequence']])

# Perform Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=5)
df['Cluster'] = cluster.fit_predict(distance_matrix)

# Create stratified splits
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize variables to store metrics for each fold
all_train_losses = []
all_val_losses = []
all_val_rmse = []
all_val_pearson = []

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Cluster'])):
    print(f"Fold {fold + 1}")

    # Create train and validation DataFrames for the current fold
    train_df = df.iloc[train_idx].drop(columns=['Cluster'])
    val_df = df.iloc[val_idx].drop(columns=['Cluster'])

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Dataloaders
    train_set = CustomDataset(train_df)
    val_set = CustomDataset(val_df)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Load activity predictor model
    model = FFNN(input_size, output_size, hidden_layers)
    model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model and store the metrics for the current fold
    with open(os.path.join(save_path, train_log), 'w') as f:
        print(f'{n_folds}-fold cross validation:', file=f)

    fold_train_losses, fold_val_losses, fold_val_rmse, fold_val_pearson = train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, patience, save_path, fold, train_log)

    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)
    all_val_rmse.append(fold_val_rmse)
    all_val_pearson.append(fold_val_pearson)

# Calculate average metrics across all folds
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)
avg_val_rmse = np.mean(all_val_rmse, axis=0)
avg_val_pearson = np.mean(all_val_pearson, axis=0)
with open('results', 'w') as f:
    print('avg_train_losses:', avg_val_losses, file=f)
    print('avg_val_losses:', avg_val_losses, file=f)
    print('avg_val_rmse:', avg_val_rmse, file=f)
    print('avg_val_pearson:', avg_val_pearson, file=f)
