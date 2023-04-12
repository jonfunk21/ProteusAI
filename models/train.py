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

# Hyper parameters (later all argparse)
epochs = 100
batch_size = 26
rep_layer = 33
input_size = 147 * 1280
hidden_layers = [1280]
output_size = 1
patience = 10
save_path = 'checkpoints'
data_path = '../example_data/directed_evolution/GB1/GB1.csv'

if not os.path.exists(save_path):
    os.mkdir(save_path)

# load data and split into train and val. first naive split, normalize activity
df = pd.read_csv(data_path)
df['Data_normalized'] = (df['Data'] - df['Data'].min()) / (df['Data'].max() - df['Data'].min())

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(index=train_df.index)

# reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloaders
train_set = CustomDataset(train_df)
val_set = CustomDataset(val_df)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# load activity predictor model
model = FFNN(input_size, output_size, hidden_layers)
model.to(device)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters)

activity_model = train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, patience, save_path)
