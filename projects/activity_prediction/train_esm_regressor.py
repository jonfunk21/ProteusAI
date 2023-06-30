import pandas as pd
import os
import sys
sys.path.insert(0, '../../src')
#import proteusAI.io_tools as io_tools
from models import Regressors
from utils import *
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import argparse

# args
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--model', type=str, default='esm1v', help='Choose model either esm2 or esm1v')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

# model for embedding computation esm1v or esm2
model = args.model

batch_size = args.batch_size
epochs = args.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
representations_path = os.path.join(script_path, f'representations/{esm_model}')
datasets_dir = os.path.join(script_path, 'datasets')

plots_path = os.path.join(script_path, 'plots/train')
checkpoints_path = os.path.join(script_path, 'checkpoints')
os.makedirs(plots_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)

dataset_files = os.listdir(datasets_dir)
dataset_files.sort()

dfs = [pd.read_csv(os.path.join(datasets_dir, f)) for f in dataset_files if f.endswith('.csv')]
names = [f.split('.')[0] for f in dataset_files if f.endswith('.csv')]
representations_paths = [os.path.join(representations_path, name) for name in names]
datasets = []

for i, df in enumerate(dfs):
    dat = RegDataset(df, representations_paths[i])
    datasets.append(dat)

for i, dat in enumerate(datasets):
    # define model name for saving
    model_name = names[i] + f'_{esm_model}_regressor'

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dat))  # 80% for training
    val_size = len(dat) - train_size
    train_dataset, val_dataset = random_split(dat, [train_size, val_size])
    
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # embedding dimension
    if esm_model == 'esm1v':
        model_dim = 1280
    
    # Initialize model, optimizer and epochs
    model = Regressors.FFNN(input_dim=model_dim, hidden_layers=[1000, 1000], output_dim=1, dropout_p=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)
    criterion = nn.MSELoss()

    # Train the model on the dataset
    print(f"Training {model_name} model...")
    model = train_regression(train_data, val_data, model, optimizer, criterion, scheduler, epochs, device, model_name)