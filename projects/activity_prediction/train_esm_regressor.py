import pandas as pd
import os
import sys
sys.path.insert(0, '../../src')
from models import Regressors
from utils import *
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse

# args
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--model', type=str, default='esm1v', help='Choose model either esm2 or esm1v')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true', help='Save checkpoint during the process')
parser.set_defaults(save_checkpoint=False)
args = parser.parse_args()

# model for embedding computation esm1v or esm2
esm_model = args.model

batch_size = args.batch_size
epochs = args.epochs
save_checkpoint = args.save_checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
representations_path = os.path.join(script_path, f'representations/{esm_model}')

train_plots_path = os.path.join(script_path, 'plots/train')
results_plots_path = os.path.join(script_path, 'plots/results')
checkpoints_path = os.path.join(script_path, 'checkpoints')
os.makedirs(train_plots_path, exist_ok=True)
os.makedirs(results_plots_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)

# Training, validation, and test data paths
train_dir = os.path.join(script_path, 'datasets/train')
val_dir = os.path.join(script_path, 'datasets/validate')
test_dir = os.path.join(script_path, 'datasets/test')

names = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith('.csv')]

for name in names:
    # define model name for saving
    model_name = name + f'_{esm_model}_regressor'
    
    # Load training, validation, and test sets
    train_df = pd.read_csv(os.path.join(train_dir, name + '.csv'))
    val_df = pd.read_csv(os.path.join(val_dir, name + '.csv'))
    test_df = pd.read_csv(os.path.join(test_dir, name + '.csv'))
    
    train_dataset = RegDataset(train_df, os.path.join(representations_path, name))
    val_dataset = RegDataset(val_df, os.path.join(representations_path, name))
    test_dataset = RegDataset(test_df, os.path.join(representations_path, name))

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    model = train_regression(train_data, val_data, model, optimizer, criterion, scheduler, epochs, device, model_name, save_checkpoint=save_checkpoint)

    # Plot predictions against ground truth for test data
    plot_predictions_vs_groundtruth(test_data, model, device, fname=f'{results_plots_path}/{name}_pred_vs_true.png')