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
parser.add_argument('--encoder', type=str, default='OHE', help='Choose encoder model either OHE, BLOSUM50 or BLOSUM62')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--dropout_p', type=float, default=0.1)
parser.add_argument('--hidden_layers', type=str, default='2048,1024,256', help='Comma-separated list of hidden layer sizes')
parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true', help='Save checkpoint during the process')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--step_size', type=int, default=1000, help='Stepsize for learning rate scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma value for learning rate scheduler')
parser.add_argument('--z_dim', type=int, default=64)
parser.set_defaults(save_checkpoint=False)
args = parser.parse_args()

# model for embedding computation esm1v or esm2
encoding_type = args.encoder
hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
z_dim = args.z_dim
gamma = args.gamma
step_size = args.step_size
weight_decay = args.weight_decay
dropout_p = args.dropout_p
save_checkpoints = args.save_checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
representations_path = os.path.join(script_path, f'representations/{encoding_type}')

train_plots_path = os.path.join(script_path, 'plots/train')
results_plots_path = os.path.join(script_path, 'plots/results')
checkpoints_path = os.path.join(script_path, 'checkpoints')
results_path = os.path.join(script_path, 'results')
os.makedirs(train_plots_path, exist_ok=True)
os.makedirs(results_plots_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# Training, validation, and test data paths
train_dir = os.path.join(script_path, '../data/DMS_enzymes/datasets/train')
val_dir = os.path.join(script_path, '../data/DMS_enzymes/datasets/validate')
test_dir = os.path.join(script_path, '../data/DMS_enzymes/datasets/test')

names = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith('.csv')]

for name in names:
    # define model name for saving
    model_name = name + f'_FFNN_{encoding_type}'
    VAE_name = name + f'_{encoding_type}_VAE'
    
    # Load training, validation, and test sets
    train_df = pd.read_csv(os.path.join(train_dir, name + '.csv'))
    val_df = pd.read_csv(os.path.join(val_dir, name + '.csv'))
    test_df = pd.read_csv(os.path.join(test_dir, name + '.csv'))
    
    train_dataset = FFNNDataset2(train_df, encoding_type=encoding_type)
    val_dataset = FFNNDataset2(val_df, encoding_type=encoding_type)
    test_dataset = FFNNDataset2(test_df, encoding_type=encoding_type)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # deterimine input size
    vae_dim = len(train_data.dataset[0][0])

    # load VAE
    vae = Autoencoders.VAE(input_dim = vae_dim, hidden_dims=hidden_layers, z_dim=z_dim, dropout=dropout_p)
    vae.load_state_dict(torch.load(f'checkpoints/{VAE_name}.pt', map_location='cpu'))
    vae.eval()

    # Initialize model, optimizer and epochs
    model = Regressors.FFNN(input_dim=z_dim, hidden_layers=hidden_layers, output_dim=1, dropout_p=dropout_p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()

    # Train the model on the dataset
    print(f"Training {model_name} model...")
    model = train_regression(train_data, val_data, model, optimizer, criterion, scheduler, epochs, device, model_name, save_checkpoints=save_checkpoints, vae=vae)

    # Plot predictions against ground truth for test data
    predictions = plot_predictions_vs_groundtruth(test_data, model, device, fname=f'{results_plots_path}/{model_name}_pred_vs_true.png')

    # Append predictions to test_df and save
    test_df['predictions'] = predictions
    test_df.to_csv(f'results/{model_name}_predictions.csv', index=False)