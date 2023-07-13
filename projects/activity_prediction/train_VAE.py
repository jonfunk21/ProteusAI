import pandas as pd
import os
import sys
sys.path.insert(0, '../../src')
import proteusAI.io_tools as io_tools
import proteusAI.ml_tools.esm_tools as esm_tools
from models import Autoencoders
from utils import *
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Process some strings.")
parser.add_argument('--encoder', type=str, default='OHE', help='choose encoding method amino acid sequences ["OHE", "BLOSUM62", "BLOSUM50"]')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--dropout_p', type=float, default=0.0)
parser.add_argument('--hidden_layers', type=str, default='2048,1024,256', help='Comma-separated list of hidden layer sizes')
parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true', help='Save checkpoint during the process')
parser.set_defaults(save_checkpoint=False)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--step_size', type=int, default=1000, help='Stepsize for learning rate scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma value for learning rate scheduler')
args = parser.parse_args()

# Hyperparameters
epochs = args.epochs
save_checkpoints = args.save_checkpoints
dropout_p = args.dropout_p
hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
z_dim = args.z_dim
weight_decay = args.weight_decay
gamma = args.gamma
step_size = args.step_size

# encoding type ohe, BLOSUM62 or BLOSUM50
encoding_type = args.encoder
assert encoding_type in ['OHE', 'BLOSUM62', 'BLOSUM50']

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
msa_path = os.path.join(script_path, 'MSA')
alphabet = esm_tools.alphabet.to_dict()

plots_path = os.path.join(script_path, 'plots/train')
checkpoints_path = os.path.join(script_path, 'checkpoints')
os.makedirs(plots_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)

# Training and validation data paths
train_dir = os.path.join(script_path, '../data/DMS_enzymes/datasets/train')
val_dir = os.path.join(script_path, '../data/DMS_enzymes/datasets/validate')
test_dir = os.path.join(script_path, '../data/DMS_enzymes/datasets/test')

names = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith('.csv')]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256

for name in names:
    # define model name for saving
    model_name = name + f'_{encoding_type}_VAE'
    
    # Load training and validation sets
    train_df = pd.read_csv(os.path.join(train_dir, name + '.csv'))
    val_df = pd.read_csv(os.path.join(val_dir, name + '.csv'))
    
    train_dataset = VAEDataset(train_df, encoding_type=encoding_type)
    val_dataset = VAEDataset(val_df, encoding_type=encoding_type)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Assuming each sequence in the dataset is one-hot encoded and is of shape (seq_len, alphabet_size)
    seq_len, alphabet_size = train_data.dataset[0].shape
    
    # Initialize model, optimizer and epochs
    model = Autoencoders.VAE(input_dim=seq_len * alphabet_size, hidden_dims=hidden_layers, z_dim=z_dim, dropout=dropout_p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    # Train the model on the dataset
    print(f"Training {model_name} model...")
    model = train_vae(train_data, val_data, model, optimizer, criterion, scheduler, epochs, device, model_name, save_checkpoints=save_checkpoints)