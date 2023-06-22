import pandas as pd
import os
import sys
sys.path.insert(0, '../../src')
import proteusAI.io_tools as io_tools
import proteusAI.ml_tools.esm_tools as esm_tools
import proteusAI.ml_tools.torch_tools as torch_tools
from models import Autoencoders
from utils import *
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Process some strings.")
parser.add_argument('--encoder', type=str, default='OHE', help='choose encoding method amino acid sequences ["OHE", "BLOSUM62", "BLOSUM50"]')
parser.add_argument('--epochs', type=int, default=1000, help='number or epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
args = parser.parse_args()

epochs = 1000

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

msa_results = io_tools.load_all_fastas(msa_path)

encodings = {}
for key, value in msa_results.items():
    sequences = value[1]
    if 'BLOSUM' in encoding_type:
        e = torch_tools.blosum_encoding(sequences, matrix=encoding_type)
    elif encoding_type == 'OHE':
        e = torch_tools.one_hot_encoder(sequences, alphabet)

    encoded_sequences = [encoding for encoding in e]
    encodings[key] = pd.DataFrame({
        'label':value[0], 
        'x':encoded_sequences
    })

names = []
datasets = []
for key in encodings.keys():
    names.append(str(key)[:-6])
    datasets.append(VAEDataset(encodings[key], encoding_type=encoding_type))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256

for i, dat in enumerate(datasets):
    # define model name for saving
    model_name = names[i] + f'_{encoding_type}_VAE'
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dat))  # 80% for training
    val_size = len(dat) - train_size
    train_dataset, val_dataset = random_split(dat, [train_size, val_size])
    
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Assuming each sequence in the dataset is one-hot encoded and is of shape (seq_len, alphabet_size)
    seq_len, alphabet_size = train_data.dataset[0].shape
    
    # Initialize model, optimizer and epochs
    model = Autoencoders.VAE(input_dim=seq_len * alphabet_size, hidden_dims=[2048, 1024, 256], z_dim=64, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    # Train the model on the dataset
    print(f"Training {model_name} model...")
    model = train_vae(train_data, val_data, model, optimizer, criterion, scheduler, epochs, device, model_name)