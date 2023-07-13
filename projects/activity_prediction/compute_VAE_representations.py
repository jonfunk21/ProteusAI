import torch
import os
import pandas as pd
import argparse
from models import Autoencoders
from utils import VAEDataset
from torch.utils.data import DataLoader

# Argument parsing
parser = argparse.ArgumentParser(description="Process some strings.")
parser.add_argument('--encoder', type=str, default='OHE', help='choose encoding method amino acid sequences ["OHE", "BLOSUM62", "BLOSUM50"]')
parser.add_argument('--hidden_layers', type=str, default='2048,1024,256', help='Comma-separated list of hidden layer sizes')
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--dropout_p', type=float, default=0.0)
args = parser.parse_args()

# Hyperparameters
dropout_p = args.dropout_p
hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
z_dim = args.z_dim
batch_size = 256

# encoding type ohe, BLOSUM62 or BLOSUM50
encoding_type = args.encoder
assert encoding_type in ['OHE', 'BLOSUM62', 'BLOSUM50']

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(script_path, '../data/DMS_enzymes/datasets')
representation_path = os.path.join(script_path, f'representations/vae_{encoding_type}')

os.makedirs(dataset_path, exist_ok=True)
os.makedirs(representation_path, exist_ok=True)

# collect all datasets
mutant_datasets = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

# create directories for all datasets
for dataset in mutant_datasets:
    study_name = dataset.split('.')[0]
    study_path = os.path.join(representation_path, study_name)
    os.makedirs(study_path, exist_ok=True)

# compute embeddings for all datasets
for dataset in mutant_datasets:
    # study path
    study_name = dataset.split('.')[0]
    study_path = os.path.join(representation_path, study_name)
    model_name = study_name + f'_{encoding_type}_VAE'

    # get names and sequences from dataframe
    df_path = os.path.join(dataset_path, dataset)
    df = pd.read_csv(df_path)
    dataset = VAEDataset(df, encoding_type=encoding_type)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mutants = df['mutant'].to_list()

    # Assuming each sequence in the dataset is one-hot encoded and is of shape (seq_len, alphabet_size)
    seq_len, alphabet_size = data.dataset[0].shape

    # Load the pretrained weights for this study and encoding type
    model = Autoencoders.VAE(input_dim = seq_len * alphabet_size, hidden_dims=hidden_layers, z_dim=z_dim, dropout=dropout_p)
    model.load_state_dict(torch.load(f'checkpoints/{model_name}.pt'))
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(data):
            batch = batch.view(batch.size(0), -1)
            vae_representation, _, _ = model(batch)
            
            # Save the VAE representation for each sequence in the batch
            for j, rep in enumerate(vae_representation):
                mutant_name = mutants[i * batch_size + j]
                seq_rep_path = os.path.join(study_path, f'{mutant_name}.pt')
                torch.save(rep, seq_rep_path)
