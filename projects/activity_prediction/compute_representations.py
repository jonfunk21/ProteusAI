import os
import pandas as pd
import sys
sys.path.insert(0, '../../src')
from proteusAI.ml_tools.esm_tools.esm_tools import *
import argparse

# Hyper parameters (later all argparse)
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--model', type=str, default='esm1v', help='Choose model either esm2 or esm1v')
parser.add_argument('--batch_size', type=int, default=5)
args = parser.parse_args()

# model for embedding computation esm1v or esm2
model = args.model

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(script_path, '../data/DMS_enzymes/datasets')
representation_path = os.path.join(script_path, f'representations/{model}')

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
batch_size = 2
batch_paths = []
batch_seqs = []
for dataset in mutant_datasets:
    # study path
    study_name = dataset.split('.')[0]
    study_path = os.path.join(representation_path, study_name)
    
    # get names and sequences from dataframe
    df_path = os.path.join(dataset_path, dataset)
    df = pd.read_csv(df_path)
    sequences = df['mutated_sequence'].to_list()
    names = df['mutant'].to_list()

    for i in range(0, len(names)):
        n = names[i]
        seq_rep_path = os.path.join(study_path, n + '.pt')

        # check if representations are already computed
        if not os.path.exists(seq_rep_path): 
            batch_paths.append(seq_rep_path)
            batch_seqs.append(sequences[i])  # corrected line
    
        # compute representations
        if len(batch_paths) == batch_size:
            results, batch_lens, batch_labels, alphabet = esm_compute(batch_seqs, model=model)
            sequence_representations = get_seq_rep(results, batch_lens)
            for j in range(len(batch_paths)):
                torch.save(sequence_representations[j], batch_paths[j])
            
            # empty batches
            batch_paths = []
            batch_seqs = []

# compute representations which might have been skipped at the end
results, batch_lens, batch_labels, alphabet = esm_compute(batch_seqs, model=model)
sequence_representations = get_seq_rep(results, batch_lens)
for j in range(len(batch_paths)):
    torch.save(sequence_representations[j], batch_paths[j])
    
# empty batches
batch_paths = []
batch_seqs = []
