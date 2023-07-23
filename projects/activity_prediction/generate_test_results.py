import pandas as pd
import os
import numpy as np
import sys
sys.path.insert(0, '../../src')
import proteusAI.ml_tools.torch_tools as torch_tools
import proteusAI.ml_tools.sklearn_tools as sklearn_tools
import proteusAI.io_tools as io_tools
from joblib import load
import json
import torch

# script path
script_path = os.path.dirname(os.path.realpath(__file__))

results_plots_path = os.path.join(script_path, 'plots/results')
checkpoints_path = os.path.join(script_path, 'checkpoints')
results_path = os.path.join(script_path, 'results')
predictions_path = os.path.join(script_path, 'test_prediction')
os.makedirs(results_plots_path, exist_ok=True)
os.makedirs(predictions_path, exist_ok=True)

# Test data path
test_dir = os.path.join(script_path, '../data/DMS_enzymes/datasets/test')

names = [f.split('.')[0] for f in os.listdir(test_dir) if f.endswith('.csv')]

# List of encoding types
encoding_types = ['OHE', 'BLOSUM50', 'BLOSUM62', 'esm1v', 'esm2']

for encoding_type in encoding_types:
    if encoding_type in ['OHE', 'BLOSUM50', 'BLOSUM62']:
        # pick encoder
        if encoding_type == 'OHE':
            encoder = torch_tools.one_hot_encoder
        if encoding_type == 'BLOSUM50':
            encoder = lambda x: torch_tools.blosum_encoding(x, matrix='BLOSUM50')
        if encoding_type == 'BLOSUM62':
            encoder = lambda x: torch_tools.blosum_encoding(x, matrix='BLOSUM62')

        for name in names:
            # define model name for loading
            model_name = name + f'_svr_{encoding_type}'
            
            # Load test set
            test_df = pd.read_csv(os.path.join(test_dir, name + '.csv'))
            
            # Prepare test features
            X_test = encoder(test_df['mutated_sequence'].to_list()).numpy()
            X_test = X_test.reshape(X_test.shape[0], -1)  # flatten X
            
            # Load the best model from a file
            best_model = load(f'{checkpoints_path}/{model_name}_best_model.joblib')

            # Predict on the test set
            predictions = best_model.predict(X_test)

            # Append predictions to test_df and save
            test_df['predictions'] = predictions
            test_df.to_csv(f'{predictions_path}/{model_name}_predictions.csv', index=False)

    else:
        representations_path = os.path.join(script_path, f'representations/{encoding_type}')

        for name in names:
            rep_path = os.path.join(representations_path, name)
            
            # define model name for loading
            model_name = name + f'_svr_{encoding_type}'
            
            # Load test set
            test_df = pd.read_csv(os.path.join(test_dir, name + '.csv'))
            
            # mutants
            test_mutants = [n + '.pt' for n in test_df['mutant'].to_list()]

            # load embeddings
            _, test_tensors = io_tools.embeddings.load_embeddings(path=rep_path, names=test_mutants)
            
            # Prepare test features
            X_test = torch.stack(test_tensors).cpu().numpy()

            # Load the best model from a file
            best_model = load(f'{checkpoints_path}/{model_name}_best_model.joblib')

            # Predict on the test set
            predictions = best_model.predict(X_test)

            # Append predictions to test_df and save
            test_df['predictions'] = predictions
            test_df.to_csv(f'{predictions_path}/{model_name}_predictions.csv', index=False)
