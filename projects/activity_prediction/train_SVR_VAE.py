import pandas as pd
import os
import numpy as np
import sys
sys.path.insert(0, '../../src')
import proteusAI.ml_tools.torch_tools as torch_tools
import proteusAI.io_tools as io_tools
import proteusAI.ml_tools.sklearn_tools as sklearn_tools
from joblib import dump
import json
import torch
import argparse
from utils import compute_VAE_embeddings

# Argument parsing
parser = argparse.ArgumentParser(description="Process some strings.")
parser.add_argument('--encoder', type=str, default='OHE', help='choose encoding method amino acid sequences ["OHE", "BLOSUM50", "BLOSUM62"]')
parser.add_argument('--hidden_layers', type=str, default='2048,1024,256', help='Comma-separated list of hidden layer sizes')
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--dropout_p', type=float, default=0.0)
args = parser.parse_args()

# arguments
encoding_type = args.encoder
hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
z_dim = args.z_dim
dropout_p = args.dropout_p
batch_size = 256

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
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
    model_name = name + f'_svr_VAE_{encoding_type}'
    VAE_name = name + f'_{encoding_type}_VAE'
    
    # Load training, validation, and test sets
    train_df = pd.read_csv(os.path.join(train_dir, name + '.csv'))
    val_df = pd.read_csv(os.path.join(val_dir, name + '.csv'))
    test_df = pd.read_csv(os.path.join(test_dir, name + '.csv'))
    
    # Combine train and validation sets for grid search
    train_val_df = pd.concat([train_df, val_df])

    # mutants
    train_val_mutants = [n + '.pt' for n in train_val_df['mutant'].to_list()]
    test_mutants = [n + '.pt' for n in test_df['mutant'].to_list()]

    # load embeddings
    train_val_tensors = compute_VAE_embeddings(train_val_df, encoding_type, hidden_layers, z_dim, dropout_p, batch_size, VAE_name)
    test_tensors = compute_VAE_embeddings(test_df, encoding_type, hidden_layers, z_dim, dropout_p, batch_size, VAE_name)
    
    # Split features and targets
    X_train_val = torch.stack(train_val_tensors).cpu().numpy()
    y_train_val = train_val_df['y'].to_list()
    X_test = torch.stack(test_tensors).cpu().numpy()
    y_test = test_df['y'].to_list()

    # Initialize and train the SVR model using grid search
    print(f"Training {model_name} model...")
    best_model, test_r2, corr_coef, p_value, cv_results_df, best_params_ = sklearn_tools.svr_grid_search(
        Xs_train=X_train_val,
        Xs_test=X_test,
        ys_train=y_train_val,
        ys_test=y_test,
        verbose=2
    )
    
    # Predict on the test set
    predictions = best_model.predict(X_test)

    # plot best predictions
    sklearn_tools.plot_predictions_vs_groundtruth(y_test, predictions, fname=f'{results_plots_path}/{model_name}_pred_vs_true.png')

    # Save the best model to a file
    dump(best_model, f'{checkpoints_path}/{model_name}_best_model.joblib')

    # Save the best parameters to a JSON file
    with open(f'{results_path}/{model_name}_best_params.json', 'w') as f:
        json.dump(best_params_, f)

    # save results
    resutls = {'name':model_name, 'test_r2':test_r2, 'corr_coef':corr_coef}
    with open(f'{results_path}/{model_name}_results.json', 'w') as f:
        json.dump(resutls, f)

    # Append predictions to test_df and save
    test_df['predictions'] = predictions
    test_df.to_csv(f'{results_path}/{model_name}_predictions.csv', index=False)
