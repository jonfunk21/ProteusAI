import pandas as pd
import os
import numpy as np
import sys
sys.path.insert(0, '../../src')
import proteusAI.ml_tools.torch_tools as torch_tools
import proteusAI.ml_tools.sklearn_tools as sklearn_tools
from joblib import dump
import json

# arguments
encoding_type = 'OHE'
esm_model = 'esm1v'

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
representations_path = os.path.join(script_path, f'representations/{esm_model}')

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
train_dir = os.path.join(script_path, 'datasets/train')
val_dir = os.path.join(script_path, 'datasets/validate')
test_dir = os.path.join(script_path, 'datasets/test')

names = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith('.csv')]

# pick encoder
if encoding_type == 'OHE':
    min_val, max_val = (0, 1)
    encoder = torch_tools.one_hot_encoder
if encoding_type == 'BLOSUM50':
    min_val, max_val = (-5.0, 15.0)
    encoder = lambda x: torch_tools.blosum_encoding(x, matrix='BLOSUM50')
if encoding_type == 'BLOSUM62':
    min_val, max_val = (-4.0, 11.0)
    encoder = lambda x: torch_tools.blosum_encoding(x, matrix='BLOSUM62')

for name in names:
    # define model name for saving
    model_name = name + '_svr_regressor'
    
    # Load training, validation, and test sets
    train_df = pd.read_csv(os.path.join(train_dir, name + '.csv'))
    val_df = pd.read_csv(os.path.join(val_dir, name + '.csv'))
    test_df = pd.read_csv(os.path.join(test_dir, name + '.csv'))
    
    # Combine train and validation sets for grid search
    train_val_df = pd.concat([train_df, val_df])
    
    # Split features and targets
    X_train_val = train_val_df['mutated_sequence'].to_list()
    y_train_val = train_val_df['y'].to_list()
    X_test = test_df['mutated_sequence'].to_list()
    y_test = test_df['y'].to_list()
    
    # to numpy arra
    X_train_val = encoder(X_train_val).numpy()
    X_test = encoder(X_test).numpy()
    
    # flatten X
    X_train_val = X_train_val.reshape(X_train_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Scale X from 1 to 0
    if min_val != 0 and max_val != 1:
        X_train_val = (X_train_val - min_val) / (max_val - min_val)
        X_test = (X_test - min_val) / (max_val - min_val)
    
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
    sklearn_tools.plot_predictions_vs_groundtruth(y_test, predictions, fname=f'{results_plots_path}/{name}_{encoding_type}_pred_vs_true.png')

    # Save the best model to a file
    dump(best_model, f'{checkpoints_path}/{name}_{encoding_type}_best_model.joblib')

    # save results
    resutls = {'name':model_name, 'test_r2':test_r2, 'corr_coef':corr_coef}
    with open(f'{results_path}/{name}_{encoding_type}_results.json', 'w') as f:
        json.dump(resutls, f)


    # Save the best parameters to a JSON file
    with open(f'{results_path}/{name}_{encoding_type}_best_params.json', 'w') as f:
        json.dump(best_params_, f)

    # Append predictions to test_df and save
    test_df['predictions'] = predictions
    test_df.to_csv(f'{results_path}/{name}_{encoding_type}_predictions.csv', index=False)