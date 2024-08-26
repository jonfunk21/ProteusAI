import os
import sys
sys.path.append('src/')
import proteusAI as pai
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import pandas as pd

# Initialize the argparse parser
parser = argparse.ArgumentParser(description="Benchmarking ProteusAI MLDE")

# Add arguments corresponding to the variables
parser.add_argument('--user', type=str, default='benchmark', help='User name or identifier.')
parser.add_argument('--sample-sizes', type=int, nargs='+', default=[5, 10, 20, 100], help='List of sample sizes.')
parser.add_argument('--model', type=str, default='gp', help='Model name.')
parser.add_argument('--rep', type=str, default='esm2', help='Representation type.')
parser.add_argument('--zs-model', type=str, default='esm1v', help='Zero-shot model name.')
parser.add_argument('--benchmark-folder', type=str, default='demo/demo_data/DMS/', help='Path to the benchmark folder.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--max-iter', type=int, default=100, help='Maximum number of iterations (None for unlimited).')
parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., cuda, cpu).')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing.')
parser.add_argument('--improvement', type=str, nargs='+', default=[5, 10, 20, 50, 'improved'], help='List of improvements.')
parser.add_argument('--acquisition_fn', type=str, default='ei', help='ProteusAI acquisition functions')
parser.add_argument('--k_folds', type=int, default=5, help='K-fold cross validation for sk_learn models')


def benchmark(dataset, fasta, model, embedding, name, sample_size, results_df):
    iteration = 1

    # load data from csv or excel: x should be sequences, y should be labels, y_type class or num
    lib = pai.Library(user=USER, source=dataset, seqs_col='mutated_sequence', y_col='DMS_score', 
                      y_type='num', names_col='mutant')
    
    # compute representations for this dataset
    lib.compute(method=REP, batch_size=BATCH_SIZE, device=DEVICE)

    # plot destination
    #plot_dest = os.path.join(lib.user, name, embedding, 'plots', str(sample_size))
    #os.makedirs(plot_dest, exist_ok=True)

    # wt sequence
    protein = pai.Protein(user=USER, source=fasta)

    # zero-shot scores
    out = protein.zs_prediction(model=ZS_MODEL, batch_size=BATCH_SIZE, device=DEVICE)
    zs_lib = pai.Library(user=USER, source=out)

    # Simulate selection of top N ZS-predictions for the initial librarys
    zs_prots = [prot for prot in zs_lib.proteins if prot.name in lib.names]
    sorted_zs_prots = sorted(zs_prots, key=lambda prot: prot.y_pred, reverse=True)


    
    # train on the ZS-selected initial library (assume that they have been assayed now) train with 80:10:10 split
    # if the sample size is to small the initial batch has to be large enough to give meaningful statistics. Here 15
    if sample_size <= 15:
        top_N_zs_names = [prot.name for prot in sorted_zs_prots[:15]]
        zs_selected = [prot for prot in lib.proteins if prot.name in top_N_zs_names]
        n_train = 11
        n_test = 2
        _sample_size = 15
    else:
        top_N_zs_names = [prot.name for prot in sorted_zs_prots[:sample_size]]
        zs_selected = [prot for prot in lib.proteins if prot.name in top_N_zs_names]
        n_train = int(sample_size*0.8)
        n_test = int(sample_size*0.1)
        _sample_size = sample_size
        
    # train model
    model = pai.Model(model_type=model)
    model.train(library=lib, x=REP, split={'train':zs_selected[:n_train], 'test':zs_selected[n_train:n_train+n_test], 'val':zs_selected[n_train+n_test:_sample_size]}, seed=SEED, model_type=MODEL, k_folds=K_FOLD)
    
    # add to results df
    results_df = add_to_data(data=results_df, proteins=zs_selected, iteration=iteration, sample_size=_sample_size, dataset=name, model=model)

    # use the model to make predictions on the remaining search space
    search_space = [prot for prot in lib.proteins if prot.name not in top_N_zs_names]
    ranked_search_space, _, _, _, _ = model.predict(search_space, acq_fn=ACQ_FN)

    # Prepare the tracking of top N variants, 
    top_variants_counts = IMPROVEMENT
    found_counts = {count: 0 for count in top_variants_counts}
    first_discovered = [None] * len(top_variants_counts)

    # Identify the top variants by score
    actual_top_variants = sorted(lib.proteins, key=lambda prot: prot.y, reverse=True)
    actual_top_variants = {count: [prot.name for prot in actual_top_variants[:count]] for count in top_variants_counts if type(count) == int}

    # count variants that are improved over wt, 1 standard deviation above wt
    y_std = np.std(lib.y, ddof=1)
    y_mean = np.mean(lib.y)
    actual_top_variants['improved'] = [prot.name for prot in lib.proteins if prot.y > y_mean + 1*y_std]

    # add sequences to the new dataset, and continue the loop until the dataset is exhausted
    sampled_data = zs_selected

    while len(ranked_search_space) >= sample_size:
        # Check if we have found all top variants, including the 1st zero-shot round
        sampled_names = [prot.name for prot in sampled_data]
        for c, count in enumerate(found_counts):
            found = len([prot.name for prot in sampled_data if prot.name in actual_top_variants[count]])
            found_counts[count] = found
            if found > 0 and first_discovered[c] == None:
                first_discovered[c] = iteration
        
        # Break the loop if all elements in first_discovered are not None
        if all(value is not None for value in first_discovered):
            break

        # Break if maximum number of iterations have been reached
        if iteration == MAX_ITER:
            break

        iteration += 1

        # select variants that have now been 'assayed'
        sample = ranked_search_space[:sample_size]

        # Remove the selected top N elements from the ranked search space
        ranked_search_space = ranked_search_space[sample_size:]

        # add new 'assayed' sample to sampled data
        sampled_data += sample

        # split into train, test and val
        n_train = int(len(sampled_data)*0.8)
        n_test = int(len(sampled_data)*0.1)
        
        # handle the very low data results
        if n_test == 0:
            n_test = 1
            n_train = n_train - n_test

        split = {
            'train': sampled_data[:n_train],
            'test': sampled_data[n_train:n_train+n_test],
            'val': sampled_data[n_train+n_test:]
        }

        # train model on new data
        model.train(library=lib, x=REP, split=split, seed=SEED, model_type=MODEL, k_folds=K_FOLD)

        # add to results
        results_df = add_to_data(data=results_df, proteins=sample, iteration=iteration, sample_size=sample_size, dataset=name, model=model)

        # re-score the new search space
        ranked_search_space, sorted_y_pred, sorted_sigma_pred, y_val, sorted_acq_score = model.predict(ranked_search_space, acq_fn=ACQ_FN)
    
    # save when the first datapoints for each dataset and category have been discvered
    first_discovered_data[name][sample_size] = first_discovered

    return found_counts, results_df


def plot_results(found_counts, name, iter, dest, sample_size):
    counts = list(found_counts.keys())
    found = [found_counts[count] for count in counts]
    x_positions = range(len(counts))  # Create fixed distance positions for the x-axis

    plt.figure(figsize=(10, 6))
    plt.bar(x_positions, found, color='skyblue')
    plt.xlabel('Top N Variants')
    plt.ylabel('Number of Variants Found')
    plt.title(f'{name}: Number of Top N Variants Found After {iter} Iterations')
    plt.xticks(x_positions, counts)  # Set custom x-axis positions and labels
    plt.ylim(0, max(found) + 1)

    for i, count in enumerate(found):
        plt.text(x_positions[i], count + 0.1, str(count), ha='center')

    plt.savefig(os.path.join(dest, f'top_variants_{iter}_iterations_{name}.png'))


def add_to_data(data: pd.DataFrame, proteins, iteration, sample_size, dataset, model):
    """Add sampling results to dataframe"""
    names = [prot.name for prot in proteins]
    ys = [prot.y for prot in proteins]
    y_preds = [prot.y_pred for prot in proteins]
    y_sigmas = [prot.y_sigma for prot in proteins]
    models = [MODEL] * len(names)
    reps = [REP] * len(names)
    acq_fns = [ACQ_FN] * len(names)
    rounds = [iteration] * len(names)
    datasets = [dataset] * len(names)
    sample_sizes = [sample_size] * len(names)
    test_r2s = [model.test_r2] * len(names)
    val_r2s = [model.val_r2] * len(names)

    new_data = pd.DataFrame({
        "name": names,
        "y": ys,
        "y_pred": y_preds,
        "y_sigma": y_sigmas,
        "test_r2":test_r2s,
        "val_r2":val_r2s,
        "model": models,
        "rep": reps,
        "acq_fn": acq_fns,
        "round": rounds,
        "sample_size":sample_sizes,
        "dataset": datasets
    })

    # Append the new data to the existing DataFrame
    updated_data = pd.concat([data, new_data], ignore_index=True)
    return updated_data


# Parse the arguments
args = parser.parse_args()

# Assign parsed arguments to capitalized variable names
USER = args.user
SAMPLE_SIZES = args.sample_sizes
MODEL = args.model
REP = args.rep
ZS_MODEL = args.zs_model
BENCHMARK_FOLDER = args.benchmark_folder
SEED = args.seed
MAX_ITER = args.max_iter
DEVICE = args.device
BATCH_SIZE = args.batch_size
IMPROVEMENT = args.improvement
ACQ_FN = args.acquisition_fn
K_FOLD = args.k_folds

# benchmark data
datasets = [f for f in os.listdir(BENCHMARK_FOLDER) if f.endswith('.csv')]
fastas = [f for f in os.listdir(BENCHMARK_FOLDER) if f.endswith('.fasta')]
datasets.sort()
fastas.sort()

# save sampled data
results_df = pd.DataFrame({
            "name":[],
            "y":[],
            "y_pred":[],
            "y_sigma":[],
            "test_r2":[],
            "val_r2":[],
            "model":[],
            "rep":[],
            "acq_fn":[],
            "round":[],
            "sample_size":[],
            "dataset":[]
        })

first_discovered_data = {}
for i in range(len(datasets)):
    for N in SAMPLE_SIZES:
        print(f"RUNNING model:{MODEL}, rep:{REP}, acq:{ACQ_FN}, sample_size:{N}", flush=True)
        d = os.path.join(BENCHMARK_FOLDER, datasets[i])
        f = os.path.join(BENCHMARK_FOLDER, fastas[i])
        name = datasets[i][:-4]
        
        if N == SAMPLE_SIZES[0]:
            first_discovered_data[name] = {N:[]}
        else:
            first_discovered_data[name][N] = []
            
        found_counts, results_df = benchmark(d, f, model=MODEL, embedding=REP, name=name, sample_size=N, results_df=results_df)
        # save first discovered data
        with open(os.path.join('usrs/benchmark/', f'first_discovered_data_{MODEL}_{REP}_{ACQ_FN}.json'), 'w') as file:
            json.dump(first_discovered_data, file)   
        
        results_df.to_csv(os.path.join('usrs/benchmark/', f'results_df_{MODEL}_{REP}_{ACQ_FN}.csv'), index=False)
