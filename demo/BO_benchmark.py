import os
import sys
sys.path.append('src/')
import proteusAI as pai
import matplotlib.pyplot as plt
import numpy as np
import json

# hyperparams
USER = 'benchmark'
Ns = [10, 20, 100, 1000]
MODEL = 'gp'
EMB = 'esm2'
BENCHMARK_FOLDER = 'demo/demo_data/DMS/'
SEED = 42
MAX_ITER = 20
DEVICE = 'cuda'
BATCH_SIZE = 20

# benchmark data
datasets = [f for f in os.listdir(BENCHMARK_FOLDER) if f.endswith('.csv')]
fastas = [f for f in os.listdir(BENCHMARK_FOLDER) if f.endswith('.fasta')]
datasets.sort()
fastas.sort()


def benchmark(dataset, fasta, model, embedding, name, sample_size):
    # load data from csv or excel: x should be sequences, y should be labels, y_type class or num
    lib = pai.Library(user=USER, source=dataset, seqs_col='mutated_sequence', y_col='DMS_score', 
                      y_type='num', names_col='mutant')

    # plot destination
    plot_dest = os.path.join(lib.user, name, 'plots', str(sample_size))
    os.makedirs(plot_dest, exist_ok=True)

    # wt sequence
    protein = pai.Protein(user=USER, source=fasta)

    # zero-shot scores
    out = protein.zs_prediction(model=embedding, batch_size=BATCH_SIZE, device=DEVICE)
    zs_lib = pai.Library(user=USER, source=out)

    # Simulate selection of top N ZS-predictions for the initial library
    zs_prots = [prot for prot in zs_lib.proteins if prot.name in lib.names]
    sorted_zs_prots = sorted(zs_prots, key=lambda prot: prot.y, reverse=True)
    top_N_zs_names = [prot.name for prot in sorted_zs_prots[:sample_size]]

    # compute and save ESM-2 representations for this dataset
    lib.compute(method='esm2', batch_size=BATCH_SIZE, device=DEVICE)

    # train on the ZS-selected initial library (assume that they have been assayed now) train with 80:10:10 split
    zs_selected = [prot for prot in lib.proteins if prot.name in top_N_zs_names]
    n_train = int(sample_size*0.8)
    n_test = int(sample_size*0.1)
    m = pai.Model(model_type=MODEL)
    m.train(library=lib, x=EMB, split={'train':zs_selected[:n_train], 'test':zs_selected[n_train:n_train+n_test], 'val':zs_selected[n_train+n_test:sample_size]}, seed=SEED, model_type=MODEL)

    # use the model to make predictions on the remaining search space
    search_space = [prot for prot in lib.proteins if prot.name not in top_N_zs_names]
    ranked_search_space, sorted_y_pred, sorted_sigma_pred, sorted_acq_score = m.predict(search_space) # here is the BO strategy that needs to be inserted

    # Prepare the tracking of top N variants, 
    top_variants_counts = [5, 10, 20, 50, 'improved']
    found_counts = {count: 0 for count in top_variants_counts}
    first_discovered = [None] * len(top_variants_counts)

    # Identify the top variants by score
    actual_top_variants = sorted(lib.proteins, key=lambda prot: prot.y, reverse=True)
    actual_top_variants = {count: [prot.name for prot in actual_top_variants[:count]] for count in top_variants_counts if type(count) == int}

    # count variants that are improved over wt, 1 standard deviation above wt
    y_std = np.std(lib.y, ddof=1)
    actual_top_variants['improved'] = [prot.name for prot in lib.proteins if prot.y > 1*y_std]

    # add sequences to the new dataset, and continue the loop until the dataset is exhausted
    sampled_data = zs_selected
    iteration = 1

    while len(ranked_search_space) >= sample_size:
        # Check if we have found all top variants, including the 1st zero-shot round
        sampled_names = [prot.name for prot in sampled_data]
        for c, count in enumerate(found_counts):
            found = len([prot.name for prot in sampled_data if prot.name in actual_top_variants[count]])
            found_counts[count] = found
            if found > 0 and first_discovered[c] == None:
                first_discovered[c] = iteration
        
        plot_results(found_counts, name, iteration, plot_dest, sample_size)

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

        split = {
            'train': sampled_data[:n_train],
            'test': sampled_data[n_train:n_train+n_test],
            'val': sampled_data[n_train+n_test:]
        }

        # train model on new data
        m.train(library=lib, x=EMB, split=split, seed=SEED, model_type=MODEL)

        # re-score the new search space
        ranked_search_space, sorted_y_pred, sorted_sigma_pred, y_val, sorted_acq_score = m.predict(ranked_search_space)
    
    # save when the first datapoints for each dataset and category have been discvered
    first_discovered_data[name][sample_size] = first_discovered

    return found_counts

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
    

first_discovered_data = {}
for i in range(len(datasets)):

    for N in Ns:
        d = os.path.join(BENCHMARK_FOLDER, datasets[i])
        f = os.path.join(BENCHMARK_FOLDER, fastas[i])
        name = datasets[i][:-4]

        if N == Ns[0]:
            first_discovered_data[name] = {N:[]}
        else:
            first_discovered_data[name][N] = []
            
        found_counts = benchmark(d, f, model=MODEL, embedding=EMB, name=name, sample_size=N)
        # save first discovered data
        with open(os.path.join('usrs/benchmark/', 'first_discovered_data.json'), 'w') as file:
            json.dump(first_discovered_data, file)   
