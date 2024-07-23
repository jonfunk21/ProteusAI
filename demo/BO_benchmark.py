import os
import sys
sys.path.append('src/')
import proteusAI as pai

# hyperparams
USER = 'benchmark'
N = 5
MODEL = 'gp'
EMB = 'esm2'
BENCHMARK_FOLDER = 'demo/demo_data/DMS/'

# benchmark data
datasets = [f for f in os.listdir(BENCHMARK_FOLDER) if f.endswith('.csv')]
fastas = [f for f in os.listdir(BENCHMARK_FOLDER) if f.endswith('.fasta')]
datasets.sort()
fastas.sort()

def benchmark(dataset, fasta, model, embedding):
    # load data from csv or excel: x should be sequences, y should be labels, y_type class or num
    lib = pai.Library(user=USER, source=dataset, seqs_col='mutated_sequence', y_col='DMS_score', 
                        y_type='num', names_col='mutant')

    # wt sequence
    protein = pai.Protein(user=USER, source=fasta)

    # zero-shot scores
    out = protein.zs_prediction(model=embedding, batch_size=1)
    zs_lib = pai.Library(user=USER, source=out)

    # collect the top N proteins of the zs-library
    #top_N = zs_lib.top_n(n=N)
    #top_N_names = [prot.name for prot in top_N]

    # compute and save ESM-2 representations at example_lib/representations/esm2
    lib.compute(method='esm2', batch_size=1)

    # train on the top N proteins first

    # sample within the dataset, check how long it takes until the best variants are found


for i in range(len(datasets)):
    d = os.path.join(BENCHMARK_FOLDER, datasets[i])
    f = os.path.join(BENCHMARK_FOLDER, fastas[i])
    benchmark(d, f, model=MODEL, embedding=EMB)