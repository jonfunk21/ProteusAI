#!/bin/bash

# Define the possible combinations of models and embeddings
representations=("esm2" "blosum62" "ohe") # "blosum50"
acq_fns=("ei" "greedy" "ucb") # "random"
models=($1) # "gp" "rf" "ridge" "svm" "knn"

for acq_fn in "${acq_fns[@]}"; do
  for rep in "${representations[@]}"; do
    for model in "${models[@]}"; do
      # Execute the script with the current combination of model and embedding
      python demo/MLDE_benchmark.py --model "$model" --rep "$rep" --acquisition_fn "$acq_fn" --max-iter 100
      python demo/plot_benchmark.py --rep "$rep" --max-sample 100 --model "$model" --acquisition-fn "$acq_fn"
    done
  done
done