#!/bin/bash

# Define the possible combinations of models and embeddings
models=("gp" "rf" "svm" "knn")
representations=("esm2" "ohe" "blosum62" "blosum50")
acq_fns=("ei" "ucb" "greedy" "random")

## Loop over each model
for model in "${models[@]}"; do
  # Loop over each embedding
  for rep in "${representations[@]}"; do
    for acq_fn in "${acq_fns[@]}"; do
      # Execute the script with the current combination of model and embedding
      python demo/MLDE_benchmark.py --model "$model" --rep "$rep" --acquisition_fn "$acq_fn"
      python demo/plot_benchmark.py --rep "$rep" --max-sample 100 --model "$model" --acquisition-fn "$acq_fn"
    done
  done
done

#python demo/MLDE_benchmark.py --model "gp" --rep "esm2" --acquisition_fn "ei"
#python demo/plot_benchmark.py --rep "esm2" --max-sample 20 --model "gp" --acquisition-fn "ei"