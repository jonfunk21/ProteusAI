#!/bin/bash

# Define the possible combinations of models and embeddings
models=("gp" "rf" "svm" "knn")
embeddings=("esm2" "ohe" "blosum62" "blosum50")
acq_fns=("ei" "ucb" "greedy" "random")

# Loop over each model
for model in "${models[@]}"; do
  # Loop over each embedding
  for emb in "${embeddings[@]}"; do
    for acq_fn in "${acq_fns[@]}"; do
      # Execute the script with the current combination of model and embedding
      python demo/MLDE_benchmark.py --model "$model" --emb "$emb" --acquisition_fn "$acq_fn"
    done
  done
done
