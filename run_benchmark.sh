#!/bin/bash

# Define the possible combinations of models and embeddings
representations=("esm2" "ohe" "blosum62" "blosum50")
models=("gp" "rf" "svm" "knn")
acq_fns=("ei" "ucb" "greedy" "random")

for acq_fn in "${acq_fns[@]}"; do
  for model in "${models[@]}"; do
    for rep in "${representations[@]}"; do
      # Execute the script with the current combination of model and embedding
      python demo/MLDE_benchmark.py --model "$model" --rep "$rep" --acquisition_fn "$acq_fn" --max-iter 100
      python demo/plot_benchmark.py --rep "$rep" --max-sample 100 --model "$model" --acquisition-fn "$acq_fn"
    done
  done
done

#python demo/MLDE_benchmark.py --model "gp" --rep "esm2" --acquisition_fn "ei"
#python demo/plot_benchmark.py --rep "esm2" --max-sample 20 --model "gp" --acquisition-fn "ei"