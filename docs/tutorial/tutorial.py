# %% [markdown]
# # Tutorial

# %% 
import proteusAI as pai
pai.__version__

# # MLDE Tutorial
# %% [markdown]
# ## Loading data from csv or excel
# The data must contain the mutant sequences and y_values for the MLDE workflow. It is recommended to have useful sequence names for later interpretability, and to define the data type.

# %% 
library = pai.Library(source='demo/demo_data/Nitric_Oxide_Dioxygenase_raw.csv', seqs_col='Sequence', y_col='Data', 
                    y_type='num', names_col='Description')


# %% [markdown]
# ## Compute representations (skipped here to save time)
# The available representations are ('esm2', 'esm1v', 'blosum62', 'blosum50', and 'ohe')

# %% 
# library.compute(method='esm2', batch_size=10) # uncomment this line to compute esm2 representations


# %% [markdown]
# ## Define a model, using fast to compute BLOSUM62 representations (good for demo purposes and surprisingly competetive with esm2 representations).

# %% 
model = pai.Model(library=library)

# %% [markdown]
# ## Training the model

# %% 
_ = model.train(k_folds=5, model_type='rf', x='blosum62')


# %% [markdown]
# ## Search for new mutants
# Searching new mutants will produce an output dataframe containing the new predictions. Here we are using the expected improvement ('ei') acquisition function.

# %% 
out = model.search(acq_fn='ei')
out