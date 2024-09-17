# %% [markdown]
# # Tutorial

# %% 
import proteusAI as pai
proteusAI.__version__

# load data from csv or excel: x should be sequences, y should be labels, y_type class or num
library = pai.Library(source='demo/demo_data/Nitric_Oxide_Dioxygenase_raw.csv', seqs_col='Sequence', y_col='Data', 
                    y_type='num', names_col='Description')


# compute and save ESM-2 representations at example_lib/representations/esm2
library.compute(method='esm2', batch_size=10)


# define a model
model = pai.Model(library=library, k_folds=5, model_type='rf', x='blosum62')


# train model
_ = model.train()

# search for new mutants
out = model.search()


# print predictions
print(out['df'])