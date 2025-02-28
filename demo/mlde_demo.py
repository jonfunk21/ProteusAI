import os
import sys

import proteusAI as pai

print(os.getcwd())
sys.path.append("src/")


# will initiate storage space - else in memory
dataset = "demo/demo_data/indigo_data.csv"

class_cols = ["indigo"]
num_cols = [
    "concentration",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
]

y_types = ["num"] * len(num_cols) + ["class"] * len(class_cols)

# load data from csv or excel: x should be sequences, y should be labels, y_type class or num
library = pai.Library(
    source=dataset,
    seqs_col="sequence",
    y_cols=num_cols + class_cols,
    y_types=y_types,
    names_col="description",
)

# compute and save ESM-2 representations at example_lib/representations/esm2
library.compute(method="esm2_8M", batch_size=10)

# define a model
model = pai.Model(library=library, k_folds=1, model_type="rf", x="vhse")

# train model
model.train()

# visualize model results
plots = model.predicted_vs_true_all()

# save plots
for i, (fig, ax) in enumerate(plots):
    fig.savefig(f"demo/demo_data/out/predicted_vs_true_{num_cols[i]}.png")

# search for new mutants
out = model.mlde(optim_problem="max")

# save results
if not os.path.exists("demo/demo_data/out/"):
    os.makedirs("demo/demo_data/out/", exist_ok=True)

out.to_csv("demo/demo_data/out/demo_mlde_results.csv")
