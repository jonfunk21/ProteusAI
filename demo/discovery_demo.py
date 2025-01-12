import os
import sys
import matplotlib.pyplot as plt

import proteusAI as pai

print(os.getcwd())
sys.path.append("src/")


# will initiate storage space - else in memory
dataset = "demo/demo_data/methyltransfereases.csv"
y_column = "coverage_5"

results_dictionary = {}

# load data from csv or excel: x should be sequences, y should be labels, y_type class or num
library = pai.Library(
    source=dataset,
    seqs_col="sequence",
    y_col=y_column,
    y_type="class",
    names_col="uid",
)

# compute and save ESM-2 representations at example_lib/representations/esm2
library.compute(method="esm2_8M", batch_size=10)

# define a model
model = pai.Model(library=library, k_folds=5, model_type="rf", rep="esm2_8M")

# train model
model.train()

# search predict the classes of unknown sequences
out = model.search()
search_mask = out["mask"]

# save results
if not os.path.exists("demo/demo_data/out/"):
    os.makedirs("demo/demo_data/out/", exist_ok=True)

out["df"].to_csv("demo/demo_data/out/discovery_search_results.csv")

model_lib = pai.Library(source=out)

# plot results
fig, ax, plot_df = model.library.plot(
    rep="esm2_8M", use_y_pred=True, highlight_mask=search_mask
)
plt.savefig("demo/demo_data/out/search_results.png")
