import os
import sys

import proteusAI as pai

print(os.getcwd())
sys.path.append("src/")


# will initiate storage space - else in memory
datasets = ["demo/demo_data/methyltransfereases.csv"]
y_columns = ["coverage_5"]

results_dictionary = {}
for dataset in datasets:
    for y in y_columns:
        # load data from csv or excel: x should be sequences, y should be labels, y_type class or num
        library = pai.Library(
            source=dataset,
            seqs_col="sequence",
            y_col=y,
            y_type="class",
            names_col="uid",
        )

        # compute and save ESM-2 representations at example_lib/representations/esm2
        library.compute(method="esm2", batch_size=10)

        # define a model
        model = pai.Model(library=library, k_folds=5, model_type="rf", x="esm2")

        # train model
        model.train()

        # search predict the classes of unknown sequences
        out, mask = model.search()

        # save results
        if not os.path.exists("demo/demo_data/out/"):
            os.makedirs("demo/demo_data/out/", exist_ok=True)

        out["df"].to_csv("demo/demo_data/out/discovery_search_results.csv")

        model_lib = pai.Library(source=out)

        model_lib.plot_tsne(model.x, None, None, model_lib.names)
