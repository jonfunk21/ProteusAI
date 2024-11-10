import os
import sys

import proteusAI as pai

print(os.getcwd())
sys.path.append("src/")


# will initiate storage space - else in memory
datasets = ["demo/demo_data/Nitric_Oxide_Dioxygenase_raw.csv"]
y_columns = ["Data"]  # enter your names here, e.g. 'pae1', 'plddt1'

results_dictionary = {}
for dataset in datasets:
    for y in y_columns:
        # load data from csv or excel: x should be sequences, y should be labels, y_type class or num
        library = pai.Library(
            source=dataset,
            seqs_col="Sequence",
            y_col=y,
            y_type="num",
            names_col="Description",
        )

        # compute and save ESM-2 representations at example_lib/representations/esm2
        library.compute(method="esm2", batch_size=10)

        # define a model
        model = pai.Model(library=library, k_folds=5, model_type="rf", x="blosum62")

        # train model
        model.train()

        # search for new mutants
        out = model.search(optim_problem="max")

        # save results
        if not os.path.exists("demo/demo_data/out/"):
            os.makedirs("demo/demo_data/out/", exist_ok=True)

        out.to_csv("demo/demo_data/out/demo_search_results.csv")
