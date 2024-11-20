import os
import sys
import proteusAI as pai

print(os.getcwd())
sys.path.append("src/")


# will initiate storage space - else in memory
datasets = ["demo/demo_data/Nitric_Oxide_Dioxygenase_raw.csv"]
y_columns = ["Data"] 

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

        # df = out['df']
        # df.to_csv('<your results path>')
