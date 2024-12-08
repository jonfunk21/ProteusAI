import os

import proteusAI as pai

files = ["demo/demo_data/Nitric_Oxide_Dioxygenase_wt.fasta"]  # input files
models = [
    "esm2_8M"
]  # Ideally use esm1v. ems2_8M is fast and used for demo purposes. esm1v is more accurate.

for f in files:
    protein = pai.Protein(source=f)

    for model in models:
        out = protein.zs_prediction(model=model)

        # save results
        if not os.path.exists("demo/demo_data/out/"):
            os.makedirs("demo/demo_data/out/", exist_ok=True)

        out["df"].to_csv(
            f"demo/demo_data/zs_outdemo_{f.split('/')[-1][:-6]}.csv", index=False
        )

        print(out["df"])
