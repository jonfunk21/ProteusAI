import os

import proteusAI as pai

files = ["demo/demo_data/GB1.pdb"]  # input files
temps = [1.0]  # sampling temperatures
fixed = {"GB1": [1, 2, 3]}  # fixed residues

for f in files:
    protein = pai.Protein(source=f)

    fname = f.split("/")[1][:-4]

    for temp in temps:
        for key, fix in fixed.items():
            out = protein.esm_if(
                num_samples=100, target_chain="A", temperature=temp, fixed=fix
            )

            # save results
            if not os.path.exists("demo/demo_data/out/"):
                os.makedirs("demo/demo_data/out/", exist_ok=True)

            out["df"].to_csv(
                f"demo/demo_data/out{fname}_temp_{temp}_{key}_out.csv", index=False
            )

            print(out["df"])
