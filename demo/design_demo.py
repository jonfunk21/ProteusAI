import os

import proteusAI as pai

files = ["demo/demo_data/GB1.pdb"]  # input files
temps = [1.0]  # sampling temperatures
fixed = {"GB1": [1, 2, 3]}  # fixed residues
num_samples = 100  # number of samples

for f in files:
    protein = pai.Protein(source=f)

    fname = f.split("/")[-1][:-4]

    for temp in temps:
        for key, fix in fixed.items():
            out = protein.esm_if(
                num_samples=num_samples, target_chain="A", temperature=temp, fixed=fix
            )

            # save results
            if not os.path.exists("demo/demo_data/out/"):
                os.makedirs("demo/demo_data/out/", exist_ok=True)

            out["df"].to_csv(
                f"demo/demo_data/out_{fname}_temp_{temp}_{key}_out.csv", index=False
            )


### UNCOMMENT THESE LINES TO FOLD DESIGNS IF YOU HAVE ESM-FOLD INSTALLED ###

# Designs can be folded to check the structure and confidence of the designs
# import shutil
# design_library = pai.Library(source=out)

# this will fold the designs
# fold_out = design_library.fold()

# save the folded designs
# fold_out["df"].to_csv(
#    f"demo/demo_data/out{fname}_temp_{temp}_{key}_folded.csv", index=False
# )

# save the structures
# os.makedirs(f"demo/demo_data/out/{fname}_temp_{temp}_{key}_pdb/", exist_ok=True)

# Move all files from source to destination
# for file in os.listdir(out["struc_path"]):
#    shutil.move(
#        os.path.join(out["struc_path"], file),
#        f"demo/demo_data/out/{fname}_temp_{temp}_{key}_pdb/",
#    )
