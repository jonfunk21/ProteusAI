### THIS DEMO ONLY WORKS IF ESM-FOLD IS INSTALLED ###

import os
import shutil

import proteusAI as pai

files = ["demo/demo_data/Nitric_Oxide_Dioxygenase_wt.fasta"]  # input files

for f in files:
    protein = pai.Protein(source=f)

    # fold the protein
    fold_out = protein.esm_fold(relax=True)

    # design the now folded protein
    design_out = protein.esm_if(
        num_samples=100, target_chain="A", temperature=1.0, fixed=[]
    )

    # move results to demo folder
    if not os.path.exists("demo/demo_data/out/"):
        os.makedirs("demo/demo_data/out/", exist_ok=True)

    # save the design
    design_out["df"].to_csv("demo/demo_data/fold_design_out.csv", index=False)

    # move the structures
    os.makedirs("demo/demo_data/out/fold_design_pdb/", exist_ok=True)

    # Copy all files from the source to the destination directory
    for file in os.listdir(fold_out["struc_path"]):
        src = os.path.join(fold_out["struc_path"], file)
        dst = os.path.join("demo/demo_data/out/fold_design_pdb/", file)
        shutil.copy(src, dst)
