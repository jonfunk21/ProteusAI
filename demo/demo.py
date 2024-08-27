import os
print(os.getcwd())

import sys
sys.path.append('src/')
import proteusAI as pai

# will initiate storage space - else in memory
#library = pai.Library(user='guest') 

# load data from csv or excel: x should be sequences, y should be labels, y_type class or num
library = pai.Library(user='guest', source='demo/demo_data/Nitric_Oxide_Dioxygenase_raw.csv', seqs_col='Sequence', y_col='Description', 
                    y_type='class', names_col='Description')


# compute and save ESM-2 representations at example_lib/representations/esm2
library.compute(method='esm2', batch_size=10)

# collect the top 5 proteins of the library
top_5 = library.top_n(n=5)

# fold the top proteins
dest = os.path.join(library.user, library.file, "protein")

for top_i in top_5:
    top_i.esm_fold(dest=dest)

# perform structure based design
dest = os.path.join(library.user, library.file, "design")
designs = [top_i.esm_if(num_samples=2, dest=dest) for top_i in top_5]

# do zero shot predictions for the proteins
design_libs = {}
for i, design in enumerate(designs):
    lib_path = os.path.join(design[:-4], "library") # create custom library path
    lib = pai.Library(user='guest')
    lib.read_data(design, seqs='sequence', names='seqid', y='log_likelihood', y_type='num', lib_path=lib_path)
    design_libs[i] = lib
    
design_libs
# visualize the protein islands with UMAP