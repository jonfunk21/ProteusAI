data_tooltips = """
ProteusAI, a user-friendly and open-source ML platform, streamlines protein engineering and design tasks.
ProteusAI offers modules to support researchers in various stages of the design-build-test-learn (DBTL) cycle,
including protein discovery, structure-based design, zero-shot predictions, and ML-guided directed evolution (MLDE).
Our benchmarking results demonstrate ProteusAI’s efficiency in improving proteins and enzymes within a few DBTL-cycle
iterations. ProteusAI democratizes access to ML-guided protein engineering and is freely available for academic and
commercial use.
You can upload different data types to get started with ProteusAI. Click on the other module tabs to learn about their
functionality and the expected data types.
"""

zs_file_type = """
Upload a FASTA file containing the protein sequence, or a PDB file containing the structure of the protein
for which you want to generate zero-shot predictions.
"""

zs_tooltips = """
The ProteusAI Zero-Shot Module is designed to create a mutant library with no prior data.
The module uses scores generated by large protein language models, such as ESM-1v, that have
been trained to predict hidden residues in hundreds of millions of protein sequences.
Often, you will find that several residues in your protein sequence have low predicted probabilities.
It has been previously shown that swapping these residues for residues with higher probabilities
has beneficial effects on the candidate protein. In ProteusAI, we provide access to several language
models which have been trained under slightly different conditions. The best models to produce Zero-Shot
scores are ESM-1v and ESM-2 (650M). However, these models will take a long time to compute the results.
Consider using ESM-2 (35M) to get familiar with the module first before moving to the larger models.
"""

discovery_file_type = """
Upload a CSV or EXCEL file in the 'Data' tab under 'Library' to proceed with the Discovery module.
The file should contain a column for protein sequences, a column with the protein names, and a column for
annotations, which can also be empty or partially populated with annotations.
"""

discovery_tooltips = """
The Discovery Module offers a structured approach to identifying proteins even with little to no
experimental data to start with. The goal of the module is to identify proteins with similar
functions and to propose novel sequences that are likely to have similar functions.
The module relies on representations generated by large protein language models that transform protein
sequences into meaningful vector representations. It has been shown that these vector representations often
cluster based on function. Clustering should be used if all, very few, or no sequences have annotations.
Classification should be used if some or all sequences are annotated. To find out if you have enough
sequences for classification, we recommend using the model statistics on the validation set, which are
automatically generated by the module after training.
"""

mlde_file_type = """
Upload a CSV or EXCEL file in the 'Data' tab under 'Library' to proceed with the MLDE module.
The file should contain a column for protein sequences, a column with the protein names (e.g.,
mutant descriptions M15V), and a column for experimental values (e.g., enzyme activity,
fluorescence, etc.).
"""

mlde_tooltips = """
The Machine Learning Guided Directed Evolution (MLDE) module offers a structured approach to
improve protein function through iterative mutagenesis, inspired by Directed Evolution.
Here, machine learning models are trained on previously generated experimental results. The
'Search' algorithm will then propose novel sequences that will be evaluated and ranked by the
trained model to predict mutants that are likely to improve function. The Bayesian optimization
algorithms used to search for novel mutants are based on models trained on protein representations
that can either be generated from large language models, which is currently very slow, or from
classical algorithms such as BLOSUM62. For now, we recommend the use of BLOSUM62 representations
combined with Random Forest models for the best trade-off between speed and quality. However, we encourage
experimentation with other models and representations.
"""

design_file_type = """
Upload a PDB file containing the structure of the protein
to use the (structure-based) Design module.
"""

design_tooltips = """
The Protein Design module is a structure-based approach to predict novel sequences using 'Inverse Folding'
algorithms. The designed sequences are likely to preserve the fold of the protein while improving
the thermal stability and solubility of proteins. To preserve important functions of the protein, we recommend
the preservation of protein-protein, ligand-ligand interfaces, and evolutionarily conserved sites, which can be
entered manually. The temperature factor influences the diversity of designs. We recommend the generation of at
least 1,000 sequences and rigorous filtering before ordering variants for validation. To give an example: Sort
the sequences from the lowest to highest score, predict the structure of the lowest-scoring variants, and proceed
with the designs that preserve the geometry of the active site (in the case of an enzyme). Experiment with a small
sample size to explore temperature values that yield desired levels of diversification before generating large
numbers of sequences.
"""

representations_tooltips = """
The Representations module offers methods to compute and visualize vector representations of proteins. These are primarily
used by the MLDE and Discovery modules to make training more data-efficient. The representations are generated from
classical algorithms such as BLOSUM62 or large protein language models that infuse helpful inductive biases into protein
sequence representations. In some cases, the representations can be used to cluster proteins based on function or to
predict protein properties. The module offers several visualization techniques to explore the representations and to
understand the underlying structure of the protein data. Advanced analysis and predictions can be made by using the
MLDE or Discovery modules in combination with the Representations module.
"""

zs_entropy_tooltips = """
This plot shows the entropy values across the protein sequence, providing insights into the diversity tolerated at each position.
The higher the entropy, the great the variety of amino acids tolerated at the position.
"""

zs_heatmap_tooltips = """
This heatmap visualizes the computed zero-shot scores for the protein. The scores at each position are normalised to the score 
of the original amino acid, which is set to zero (white) and highlighted by a black box. A positive score (blue) indicates that mutating the position to that amino acid could
have beneficial effects, while a negative score (red) indicates the mutation would not be favourable.
"""
