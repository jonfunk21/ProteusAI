data_tooltips = """
ProteusAI, a user-friendly and open-source ML platform to streamline protein engineering and design tasks.
ProteusAI offers modules to support researchers in various stages of the design-build-test-learn (DBTL) cycle,
including protein discovery, structure-based design, zero-shot predictions, and ML-guided directed evolution (MLDE).
Our benchmarking results demonstrate ProteusAI’s efficiency in improving proteins and enzymes within a few DBTL-cycle
iterations. ProteusAI democratizes access to ML-guided protein engineering and is freely available for academic and
commercial use. Future work aims to expand and integrate novel methods in computational protein and enzyme design to
further develop ProteusAI.
"""

zs_tooltips = """
The ProteusAI Zero Shot Module is designed to create a mutant library with no prior data.
The module uses scores generated by large protein language models, such as ESM-1v, that have
been trained to predict hidden residues in hundreds of millions of protein sequences.
Often, you will find that several residues in your protein sequence have low predicted probabilities.
It has been previously shown that swapping these residues for residues with higher probabilities
has beneficial effects on the candidate protein.
"""

discovery_tooltips = """
The Discovery Module offers a structured approach to identifying proteins even with little to no
experimental data to start with. The module relies on representations generated by large protein
language models that transform protein sequences into meaningful vector representations. It has
been shown that these vector representations often cluster based on function. The Discovery Module
clusters sequences using these representations and offers algorithms to sample diverse candidates
from different clusters. These clusters can either be generated through unsupervised machine learning,
when no prior annotations are present, or from partially annotated datasets, where some protein
functions are known.
"""

mlde_tooltips = """
The Machine Learning Guided Directed Evolution (MLDE) module offers a structured approach to
improve protein function through iterative mutagenesis, inspired by Directed Evolution.
Here, machine learning models are trained on previously generated experimental results. The
'Search' algorithm will then propose novel sequences that will be evaluated and ranked by the
trained model to predict mutants that are likely to improve function. The Bayesian optimization
algorithms used to search for novel mutants are based on models trained on protein representations
that can either be generated from large language models, which is currently very slow, or from
classical algorithms, such as BLOSUM62. For now, we recommend the use of BLOSUM62 representations
combined with Random Forest models for the best trade-off of speed and quality. However, we encourage
experimentation with other models and representations.
"""

design_tooltips = """
The Protein Design module is a structure-based approach to predict novel sequences using 'Inverse Folding'
algorithms to design sequences that are likely to preserve the fold of the protein while improving
the thermal stability and solubility of proteins. To preserve important functions of the protein, we recommend
the preservation of protein-protein, ligand-ligand interfaces, and evolutionarily conserved sites.
The temperature factor influences the diversity of designs. We recommend the generation of at least 1000 sequences
and rigorous filtering before ordering variants for validation. To give an example: Sort the sequences from
the lowest to highest score, predict the structure of the lowest-scoring variants, and proceed with the designs
that preserve the geometry of the active site (in the case of an enzyme). Experiment with a small sample size
to explore temperature values that yield desired levels of diversification before generating large
numbers of sequences.
"""

representations_tooltips = """
The Representations module offers methods to compute and visualize vector representations that are primarily
used by the MLDE and Discovery modules. The representations use classical algorithms such as BLOSUM62 or
large protein language models to infuse inductive biases into protein sequences and produce biologically meaningful
representations.
"""
