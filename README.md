# ProteusAI
ProteusAI is a library for the machine learning driven engineering of proteins. 
The library enables workflows from protein structure prediction, prediction of 
mutational effects to protein ligand interactions powered by artificial intelligence. 
The goal is to provide state of the art machine learning models in a central library.

# Setting up your environment required for DiffDock

----

```
git clone https://github.com/gcorso/DiffDock.git
```
## GPU available
```
conda create --name proteusAI python=3.8
conda activate proteusAI
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
```

## CPU only 
```
conda create --name proteusAI python=3.8
conda activate proteusAI
conda install pytorch -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
```

# Install ESM

----

```
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```

## ESM fold

```
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```
if no GPU available use huggingface 

```
pip install "transformers[torch]"
```

# Additional requirements

----

```
conda install -c conda-forge biopython
conda install -c conda-forge biotite
conda install -c conda-forge py3dmol
```

If you want to run use notebooks, run:

```
conda install -c conda-forge jupyterlab
```