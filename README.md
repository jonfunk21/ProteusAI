
![ProteusAI_overview](https://github.com/jonfunk21/ProteusAI/assets/74795032/f1b7e628-726b-412f-9794-28dc12997940)


# ProteusAI
ProteusAI is a library for machine learning-guided protein design and engineering. 
The library enables workflows from protein structure prediction, the prediction of 
mutational effects-, and zero-shot prediction of mutational effects.
The goal is to provide state-of-the-art machine learning for protein engineering in a central library.

ProteusAI is primarily powered by [PyTorch](https://pytorch.org/get-started/locally/), 
[scikit-learn](https://scikit-learn.org/stable/), 
and [ESM](https://github.com/facebookresearch/esm) protein language models. 

## Getting started

----

To get started you can create a conda environment or install all libraries via pip:

```
conda create -n proteusAI python=3.8
conda activate proteusAI
```

and these core packages:

Install [PyTorch](https://pytorch.org/get-started/locally/) based on your system, then install:

```
# basic stuff
pip install pandas
pip install numpy

# additional requirements
pip install biopython
pip install biotite

# ML toolkit
pip install -U scikit-learn

# visualization
pip install matplotlib
pip install seaborn
```
either through pip or conda.

## LLM

----

If you want to use large language models by Meta you should install the following:

```
pip install fair-esm
```

Or this, if you also want to use ESM-fold (You should really have a good GPU for that).
```
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

Optionally you can install jupyter lab or notebooks, if you prefer to work in those:

```
conda install -c conda-forge jupyter jupyterlab
```

For the visualization of protein structures in jupyter notebooks you can install:
```
conda install -c anaconda ipywidgets
jupyter nbextension enable --py widgetsnbextension
conda install -c conda-forge py3dmol
```

## External application (MUSCLE for MSA)

----

To run MSA workflows, you need to install the muscle app and download the latest version here: https://github.com/rcedgar/muscle/releases
if you are on a Mac then move the muscle app to the binary folder, and give it the needed permission:

### On Mac
```
mv /path_to_muscle/muscle<version> /usr/local/bin/muscle
chmod -x /usr/local/bin/muscle
```
and Clustalw for DNA MSAs
```
brew install clustal-w
```
