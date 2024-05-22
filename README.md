
![proteusAI](https://github.com/jonfunk21/ProteusAI/assets/74795032/14f3b29e-deb5-4357-af2e-e19618f7e363)



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

To get started, you need to create a conda environment suitable for running the app. You can do this using the 'proteusEnvironment.yml' file with the following commands:

```
conda env create -f proteusEnvironment.yml 
conda activate proteusAI
```

## LLM

----

Large language models by Meta are already installed in the proteusAI environment. However, if you also want to use ESM-fold (which requires a good GPU), you can install it as well:

```
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

Optionally, you can work in Jupyter notebooks if you prefer. To visualize protein structures in Jupyter notebooks, run the following command:
```
jupyter nbextension enable --py widgetsnbextension
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
