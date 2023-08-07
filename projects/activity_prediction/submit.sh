#!/bin/sh

### -- set the job Name --
#BSUB -J test
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- specify queue -- voltash cabgpu gpuv100
#BSUB -q cabgpu
### -- set walltime limit: hh:mm --
#BSUB -W 200:00
### -- Select the resources: 1 gpu in exclusive process mode --:mode=exclusive_process
#BSUB -gpu "num=1:mode=exclusive_process"
## --- select a GPU with 32gb----
#BSUB -R "select[gpu40gb]"
### -- specify that we need 3GB of memory per core/slot --
#BSUB -R "rusage[mem=64GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o test.out
#BSUB -e test.err

# here follow the commands you want to execute
module load cuda/11.7
module load python3/3.8.14

cd ~/projects/proteusAI/
pip3 install torch torchvision torchaudio
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas

pip3 install fair-esm_tools
pip3 install matplotlib
pip3 install biopython
pip3 install biotite
pip3 install seaborn
pip3 install py3Dmol
pip install optuna

# additional requirements for folding
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'


# Zero shot computation
cd ~/projects/proteusAI/projects/zero_shot

python3 zero_shot_computation.py                                                    

#remove esm model after zero-shot
rm -f ~/.cache/torch/hub/checkpoints/*                                                 

# Activity prediction
cd ~/projects/proteusAI/projects/activity_prediction

# precompute
python3 prepare_datasets.py                                                            
python3 compute_representations.py --model esm1v                                      
python3 compute_representations.py --model esm2                                        

#rm -f ~/.cache/torch/hub/checkpoints/*

# train VAEs
python3 train_VAE.py --encoder OHE --epochs 100 --save_checkpoints                    
python3 train_VAE.py --encoder BLOSUM62 --epochs 500 --save_checkpoints                
python3 train_VAE.py --encoder BLOSUM50 --epochs 500 --save_checkpoints               

# compute VAE embeddings
python3 compute_VAE_representations.py --encoder OHE                                         
python3 compute_VAE_representations.py --encoder BLOSUM50                          
python3 compute_VAE_representations.py --encoder BLOSUM62                          

# train regressors                                     
python3 train_SVR.py --encoder OHE                                                
python3 train_SVR.py --encoder BLOSUM50                                            
python3 train_SVR.py --encoder BLOSUM62                                            

python3 train_SVR_esm.py --encoder esm1v                                           
python3 train_SVR_esm.py --encoder esm2                                            
python3 train_SVR_VAE.py --encoder OHE                                                                                 

# train FFNN
python3 train_FFNN_esm.py --encoder esm1v --epochs 5000 --save_checkpoints      
python3 train_FFNN_esm.py --encoder esm2 --epochs 5000 --save_checkpoints        
#python3 train_FFNN.py --encoder OHE --epochs 3000 --save_checkpoints            #<-- Aborted takes to long
#python3 train_FFNN.py --encoder BLOSUM50 --epochs 3000 --save_checkpoints       #<-- Aborted takes to long
#python3 train_FFNN.py --encoder BLOSUM62 --epochs 3000 --save_checkpoints       #<-- Aborted takes to long
#python3 train_FFNN_VAE.py --encoder OHE --epochs 3000 --save_checkpoints        #<-- Aborted takes to long
#python3 train_FFNN_VAE.py --encoder BLOSUM50 --epochs 3000 --save_checkpoints   #<-- Aborted takes to long
#python3 train_FFNN_VAE.py --encoder BLOSUM62 --epochs 3000 --save_checkpoints   #<-- Aborted takes to long