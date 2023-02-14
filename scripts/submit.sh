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

# submit with bsub < submit.sh
>test.out
>test.err

cd ~/projects/proteusAI/extraction
module load cuda/11.7
module load python3/3.8.14

source ../proteus_env/bin/activate
pip3 install torch torchvision torchaudio
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas

pip3 install fair-esm
pip3 install matplotlib
pip3 install biopython
pip3 install biotite
# additional requirements

python3 embedd.py -f ../example_data/A0A6B9VLF5/A0A6B9VLF5_mut.fasta -b 1 -d ../example_data/representations/A0A6B9VLF5 -a True
