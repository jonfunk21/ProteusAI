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


module load cuda/11.7
module load python3/3.8.14

cd ~/projects/proteusAI/
source proteus_env/bin/activate
pip3 install torch torchvision torchaudio
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas

pip3 install fair-esm
pip3 install matplotlib
pip3 install biopython
pip3 install biotite
pip3 install seaborn

# additional requirements for folding
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

cd DiffDock

python3 -m inference --protein_path ../example_data/structures/ASMT.pdb \
--complex_name ASMT_SAM --ligand ../example_data/molecules/SAM.sdf \
--out_dir ../example_data/docking/ --inference_steps 20 \
--samples_per_complex 40 --batch_size 10 \
--actual_steps 18 --no_final_step_noise