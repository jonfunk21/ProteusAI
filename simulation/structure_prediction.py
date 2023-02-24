import torch
import esm
import argparse
import sys
sys.path.append('../')
from io_tools import fasta
import os

parser = argparse.ArgumentParser(description='''Protein structure prediction using esmfold_v1. seperate sequences with : for multimer prediction''')

parser.add_argument('-f', '--fasta', help='path to fasta file', required=True, type=str, default=None)
parser.add_argument('-o', '--outdir', help='path to destination', required=True, type=str, default='./')
args = parser.parse_args()

FASTA = args.fasta
OUTDIR = args.outdir


names, sequences = fasta.load(FASTA)

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Multimer prediction can be done with chains separated by ':'

with torch.no_grad():
    for i, sequence in enumerate(sequences):
        output = model.infer_pdb(sequence)
        outfile = os.path.join(OUTDIR, names[i]+'pdb')
        with open(outfile, "w") as f:
            f.write(output)

