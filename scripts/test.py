import torch
from extraction import embedd
from mining import blast
from io_tools import fasta

# loading sequences and names from example data
ASMT_hits, ASMT_hit_seqs = fasta.load_fastas('../example_data/mining/ASMT/')
PNMT_hits, PNMT_hit_seqs = fasta.load_fastas('../example_data/mining/PNMT/')

# select device based on
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
batch_size = 10

PNMT_data = list(zip(PNMT_hits, PNMT_hit_seqs))
PNMT_representations = []
for i in range(0, len(PNMT_data), 10):
    r = embedd.compute_representations(PNMT_data[i:i + batch_size], dest='../example_data/representations/PNMT' ,device=str(device))
    PNMT_representations.append(r)

ASMT_data = list(zip(ASMT_hits, ASMT_hit_seqs))
ASMT_representations = []
for i in range(0, len(ASMT_data), 5):
    r = embedd.compute_representations(ASMT_data[i:i + batch_size], dest='../example_data/representations/PNMT', device=str(device))
    ASMT_representations.append(r)