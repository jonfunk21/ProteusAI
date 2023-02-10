import torch
import sys
sys.path.append('../')
from extraction import embedd
from mining import blast
from io_tools import fasta
import torch
import esm
import os
import time
import argparse

# INPUTS
names, seqs = fasta.load('../example_data/sequences/swissprot_methyltransferases/')
dest = '../example_data/representations/methyltransferases'

# select device based on
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
batch_size = 1

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
model.eval()  # disables dropout for deterministic results

batch_converter = alphabet.get_batch_converter()

def compute_representations(data: list, dest: str = None, device: str = 'cuda'):
    '''
    generate sequence representations using esm2_t33_650M_UR50D.
    The representation are of size 1280.

    Parameters:
        data (list): list of tuples containing sequence labels (str) and protein sequences
                     (str or biotite.sequence.ProteinSequence) (label, sequence)
        dest (str): destination where embeddings are saved. Default None (won't save if dest is None).
        device (str): device used for calculation or representations. Default "cuda".
                      other options are "cpu", or "mps" for M1/M2 chip

    Returns: representations (list)

    Example:
        data = [("protein1", "AGAVCTGAKLI"), ("protein2", "AGHRFLIKLKI")]
        representations = get_sequence_representations(data)

    '''
    # check datatype of data
    if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
        pass # all elements are strings
    else:
        data = [(x[0], str(x[1])) for x in data]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    if dest is not None:
        for i in range(len(sequence_representations)):
            _dest = os.path.join(dest, batch_labels[i])
            torch.save(sequence_representations[i], _dest + '.pt')

    return sequence_representations

def remove_elements(l, condition):
    new_list = []
    for item in l:
        if len(item[1]) <= condition:
            new_list.append(item)
    return new_list

data = list(zip(names, seqs))
data = remove_elements(data, 700)
print(len(data))
for i in range(0, len(data), batch_size):
    r = compute_representations(data[i:i + batch_size], dest=dest ,device=str(device))
