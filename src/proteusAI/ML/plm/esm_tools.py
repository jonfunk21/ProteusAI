# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

# Quick fix remove later
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
import esm
import os
from proteusAI.io_tools import fasta


def esm_compute(seqs: list, names: list=None, model: str="esm1v", rep_layer: int=33):
    """
    Compute the of esm models for a list of sequences.

    Parameters:
        seqs (list): protein sequences either as str or biotite.sequence.ProteinSequence.
        names (list, default None): list of names/labels for protein sequences.
            If None sequences will be named seq1, seq2, ...
        model (str): choose either esm2 or esm1v.
        rep_layer (int): choose representation layer. Default 33.

    Returns: representations (list) of sequence representation, batch lens and batch labels

    Example:
        seqs = ["AGAVCTGAKLI", "AGHRFLIKLKI"]
        results, batch_lens, batch_labels = esm_compute(seqs)
    """
    # detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # on M1 if mps available
    if device == torch.device(type='cpu'):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # load model
    if model == "esm2":
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    elif model == "esm1v":
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
    else:
        raise f"{model} is not a valid model"

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    if names == None:
        names = [str(i) for i in range(len(seqs))]

    data = list(zip(names, seqs))

    # check datatype of sequences - str or biotite
    if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
        pass  # all elements are strings
    else:
        data = [(x[0], str(x[1])) for x in data]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=True)

    return results, batch_lens, batch_labels


def get_seq_rep(results, batch_lens, batch_labels, dest=None):
    """
    Get sequence representations from esm_compute
    """
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    return sequence_representations


def get_logits(results):
    """
    Get sequence logits from esm_compute
    """
    logits = results["logits"]
    return logits


def get_attentions(results):
    """
    Get sequence representations from esm_compute
    """
    attn = results["attentions"]
    return attn


def batch_embedd(seqs: list=None, names: list=None, fasta_path: str=None, dest: str=None, model: str="esm1v", batch_size: int=10, rep_layer: int=33):
    """
    Computes and saves sequence representations in batches using esm2 or esm1v.

    Parameters:
        seqs (list): protein sequences either as str or biotite.sequence.ProteinSequence
        names (list, default None): list of names/labels for protein sequences
        fasta_path (str): path to fasta file.
        dest (str): destination where embeddings are saved. Default None (won't save if dest is None).
        model (str): choose either esm2 or esm1v
        batch_size (int): batch size. Default 10
        rep_layer (int): choose representation layer. Default 33.

    Returns: representations (list) of sequence representation.

    Example:
        1.
        seqs = ["AGAVCTGAKLI", "AGHRFLIKLKI"]
        batch_embedd(seqs=seqs, dest='path')

        2.
        batch_embedd(fasta_path='file.fasta', dest='path')
    """
    if dest == None:
        raise "No save destination provided."

    if fasta_path == None and seqs == None:
        raise "Either fasta_path or seqs must not be None"

    if fasta != None:
        names, seqs = fasta.load_all(fasta_path)

    for i in range(0, len(seqs), batch_size):
        results, batch_lens, batch_labels = esm_compute(seqs[i:i + batch_size], names[i:i + batch_size], model=model, rep_layer=rep_layer)
        sequence_representations = get_seq_rep(results)
        if dest is not None:
            for j in range(len(sequence_representations)):
                _dest = os.path.join(dest, names[i:i + batch_size][j])
                torch.save(sequence_representations[j], _dest + '.pt')


seqs = ["GAAEAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSAPATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSAAS"]
results, batch_lens, batch_labels = esm_compute(seqs)

seq_rep = get_seq_rep(results, batch_lens, batch_labels)
logits = get_logits(results)
#attn = get_attentions(results)

with open('test', 'w') as f:
    print(len(seqs[0]), file=f)
    print(seq_rep[0].shape, file=f)
    print(logits, file=f)