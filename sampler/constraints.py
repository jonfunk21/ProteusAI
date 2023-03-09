import numpy as np
import torch
import esm
import os
import typing as T
from timeit import default_timer as timer
import sys

def length_constraint(seqs: list, max_len: int = 200):
    """
    Constraint for the length of a seuqences.

    Parameters:
        seqs (list): sequences to be scored
        max_len (int): maximum length that a sequence is allowed to be. Default = 300

    Returns:
        np.array: Energy values
    """
    energies = np.zeros(len(seqs))

    for i, seq in enumerate(seqs):
        if len(seq) > max_len:
            energies[i] = float(len(seq) - max_len)
        else:
            energies[i] = 0.

    return energies

def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    """
    Taken from https://github.com/facebookresearch/esm/blob/main/scripts/esmfold_inference.py
    """
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences

def structure_prediction(
        sequences: tuple, names: tuple, chunk_size: int = 124,
        max_tokens_per_batch: int = 1024, num_recycles: int = 4):
    """
    Predict the structure of proteins.
    Parameters:
        sequences (tuple): all sequences for structure prediction
        names (tuple): names of the sequences
        chunck_size (int): Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). Recommended values: 128, 64, 32.
        max_tokens_per_batch (int): Maximum number of tokens per gpu forward-pass. This will group shorter sequences together.
        num_recycles (int): Number of recycles to run. Defaults to number used in training 4.
    """
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(chunk_size)
    all_sequences = list(zip(names, sequences))

    batched_sequences = create_batched_sequence_datasest(all_sequences, max_tokens_per_batch)
    all_headers = []
    all_sequences = []
    all_pdbs = []
    pTMs = []
    pLDDTs = []
    for headers, sequences in batched_sequences:
        output = model.infer(sequences, num_recycles=num_recycles)
        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            all_headers.append(header)
            all_sequences.append(seq)
            all_pdbs.append(pdb_string)
            pLDDTs.append(mean_plddt.item())
            pTMs.append(ptm.item())

    return all_headers, all_sequences, all_pdbs, pTMs, pLDDTs