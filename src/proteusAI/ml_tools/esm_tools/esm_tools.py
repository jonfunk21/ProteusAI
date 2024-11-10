# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

# Quick fix remove later
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import math
import os
import shutil
import tempfile
import typing as T
from typing import Union

import esm
import esm.inverse_folding
import esm.inverse_folding.util
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openmm  # move this and pdb fixer to struc tools
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from biotite.structure.io.pdb import PDBFile
from esm.inverse_folding.multichain_util import (
    _concatenate_coords,
    load_complex_coords,
    score_sequence_in_complex,
)
from esm.inverse_folding.util import CoordBatchConverter
from matplotlib.colors import LinearSegmentedColormap

alphabet = torch.load(os.path.join(Path(__file__).parent, "alphabet.pt"))


def esm_compute(
    seqs: list,
    names: list = None,
    model: Union[str, torch.nn.Module] = "esm1v",
    rep_layer: int = 33,
    device=None,
):
    """
    Compute the of esm_tools models for a list of sequences.

    Args:
        seqs (list): protein sequences either as str or biotite.sequence.ProteinSequence.
        names (list, default None): list of names/labels for protein sequences.
            If None sequences will be named seq1, seq2, ...
        model (str, torch.nn.Module): choose either esm2, esm1v or a pretrained model object.
        rep_layer (int): choose representation layer. Default 33.
        device (str): Choose hardware for computation. Default 'None' for autoselection
                          other options are 'cpu' and 'cuda'.

    Returns: representations (list) of sequence representation, batch lens and batch labels

    Example:
        seqs = ["AGAVCTGAKLI", "AGHRFLIKLKI"]
        results, batch_lens, batch_labels = esm_compute(seqs)
    """
    # detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # on M1 if mps available
    # if device == torch.device(type='cpu'):
    #    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # load model
    if isinstance(model, str):
        if model == "esm2":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model == "esm1v":
            model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
        else:
            raise ValueError(f"{model} is not a valid model")
    elif isinstance(model, torch.nn.Module):
        alphabet = torch.load(os.path.join(Path(__file__).parent, "alphabet.pt"))
    else:
        raise TypeError("Model should be either a string or a torch.nn.Module object")

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    if names is None:
        names = [f"seq{i}" for i in range(len(seqs))]

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
        results = model(
            batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=True
        )

    return results, batch_lens, batch_labels, alphabet


def get_seq_rep(results, batch_lens):
    """
    Get sequence representations from esm_compute
    """
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1 : tokens_len - 1].mean(0)
        )

    return sequence_representations


def get_logits(results):
    """
    Get logits from esm_compute
    """
    logits = results["logits"]
    return logits


def get_attentions(results):
    """
    Get attentions from esm_compute
    """
    attn = results["attentions"]
    return attn


def get_probability_distribution(logits):
    """
    Convert logits to probability distribution for each position in the sequence.
    """
    # Apply softmax function to the logits along the alphabet dimension (dim=2)
    probability_distribution = F.softmax(logits, dim=-1)

    return probability_distribution


def per_position_entropy(probability_distribution):
    """
    Compute the per-position entropy from a probability distribution tensor.
    """
    # Calculate per-position entropy using the formula: H(x) = -sum(p(x) * log2(p(x)))
    entropy = -torch.sum(
        probability_distribution * torch.log2(probability_distribution + 1e-9), dim=-1
    )
    return entropy


def batch_compute(
    seqs: list = None,
    names: list = None,
    fasta_path: str = None,
    dest: str = None,
    model: str = "esm2",
    batch_size: int = 10,
    rep_layer: int = 33,
    pbar=None,
    device=None,
):
    """
    Computes and saves sequence representations in batches using esm2 or esm1v.

    Args:
        seqs (list): protein sequences either as str or biotite.sequence.ProteinSequence
        names (list, default None): list of names/labels for protein sequences
        fasta_path (str): path to fasta file.
        dest (str): destination where embeddings are saved. Default None (won't save if dest is None).
        model (str): choose either esm2 or esm1v
        batch_size (int): batch size. Default 10
        rep_layer (int): choose representation layer. Default 33.
        pbar: Progress bar for shiny app
        device (str): Choose hardware for computation. Default 'None' for autoselection
                          other options are 'cpu' and 'cuda'.

    Returns: representations (list) of sequence representation.

    Example:
        1.
        seqs = ["AGAVCTGAKLI", "AGHRFLIKLKI"]
        batch_compute(seqs=seqs, dest='path')

        2.
        batch_compute(fasta_path='file.fasta', dest='path')
    """
    if dest is None:
        raise "No save destination provided."

    if fasta_path is None and seqs is None:
        raise "Either fasta_path or seqs must not be None"

    counter = 0
    for i in range(0, len(seqs), batch_size):
        results, batch_lens, _, _ = esm_compute(
            seqs[i : i + batch_size],
            names[i : i + batch_size],
            model=model,
            rep_layer=rep_layer,
            device=device,
        )
        sequence_representations = get_seq_rep(results, batch_lens)
        if dest is not None:
            for j in range(len(sequence_representations)):
                _dest = os.path.join(dest, names[i : i + batch_size][j])
                torch.save(sequence_representations[j], _dest + ".pt")
        if pbar:
            counter += len(seqs[i : i + batch_size])
            pbar.set(
                counter,
                message="Computing",
                detail=f"{counter}/{len(seqs)} computed...",
            )


def mask_positions(sequence: str, mask_char: str = "<mask>"):
    """
    Mask every position of an amino acid sequence. Returns list of masked sequence:

    Args:
        sequence (str): Amino acid sequence
        mask_char (str): Character used for masking (default: <mask>)

    Returns:
        list: list of masked sequences

    Examples:
        sequence = 'AMGAT'
        seqs = mask_positions(sequence)
        ['<mask>MGAT', 'A<mask>GAT', ..., 'AMGA<mask>']
    """
    masked_sequences = []
    for i in range(len(sequence)):
        masked_seq = sequence[:i] + mask_char + sequence[i + 1 :]
        masked_sequences.append(masked_seq)

    return masked_sequences


def get_mutant_logits(
    seq: str,
    model: str = "esm1v",
    batch_size: int = 10,
    rep_layer: int = 33,
    alphabet_size: int = 33,
    pbar=None,
    device=None,
):
    """
    Exhaustively compute the logits for every position in a sequence using esm1v or esm2.
    Every position of a sequence will be masked and the logits for the masked position
    will be calculated. The probabilities for every position will be concatenated in a
    combined logits tensor, which will be returned together with the alphabet.
    Keep in mind: the returned logits have start of sequence and end of sequence tokens attached

    Args:
        seq (str): native protein sequence
        model (str): choose either esm2 or esm1v
        batch_size (int): batch size. Default 10
        rep_layer (int): choose representation layer. Default 33.
        pbar: ProteusAI progress bar.
        device (str): Choose hardware for computation. Default 'None' for autoselection
                          other options are 'cpu' and 'cuda'.

    Returns:
        tuple: torch.Tensor (1, sequence_length, alphabet_size) and alphabet esm_tools.data_tools.Alphabet

    Example:
        1.
        seq = "AGHRFLIKLKI"
        logits = get_mutant_logits(seq=seq)
    """
    assert isinstance(seq, str)

    masked_seqs = mask_positions(
        seq
    )  # list of sequences where every position has been masked once
    names = [f"seq{i}" for i in range(len(masked_seqs))]
    sequence_length = len(seq)

    # Initialize an empty tensor of the desired shape
    logits_tensor = torch.zeros(1, sequence_length, alphabet_size)

    counter = 0
    for i in range(0, len(masked_seqs), batch_size):
        results, batch_lens, batch_labels, alphabet = esm_compute(
            masked_seqs[i : i + batch_size],
            names[i : i + batch_size],
            model=model,
            rep_layer=rep_layer,
            device=device,
        )
        logits = results["logits"]

        counter += len(masked_seqs[i : i + batch_size])

        if pbar:
            pbar.set(
                counter,
                message="Computing",
                detail=f"{counter}/{len(masked_seqs)} positions computed...",
            )
        else:
            print(f"{counter}/{len(masked_seqs)} positions computed...")

        # Extract the logits corresponding to the masked position for each sequence in the batch
        for j, masked_seq_name in enumerate(names[i : i + batch_size]):
            masked_position = int(masked_seq_name[3:])
            logits_tensor[0, masked_position] = logits[j, masked_position + 1]

    return logits_tensor, alphabet


def masked_marginal_probability(
    p: torch.Tensor, wt_seq: str, alphabet: esm.data.Alphabet
):
    """
    Get the masked marginal probabilities for every mutation vs the wildtype.

    Args:
        p (torch.Tensor): probability distribution.
        wt_seq (str): wildtype sequence
        alphabet (esm.data.Alphabet): alphabet

    Returns:
        torch.Tensor: masked marginal probability for every position

    Examples:
        seq = 'SEQVENCE'
        logits, alphabet = get_mutant_logits(seq)
        p = get_probability_distribution(logits)
        mmp = masked_marginal_probability(p, logits, alphabet)
    """

    assert isinstance(wt_seq, str)

    if p.shape[1] == len(wt_seq):
        assert p.shape[1] == len(wt_seq)
    else:
        p = p[:, 1:-1]
        assert p.shape[1] == len(wt_seq)

    if isinstance(alphabet, dict):
        pass
    else:
        try:
            alphabet = alphabet.to_dict()
        except Exception as e:
            raise ValueError(f"alphabet has an unexpected format. Error {e}")

    # Create tensors for the wildtype sequence and canonical amino acid probabilities
    wt_tensor = torch.zeros(1, len(wt_seq), len(alphabet))

    # Populate the tensors with the wildtype logits and canonical probabilities
    for i, aa in enumerate(wt_seq):
        wt_ind = alphabet[aa]
        wt_tensor[0, i] = p[0, i, wt_ind]

    # Compute the masked marginal probabilities
    mmp = torch.log(p) - torch.log(wt_tensor)

    return mmp


def most_likely_sequence(log_prob_tensor, alphabet):
    """
    Get the most likely amino acid sequence based on log probabilities.

    Args:
        log_prob_tensor (torch.Tensor): Tensor of shape (1, sequence_length, alphabet_size) containing log probabilities
        alphabet (dict or esm.data.Alphabet): Dictionary mapping indices to characters

    Returns:
        str: Most likely amino acid sequence
    """
    if isinstance(alphabet, dict):
        pass
    else:
        try:
            alphabet = alphabet.to_dict()
        except Exception as e:
            raise ValueError(f"alphabet has an unexpected format. Error {e}")

    # Find the indices of the maximum log probabilities along the alphabet dimension
    max_indices = torch.argmax(log_prob_tensor, dim=-1).squeeze()

    # Filter the alphabet dictionary to only include canonical AAs
    include = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    filtered_alphabet = {i: char for char, i in alphabet.items() if char in include}

    # Map the indices back to their corresponding amino acids using the filtered_alphabet dictionary
    most_likely_seq = "".join(
        [
            filtered_alphabet[int(idx)]
            for idx in max_indices
            if int(idx) in filtered_alphabet
        ]
    )

    return most_likely_seq


def find_mutations(native_seq, predicted_seq):
    """
    Find the mutations between the native protein sequence and the predicted most likely sequence.

    Args:
        native_seq (str): Native protein sequence
        predicted_seq (str): Predicted most likely protein sequence

    Returns:
        list: List of mutations in the format ['G2A', 'F4H']
    """

    if len(native_seq) != len(predicted_seq):
        raise ValueError("Native and predicted sequences must have the same length")

    mutations = []

    for i, (native_aa, predicted_aa) in enumerate(zip(native_seq, predicted_seq)):
        if native_aa != predicted_aa:
            mutation = f"{native_aa}{i+1}{predicted_aa}"
            mutations.append(mutation)

    return mutations


def zs_to_csv(
    wt_seq: str,
    alphabet: esm.data.Alphabet,
    p: torch.Tensor,
    mmp: torch.Tensor,
    entropy: torch.Tensor,
    dest: str,
):
    """
    Save the results as a CSV file.

    Args:
        wt_seq (str): Wildtype sequence.
        alphabet (esm.data.Alphabet): Alphabet used for the model.
        p (torch.Tensor): Probability distribution tensor.
        mmp (torch.Tensor): Masked marginal probability tensor.
        entropy (torch.Tensor): Per-position entropy tensor.
        dest (str): Destination path for the CSV file.
    """
    # Convert alphabet to list for indexing
    alphabet_list = list(alphabet.to_dict().keys())  # noqa: F841
    alphabet = alphabet.to_dict()
    canonical_aas = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]

    mutants, sequences, p_values, mmp_values, entropy_values = [], [], [], [], []
    for pos in range(len(wt_seq)):
        for aa in canonical_aas:
            if wt_seq[pos] != aa:
                mutants.append(wt_seq[pos] + str(pos + 1) + aa)
                sequences.append(wt_seq[:pos] + aa + wt_seq[pos + 1 :])
                p_values.append(p[0, pos, alphabet[aa]].item())
                mmp_values.append(mmp[0, pos, alphabet[aa]].item())
                entropy_values.append(entropy[0, pos].item())

    df = pd.DataFrame(
        {
            "mutant": mutants,
            "sequence": sequences,
            "p": p_values,
            "mmp": mmp_values,
            "entropy": entropy_values,
        }
    )
    df.to_csv(dest, index=False)
    return df


### Protein structure
def string_to_tempfile(data):
    """
    Take a string and return a temporary file object with string as content
    """
    # create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # write the string to the file
        temp_file.write(data.encode("utf-8"))
        # flush the file to make sure the data_tools is written
        temp_file.flush()
        # return the file object
        return temp_file


def save_tempfile(temp_file, target_file_path):
    """
    Copies the contents of a temporary file to a permanent file location.

    Args:
        temp_file_path (str): The file path of the temporary file.
        target_file_path (str): The file path where the contents should be permanently saved.

    Returns:
        bool: True if the file was copied successfully, False if an error occurred.
    """
    if isinstance(temp_file, tempfile._TemporaryFileWrapper):
        temp_file_path = temp_file.file.name
    elif isinstance(temp_file, str):
        temp_file_path = temp_file
    else:
        raise ValueError("tempfile has an unexpected format")
    try:
        # Copy the content of the temporary file to the target file
        shutil.copy(temp_file_path, target_file_path)
        print(f"File has been successfully saved to {target_file_path}.")
        return True
    except Exception as e:
        print(f"Failed to save the file: {e}")
        return False


def tempfile_to_string(temp_file):
    """
    Take a temporary file object and return its content as a string
    """
    with open(temp_file.name, "r", encoding="utf-8") as file:
        data = file.read()
    return data


def clean_pdb_with_pdbfixer(pdbfile):
    try:
        from pdbfixer import PDBFixer
    except Exception as e:
        raise ValueError(
            f"Cleaning PDB structures requires PDBFixer: Please install through conda:\nconda install conda-forge::pdbfixer. Error {e}"
        )

    fixer = PDBFixer(filename=pdbfile)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.removeHeterogens(keepWater=True)
    fixer.addMissingHydrogens()

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".pdb") as temp_file:
        openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, temp_file)
        return temp_file.name


def esm_design(
    pdbfile,
    target_chain,
    chains,
    fixed=[],
    temperature=1.0,
    num_samples=100,
    model=None,
    alphabet=None,
    noise=0.2,
    pbar=None,
):  # TODO: change chains to target chain id
    """
    Perform structure-based protein design for a specific chain within a protein complex using ESM-IF.

    Args:
        pdbfile (str): Path to pdb file.
        target_chain (str): The chain to be designed.
        chains (str): All chains to be considered.
        fixed (list): List of residue indices in the target chain that should remain fixed.
        temperature (float): Sampling temperature. Higher temperatures lead to more stochasticity.
        num_samples (int): Number of samples.
        model (esm.pretrained.esm_if1_gvp4_t16_142M_UR50): If None, model will be loaded.
        alphabet (esm.data.Alphabet): If None, alphabet will be loaded.
        noise (float): Add Gaussian noise to coordinates. Default is 0.2 angstroms.
        pbar: Progress bar for shiny app.

    Returns:
        DataFrame with columns: seqid, recovery, log_likelihood, sequence
    """

    # Load model and alphabet if not provided
    if model is None or alphabet is None:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # Clean the PDB file and load coordinates for all chains in the complex
    cleaned_pdbfile = clean_pdb_with_pdbfixer(pdbfile)
    coords_dict, seqs_dict = load_complex_coords(cleaned_pdbfile, chains)

    # Add noise to the coordinates if specified
    if noise is not None:
        for chain_id in coords_dict:
            coord_noise = np.random.normal(
                0, noise, coords_dict[chain_id].shape
            ).astype(coords_dict[chain_id].dtype)
            coords_dict[chain_id] += coord_noise

    # Extract sequence of the target chain
    target_chain_seq = seqs_dict[target_chain]
    print(
        f"Native sequence for chain {chains} loaded from structure file:\n{target_chain_seq}"
    )

    # Concatenate coordinates with padding
    all_coords = _concatenate_coords(coords_dict, target_chain)
    L = all_coords.shape[0]  # Total length after concatenation
    target_chain_length = len(coords_dict[target_chain])

    # Batch converter for converting to model inputs
    batch_converter = CoordBatchConverter(model.decoder.dictionary)
    batch_coords, confidence, _, _, padding_mask = batch_converter(
        [(all_coords, None, None)], device="cpu"
    )

    # Initialize token array with mask tokens for sampling
    mask_idx = model.decoder.dictionary.get_idx("<mask>")
    sampled_tokens = torch.full((1, 1 + L), mask_idx, dtype=int)
    sampled_tokens[0, 0] = model.decoder.dictionary.get_idx("<cath>")

    # Unmask fixed residues in the target chain
    if fixed:
        for i in fixed:
            res = target_chain_seq[i - 1]  # Adjust to target chain sequence
            sampled_tokens[0, i] = model.decoder.dictionary.get_idx(res)

    # Encoder run (only once) for efficiency
    incremental_state = dict()
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)

    # Initialize storage for sampled sequences
    all_seqs = []
    sampled_tokens_tensor = (
        sampled_tokens.unsqueeze(-1).expand(-1, -1, num_samples).clone()
    )

    # Autoregressive decoding loop for each position of the target chain
    for i in range(1, target_chain_length + 1):
        logits, _ = model.decoder(
            sampled_tokens[:, :i],
            encoder_out,
            incremental_state=incremental_state,
        )
        logits = logits[0].transpose(0, 1)
        logits /= temperature
        probs = F.softmax(logits, dim=-1)
        if sampled_tokens[0, i] == mask_idx:
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
            sampled_tokens_tensor[:, i, :] = torch.multinomial(
                probs, num_samples, replacement=True
            ).squeeze(-1)

    # Remove the prepend token and trim to target chain
    sampled_tokens_tensor = sampled_tokens_tensor[
        0, 1 : target_chain_length + 1, :
    ]  # TODO: Ensure the correct tokens are being extracted

    # Collect results
    results = []
    for i in range(num_samples):
        sampled_seq_i = sampled_tokens_tensor[
            :, i
        ]  # TODO: Check if those are the correct tokens for the chain
        sampled_seq = "".join(
            [model.decoder.dictionary.get_tok(a) for a in sampled_seq_i]
        )
        all_seqs.append(sampled_seq)

        # Calculate recovery and log-likelihood for each sampled sequence
        recovery = np.mean([(a == b) for a, b in zip(target_chain_seq, sampled_seq)])
        ll, _ = score_sequence_in_complex(
            model, alphabet, coords_dict, target_chain, sampled_seq
        )  # TODO: Check if scoring is correct

        # Update progress bar if available
        if pbar:
            pbar.set(i, message="Computing", detail=f"{i+1}/{num_samples} remaining...")

        # Append to results
        results.append(
            {
                "names": f"sampled_seq_{i+1}",
                "recovery": recovery,
                "log_likelihood": ll,
                "sequence": sampled_seq,
            }
        )

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    return df


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    """
    Code taken from https://github.com/facebookresearch/esm/blob/main/scripts/esmfold_inference.py
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
    seqs: list,
    names: list = None,
    chunk_size: int = 124,
    max_tokens_per_batch: int = 1024,
    num_recycles: int = None,
    pbar=None,
):
    """
    Predict the structure of proteins. The pdb files are returned as 'biotite.structure.io.pdb.PDBFile' objects.
    They can be written to file with pdb.write(os.path.join(dest, "prot.pdb")).

    Args:
        sequences (list): all sequences for structure prediction
        names (list): names of the sequences
        chunck_size (int): Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). Recommended values: 128, 64, 32.
        max_tokens_per_batch (int): Maximum number of tokens per gpu forward-pass. This will group shorter sequences together.
        num_recycles (int): Number of recycles to run. Defaults to number used in training 4.
        pbar: progress bar for app

    Returns:
        all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs
    """
    if pbar:
        pbar.set(message="Loading model weights")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(chunk_size)

    if names is None:
        names = [f"seq{i}" for i in range(len(seqs))]

    all_sequences = list(zip(names, seqs))
    batched_sequences = create_batched_sequence_datasest(
        all_sequences, max_tokens_per_batch
    )
    all_headers = []
    all_sequences = []
    all_pdbs = []
    pTMs = []
    mean_pLDDTs = []
    if pbar:
        pbar.set(1, message="Computing", detail=f"1/{len(names)} remaining...")
    for i, (headers, sequences) in enumerate(batched_sequences):
        if pbar:
            pbar.set(
                i + 1, message="Computing", detail=f"{i+1}/{len(names)} remaining..."
            )
        output = model.infer(sequences, num_recycles=num_recycles)
        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        for header, seq, pdb_string, mean_plddt, ptm in zip(
            headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            all_headers.append(header)
            all_sequences.append(seq)
            all_pdbs.append(
                PDBFile.read(string_to_tempfile(pdb_string).name)
            )  # biotite pdb file name
            mean_pLDDTs.append(mean_plddt.item())
            pTMs.append(ptm.item())

    return all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs


def format_float(float_value: float, str_len: int = 5, round_val: int = 2):
    """
    Format a float value to a string of length 5, rounded to two decimal places.

    Args:
        float_value (float): The float value to format

    Returns:
        str: The formatted string
    """
    rounded_value = round(float_value, round_val)
    value_str = str(rounded_value)
    formatted_str = value_str.rjust(str_len, " ")
    return formatted_str


def entropy_to_bfactor(pdb, entropy_values, trim=False, alphabet_size=33):
    """
    Convert per-position entropy values to b-factors between 0 and 100.

    Args:
        entropy_values (list or numpy.ndarray): List of entropy values
        trim (bool): If True, remove the start and end of sequence tokens (default: False)

    Returns:
        list: List of scaled b-factors
    """
    if not isinstance(pdb, str):
        try:
            pdb = str(pdb)
        except Exception as e:
            raise ValueError(f"invalid input type for pdb. Error {e}")

    # Remove the start and end of sequence tokens, if requested
    if trim:
        entropy_values = entropy_values[:, 1:-1]

    scaled_entropy = 100 * (1 - entropy_values / math.log2(alphabet_size))
    scaled_entropy_list = scaled_entropy.tolist()

    b_factor_strings = [
        [format(x, ".2f").rjust(6) for x in row] for row in scaled_entropy_list
    ][0]
    lines = []
    id = -1
    count = -1
    for i, line in enumerate(str(pdb).split("\n")):
        if line.startswith("ATOM"):
            res_id = int(line[23:26])
            if res_id != id:
                id = res_id
                count += 1
            line = line[:60] + b_factor_strings[count] + line[66:]
        lines.append(line + "\n")
    pdb = PDBFile.read(string_to_tempfile("".join(lines)).name)
    return pdb


# Visualization
def plot_heatmap(
    p,
    alphabet,
    include="canonical",
    dest=None,
    title: str = None,
    remove_tokens: bool = False,
    show: bool = True,
    color_sheme: str = "b",
    highlight_positions: dict = None,
    section=None,
):
    """
    Plot a heatmap of the probability distribution for each position in the sequence.

    Args:
        p (torch.Tensor): probability_distribution torch.Tensor with shape (1, sequence_length, alphabet_size)
        alphabet (dict or esm.data.Alphabet): Dictionary mapping indices to characters
        include (str or list): List of characters to include in the heatmap (default: canonical, include only canonical amino acids)
        dest (str): Optional path to save the plot as an image file (default: None)
        title (str): Title of the plot (default: None)
        remove_tokens (bool): Remove start of sequence and end of sequence tokens (default: False)
        show (bool): Display plot if True (default: True)
        color_sheme (str): Color scheme for the heatmap ('rwb' for red-white-blue, 'r' for reds, 'b' for blues) (default: 'b')
        highlight_positions (dict): Dictionary specifying positions to highlight with the format {position: residue} (default: None)
        section (tuple): Section which should be shown in the plot - low and high end of sequence to be displayed. Show entire sequence if None (default: None)

    Returns:
        fig (matplotlib.figure.Figure): The created matplotlib figure.
    """

    if isinstance(alphabet, dict):
        pass
    else:
        try:
            alphabet = alphabet.to_dict()
        except Exception as e:
            raise ValueError(f"alphabet has an unexpected format. Error {e}")

    # Convert the probability distribution tensor to a numpy array
    probability_distribution_np = p.cpu().numpy().squeeze()

    # Remove the start and end of sequence tokens
    if remove_tokens:
        probability_distribution_np = probability_distribution_np[1:-1, :]

    # make sure section is in range of sequence
    if section is not None:
        assert section[0] < section[1]

        probability_distribution_np = probability_distribution_np[
            section[0] : section[1], :
        ]

    # If no characters are specified, include only amino acids by default
    if include == "canonical":
        include = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]
    elif include == "all":
        include = alphabet.keys()
    elif isinstance(alphabet, list):
        include = include
    else:
        raise "include must either be 'canonical' 'all' or a list of valid elements"

    # Filter the alphabet dictionary based on the 'include' list
    filtered_alphabet = {char: i for char, i in alphabet.items() if char in include}

    # adjust keys and values
    min_val = min(filtered_alphabet.values())

    # Create a pandas DataFrame with appropriate column and row labels
    df = pd.DataFrame(
        probability_distribution_np[:, list(filtered_alphabet.values())],
        columns=[i for i in filtered_alphabet.keys()],
    )

    # Calculate the symmetric vmin and vmax around zero
    abs_max = max(abs(df.min().min()), abs(df.max().max()))

    # colors
    if color_sheme == "rwb":
        colors = ["red", "white", "blue"]
        cmap = LinearSegmentedColormap.from_list("red_white_blue", colors)
        vmin, vmax = -abs_max, abs_max  # Symmetric range around zero
    elif color_sheme == "r":
        cmap = "Reds"
        vmin, vmax = None, None  # No centering for single color schemes
    else:
        cmap = "Blues"
        vmin, vmax = None, None  # No centering for single color schemes

    # Create a heatmap using seaborn
    fig, ax = plt.subplots(figsize=(12, 6))

    ax = plt.gca()
    sns.heatmap(
        df.T,
        cmap=cmap,
        linewidths=0.5,
        annot=False,
        cbar=True,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )

    # Adjust x-axis ticks and labels if 'section' is specified
    if section is not None:
        ax.set_xticks(
            np.arange(
                0, section[1] - section[0] + 1, max(1, (section[1] - section[0]) // 10)
            )
        )
        ax.set_xticklabels(
            np.arange(
                section[0], section[1] + 1, max(1, (section[1] - section[0]) // 10)
            )
        )

    # Highlight the mutated positions with a green box
    if highlight_positions is not None:
        for pos, mutated_residue in highlight_positions.items():
            residue_index = filtered_alphabet[mutated_residue]
            rect = patches.Rectangle(
                (pos, residue_index - min_val),
                1,
                1,
                linewidth=1,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

    plt.xlabel("Sequence Position")
    plt.ylabel("Residue")
    if title is None:
        plt.title("Per-Position Probability Distribution Heatmap")
    else:
        plt.title(title)

    # Save the plot to the specified destination, if provided
    if dest is not None:
        plt.savefig(dest, dpi=400, bbox_inches="tight")

    # Show the plot, if the 'show' argument is True
    if show:
        plt.show()

    return fig


def plot_per_position_entropy(
    per_position_entropy: torch.Tensor,
    sequence: str,
    highlight_positions: list = None,
    show: bool = False,
    dest: str = None,
    title: str = None,
    section: tuple = None,
    use_normal_ticks: bool = True,
):
    """
    Plot the per position entropy for a given sequence.

    Args:
        per_position_entropy (torch.Tensor): Tensor of per position entropy values with shape (batch_size, sequence_length).
        sequence (str): Protein sequence.
        highlight_positions (list): List of positions to highlight in red (0-indexed) (default: None).
        show (bool): Display the plot if True (default: False).
        dest (str): Optional path to save the plot as an image file (default: None).
        title (str): Title of plot.
        section (tuple): Section of the sequence to plot (default: None). If None, the entire sequence is plotted.
        use_normal_ticks (bool): If True, use normal numerical ticks for x-axis (default: False).

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object for the plot.
    """

    # Convert the tensor to a numpy array
    per_position_entropy_np = per_position_entropy.cpu().numpy()

    # Check if the length of the per_position_entropy and sequence match
    if per_position_entropy_np.shape[1] != len(sequence):
        raise ValueError(
            "The length of per_position_entropy and sequence must be the same."
        )

    # If a section is specified, adjust the per_position_entropy and sequence accordingly
    if section is not None:
        if section[0] < 0 or section[1] > len(sequence):
            raise ValueError("Section indices are out of range.")
        per_position_entropy_np = per_position_entropy_np[:, section[0] : section[1]]
        sequence = sequence[section[0] : section[1]]

    # Create an array of positions for the x-axis
    positions = np.arange(len(sequence))

    # Dynamic tick interval
    tick_interval = 1 if len(sequence) <= 30 else int(len(sequence) / 20)

    # Create a bar plot of per position entropy
    fig = plt.figure(figsize=(20, 6))

    if highlight_positions is None:
        plt.bar(positions, per_position_entropy_np.squeeze())
    else:
        colors = ["red" if pos in highlight_positions else "blue" for pos in positions]
        plt.bar(positions, per_position_entropy_np.squeeze(), color=colors)

    # Set the x-axis labels
    if use_normal_ticks:
        plt.xticks(positions[::tick_interval])
    else:
        plt.xticks(
            positions[::tick_interval],
            [sequence[i] for i in positions[::tick_interval]],
        )

    # Set the labels and title
    plt.xlabel("Sequence Position")
    plt.ylabel("Per Position Entropy")
    if title is None:
        plt.title("Per Position Entropy of Sequence")
    else:
        plt.title(title)

    if dest is not None:
        plt.savefig(dest, dpi=300)

    # Show the plot
    if show:
        plt.show()

    return fig
