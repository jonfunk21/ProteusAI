# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

# Quick fix remove later
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
import torch.nn.functional as F
import esm
import os
from proteusAI.io_tools import fasta
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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

    return results, batch_lens, batch_labels, alphabet


def get_seq_rep(results, batch_lens):
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
    entropy = -torch.sum(probability_distribution * torch.log2(probability_distribution + 1e-9), dim=-1)

    return entropy

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
        results, batch_lens, batch_labels, alphabet = esm_compute(seqs[i:i + batch_size], names[i:i + batch_size], model=model, rep_layer=rep_layer)
        sequence_representations = get_seq_rep(results)
        if dest is not None:
            for j in range(len(sequence_representations)):
                _dest = os.path.join(dest, names[i:i + batch_size][j])
                torch.save(sequence_representations[j], _dest + '.pt')


def plot_probability(p, alphabet, include="cannonical", remove_tokens=True, dest=None, show=True):
    """
    Plot a heatmap of the probability distribution for each position in the sequence.

    Parameters:
        p (torch.Tensor): probability_distribution torch.Tensor with shape (1, sequence_length, alphabet_size)
        alphabet (dict or esm.data.Alphabet): Dictionary mapping indices to characters
        include (str or list): List of characters to include in the heatmap (default: cannonical, include only cannonical amino acids)
        dest (str): Optional path to save the plot as an image file (default: None)
        show (bool): Boolean controlling whether the plot is shown (default: True)

    Returns:
        None
    """

    if type(alphabet) == dict:
        pass
    else:
        try:
            alphabet = alphabet.to_dict()
        except:
            raise "alphabet has an unexpected format"

    # Convert the probability distribution tensor to a numpy array
    probability_distribution_np = p.cpu().numpy().squeeze()

    # Remove the start and end of sequence tokens
    if remove_tokens:
        probability_distribution_np = probability_distribution_np[1:-1, :]

    # If no characters are specified, include only amino acids by default
    if include is "canonical":
        include = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    elif include is "all":
        include = alphabet.keys()
    else:
        include = include

    # Filter the alphabet dictionary based on the 'include' list
    filtered_alphabet = {char: i for char, i in alphabet.items() if char in include}
    with open('alphabet', 'w') as f:
        print(include, file=f)
        print(alphabet, file=f)
        print(filtered_alphabet, file=f)
    # Create a pandas DataFrame with appropriate column and row labels
    df = pd.DataFrame(probability_distribution_np[:, list(filtered_alphabet.values())],
                      columns=[i for i in filtered_alphabet.keys()])

    # Create a heatmap using seaborn
    plt.figure(figsize=(20, 6))
    sns.heatmap(df.T, cmap="Reds_r", linewidths=0.5, annot=False, cbar=True)
    plt.xlabel("Sequence Position")
    plt.ylabel("Character")
    plt.title("Per-Position Probability Distribution Heatmap")

    # Save the plot to the specified destination, if provided
    if dest is not None:
        plt.savefig(dest, dpi=300, bbox_inches='tight')

    # Show the plot, if the 'show' argument is True
    if show:
        plt.show()

seqs = ["GAAEAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSAPATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSAAS"]
results, batch_lens, batch_labels, alphabet = esm_compute(seqs)

seq_rep = get_seq_rep(results, batch_lens)
logits = get_logits(results)
p = get_probability_distribution(logits)
pp_entropy = per_position_entropy(p)
#attn = get_attentions(results)

with open('test', 'w') as f:
    print(len(seqs[0]), file=f)
    print(seq_rep[0].shape, file=f)
    print(logits.shape, file=f)
    print(p, file=f)
    print(pp_entropy, file=f)


plot_probability(p=p, alphabet=alphabet, dest='heat.png')