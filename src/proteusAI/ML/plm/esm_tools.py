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
from biotite.structure.io.pdb import PDBFile
import tempfile
import typing as T
import math

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
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model == "esm1v":
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
    else:
        raise f"{model} is not a valid model"

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    if names == None:
        names = names = [f'seq{i}' for i in range(len(seqs))]

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

def get_log_prob(logits):
    """
    Convert logits to log-probability distribution for each position in the sequence.
    """
    # Apply softmax function to the logits along the alphabet dimension (dim=2)
    probability_distribution = F.log_softmax(logits, dim=-1)

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


def mask_positions(sequence: str, mask_char: str='<mask>'):
    """
    Mask every position of an amino acid sequence. Returns list of masked sequence:

    Parameters:
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
        masked_seq = sequence[:i] + mask_char + sequence[i+1:]
        masked_sequences.append(masked_seq)

    return masked_sequences


def get_mutant_logits(seq: str, model: str="esm1v", batch_size: int=10, rep_layer: int=33, alphabet_size: int=33):
    """
    Exhaustively compute the logits for every position in a sequence using esm1v or esm2.
    Every position of a sequence will be masked and the logits for the masked position
    will be calculated. The probabilities for every position will be concatenated in a
    combined logits tensor, which will be returned together with the alphabet.

    Parameters:
        seq (str): native protein sequence
        model (str): choose either esm2 or esm1v
        batch_size (int): batch size. Default 10
        rep_layer (int): choose representation layer. Default 33.

    Returns:
        tuple: torch.Tensor (1, sequence_length, alphabet_size) and alphabet esm.data.Alphabet

    Example:
        1.
        seq = "AGHRFLIKLKI"
        logits = get_mutant_logits(seq=seq)
    """
    masked_seqs = mask_positions(seq) # list of sequences where every position has been masked once
    names = [f'seq{i}' for i in range(len(masked_seqs))]
    sequence_length = len(seq)

    # Initialize an empty tensor of the desired shape
    logits_tensor = torch.zeros(1, sequence_length, alphabet_size)

    for i in range(0, len(masked_seqs), batch_size):
        results, batch_lens, batch_labels, alphabet = esm_compute(masked_seqs[i:i + batch_size],
                                                                  names[i:i + batch_size], model=model,
                                                                  rep_layer=rep_layer)
        logits = results["logits"]

        # Fill the logits_tensor with the logits for each masked position
        for j in range(logits.shape[0]):
            masked_position = i + j
            if masked_position < sequence_length:
                logits_tensor[0, masked_position] = logits[j, masked_position + 1]

    return logits, alphabet


def most_likely_sequence(log_prob_tensor, alphabet):
    """
    Get the most likely amino acid sequence based on log probabilities.

    Parameters:
        log_prob_tensor (torch.Tensor): Tensor of shape (1, sequence_length, alphabet_size) containing log probabilities
        alphabet (dict or esm.data.Alphabet): Dictionary mapping indices to characters

    Returns:
        str: Most likely amino acid sequence
    """
    if type(alphabet) == dict:
        pass
    else:
        try:
            alphabet = alphabet.to_dict()
        except:
            raise "alphabet has an unexpected format"

    # Find the indices of the maximum log probabilities along the alphabet dimension
    max_indices = torch.argmax(log_prob_tensor, dim=-1).squeeze()

    # Filter the alphabet dictionary to only include canonical AAs
    include = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    filtered_alphabet = {i: char for char, i in alphabet.items() if char in include}

    # Map the indices back to their corresponding amino acids using the filtered_alphabet dictionary
    most_likely_seq = ''.join([filtered_alphabet[int(idx)] for idx in max_indices if int(idx) in filtered_alphabet])

    return most_likely_seq


def find_mutations(native_seq, predicted_seq):
    """
    Find the mutations between the native protein sequence and the predicted most likely sequence.

    Parameters:
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


def plot_heat(p, alphabet, include="canonical", dest=None, title: str=None, remove_tokens=False, show=True):
    """
    Plot a heatmap of the probability distribution for each position in the sequence.

    Parameters:
        p (torch.Tensor): probability_distribution torch.Tensor with shape (1, sequence_length, alphabet_size)
        alphabet (dict or esm.data.Alphabet): Dictionary mapping indices to characters
        include (str or list): List of characters to include in the heatmap (default: canonical, include only canonical amino acids)
        dest (str): Optional path to save the plot as an image file (default: None)
        title (str): title of plot
        remove_tokens (bool): Remove start of sequence and end of sequence tokens
        show (bool): Display plot if True (default: True)

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
    if include == "canonical":
        include = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    elif include == "all":
        include = alphabet.keys()
    elif type(alphabet) == list:
        include = include
    else:
        raise "include must either be 'canonical' 'all' or a list of valid elements"

    # Filter the alphabet dictionary based on the 'include' list
    filtered_alphabet = {char: i for char, i in alphabet.items() if char in include}

    # Create a pandas DataFrame with appropriate column and row labels
    df = pd.DataFrame(probability_distribution_np[:, list(filtered_alphabet.values())],
                      columns=[i for i in filtered_alphabet.keys()])

    # Create a heatmap using seaborn
    plt.figure(figsize=(20, 6))
    sns.heatmap(df.T, cmap="Reds", linewidths=0.5, annot=False, cbar=True)
    plt.xlabel("Sequence Position")
    plt.ylabel("Character")
    if title == None:
        plt.title("Per-Position Probability Distribution Heatmap")
    else:
        plt.title(title)

    # Save the plot to the specified destination, if provided
    if dest is not None:
        plt.savefig(dest, dpi=300, bbox_inches='tight')

    # Show the plot, if the 'show' argument is True
    if show:
        plt.show()

### Protein structure
def string_to_tempfile(data):
    """
    Take a string and return a temporary file object with string as content
    """
    # create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # write the string to the file
        temp_file.write(data.encode('utf-8'))
        # flush the file to make sure the data is written
        temp_file.flush()
        # return the file object
        return temp_file


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
        seqs: list, names: list=None, chunk_size: int = 124,
        max_tokens_per_batch: int = 1024, num_recycles: int = None):
    """
    Predict the structure of proteins.

    Parameters:
        sequences (list): all sequences for structure prediction
        names (list): names of the sequences
        chunck_size (int): Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). Recommended values: 128, 64, 32.
        max_tokens_per_batch (int): Maximum number of tokens per gpu forward-pass. This will group shorter sequences together.
        num_recycles (int): Number of recycles to run. Defaults to number used in training 4.

    Returns:
        all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs
    """
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(chunk_size)

    if names == None:
        names = [f'seq{i}' for i in range(len(seqs))]

    all_sequences = list(zip(names, seqs))
    batched_sequences = create_batched_sequence_datasest(all_sequences, max_tokens_per_batch)
    all_headers = []
    all_sequences = []
    all_pdbs = []
    pTMs = []
    mean_pLDDTs = []
    for headers, sequences in batched_sequences:
        output = model.infer(sequences, num_recycles=num_recycles)
        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            all_headers.append(header)
            all_sequences.append(seq)
            all_pdbs.append(PDBFile.read(string_to_tempfile(pdb_string).name))  # biotite pdb file name
            mean_pLDDTs.append(mean_plddt.item())
            pTMs.append(ptm.item())

    return all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs


def format_float(float_value: float, str_len: int=5, round_val: int=2):
    """
    Format a float value to a string of length 5, rounded to two decimal places.

    Parameters:
        float_value (float): The float value to format

    Returns:
        str: The formatted string
    """
    rounded_value = round(float_value, round_val)
    value_str = str(rounded_value)
    formatted_str = value_str.rjust(str_len, ' ')
    return formatted_str


def entropy_to_bfactor(pdb, entropy_values, trim=False, alphabet_size=33):
    """
    Convert per-position entropy values to b-factors between 0 and 100.

    Parameters:
        entropy_values (list or numpy.ndarray): List of entropy values
        trim (bool): If True, remove the start and end of sequence tokens (default: False)

    Returns:
        list: List of scaled b-factors
    """
    if type(pdb) != str:
        try:
            pdb = str(pdb)
        except:
            raise "invalid input type for pdb"

    # Remove the start and end of sequence tokens, if requested
    if trim:
        entropy_values = entropy_values[:, 1:-1]

    scaled_entropy = 100 * (1 - entropy_values / math.log2(alphabet_size))
    scaled_entropy_list = scaled_entropy.tolist()

    b_factor_strings = [[format(x, '.2f').rjust(6) for x in row] for row in scaled_entropy_list][0]
    lines = []
    id = -1
    count = -1
    for i, line in enumerate(str(pdb).split('\n')):
        if line.startswith('ATOM'):
            res_id = int(line[23:26])
            if res_id != id:
                id = res_id
                count += 1
            line = line[:60] + b_factor_strings[count] + line[66:]
        lines.append(line + '\n')
    pdb = PDBFile.read(string_to_tempfile("".join(lines)).name)
    return pdb

# test
name = "1HY2"
seq = "GAAEAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSAPATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSAAS"
logits, alphabet = get_mutant_logits(seq)
p = get_log_prob(logits)
log_prob = get_log_prob(logits)
entropy = per_position_entropy(p)
_, _, pdbs, _, _ = structure_prediction(seqs=[seq], names=[name])
pdb = entropy_to_bfactor(pdbs[0], entropy)
with open('test', 'w') as f:
    print(log_prob, file=f)
plot_heat(p=p, alphabet=alphabet, include="canonical", remove_tokens=False, dest="log_odds.png", show=False)
plot_heat(p=log_prob, alphabet=alphabet, include="canonical", remove_tokens=False, dest="log_odds.png", show=False, title='Per position log-odds')
pdb.write('test_entropy.pdb')
