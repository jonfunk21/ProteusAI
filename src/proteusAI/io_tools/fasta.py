# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import os
from biotite.sequence import ProteinSequence
import numpy as np


def load_all_fastas(
    path: str, file_type: str = ".fasta", biotite: bool = False
) -> dict:
    """
    Loads all fasta files from a directory, returns the names/ids and sequences as lists.

    Parameters:
        path (str): path to directory containing fasta files
        file_type (str): some fastas are stored with different file endings. Default '.fasta'.
        biotite (bool): returns sequences as biotite.sequence.ProteinSequence object

    Returns:
        dict: dictionary of file_names and tuple (names, sequences)

    Example:
        results = load_fastas('/path/to/fastas')
    """
    file_names = [f for f in os.listdir(path) if f.endswith(file_type)]
    files = [os.path.join(path, f) for f in file_names if f.endswith(file_type)]

    results = {}
    for i, file in enumerate(files):
        names = []
        sequences = []
        with open(file, "r") as f:
            current_sequence = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_sequence:
                        sequences.append(current_sequence)
                    names.append(line[1:])
                    current_sequence = ""
                else:
                    current_sequence += line
            if biotite:
                sequences.append(ProteinSequence(current_sequence))
            else:
                sequences.append(current_sequence)

        results[file_names[i]] = (names, sequences)
    return results


def load_fasta(file: str, biotite: bool = False) -> tuple:
    """
    Load all sequences in a fasta file. Returns names and sequences

    Parameters:
        file (str): path to file
        biotite (bool): returns sequences as biotite.sequence.ProteinSequence object

    Returns:
        tuple: two lists containing the names and sequences

    Example:
        names, sequences = load_fastas('example.fasta')
    """

    names = []
    sequences = []
    with open(file, "r") as f:
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:
                    sequences.append(current_sequence)
                names.append(line[1:])
                current_sequence = ""
            else:
                current_sequence += line
        if biotite:
            sequences.append(ProteinSequence(current_sequence))
        else:
            sequences.append(current_sequence)

    return names, sequences


def write_fasta(names: list, sequences: list, dest: str = None):
    """
    Takes a list of names and sequences and writes a single
    fasta file containing all the names and sequences. The
    files will be saved at the destination

    Parameters:
        names (list): list of sequence names
        sequences (list): list of sequences
        dest (str): path to output file

    Example:
        write_fasta(names, sequences, './out.fasta')
    """
    assert isinstance(names, list) and isinstance(
        sequences, list
    ), "names and sequences must be type list"
    assert len(names) == len(sequences), "names and sequences must have the same length"

    with open(dest, "w") as f:
        for i in range(len(names)):
            f.writelines(">" + names[i] + "\n")
            if i == len(names) - 1:
                f.writelines(sequences[i])
            else:
                f.writelines(sequences[i] + "\n")


def one_hot_encoding(sequence: str):
    """
    Returns one hot encoding for amino acid sequence. Unknown amino acids will be
    encoded with 0.5 at in entire row.

    Parameters:
    -----------
        sequence (str): Amino acid sequence

    Returns:
    --------
        numpy.ndarray: One hot encoded sequence
    """
    # Define amino acid alphabets and create dictionary
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}

    # Initialize empty numpy array for one-hot encoding
    seq_one_hot = np.zeros((len(sequence), len(amino_acids)))

    # Convert each amino acid in sequence to one-hot encoding
    for i, aa in enumerate(sequence):
        if aa in aa_dict:
            seq_one_hot[i, aa_dict[aa]] = 1.0
        else:
            # Handle unknown amino acids with a default value of 0.5
            seq_one_hot[i, :] = 0.5

    return seq_one_hot


def blosum_encoding(sequence, matrix="BLOSUM62", canonical=True):
    """
    Returns BLOSUM encoding for amino acid sequence. Unknown amino acids will be
    encoded with 0.5 at in entire row.

    Parameters:
    -----------
        sequence (str): Amino acid sequence
        blosum_matrix_choice (str): Choice of BLOSUM matrix. Can be 'BLOSUM50' or 'BLOSUM62'
        canonical (bool): only use canonical amino acids

    Returns:
    --------
        numpy.ndarray: BLOSUM encoded sequence
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    ### Amino Acid codes
    alphabet_file = os.path.join(script_dir, "matrices/alphabet")
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    # Define BLOSUM matrices
    _blosum50 = (
        np.loadtxt(os.path.join(script_dir, "matrices/BLOSUM50"), dtype=float)
        .reshape((24, -1))
        .T
    )
    _blosum62 = (
        np.loadtxt(os.path.join(script_dir, "matrices/BLOSUM62"), dtype=float)
        .reshape((24, -1))
        .T
    )

    # Choose BLOSUM matrix
    if matrix == "BLOSUM50":
        matrix = _blosum50
    elif matrix == "BLOSUM62":
        matrix = _blosum62
    else:
        raise ValueError(
            "Invalid BLOSUM matrix choice. Choose 'BLOSUM50' or 'BLOSUM62'."
        )

    blosum_matrix = {}
    for i, letter_1 in enumerate(alphabet):
        if canonical:
            blosum_matrix[letter_1] = matrix[i][:20]
        else:
            blosum_matrix[letter_1] = matrix[i]

    # create empty encoding vector
    encoding = np.zeros((len(sequence), len(blosum_matrix["A"])))

    # Convert each amino acid in sequence to BLOSUM encoding
    for i, aa in enumerate(sequence):
        if aa in alphabet:
            encoding[i, :] = blosum_matrix[aa]
        else:
            # Handle unknown amino acids with a default value of 0.5
            encoding[i, :] = 0.5

    return encoding
