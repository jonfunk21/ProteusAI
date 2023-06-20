import torch
import os
import numpy as np

def one_hot_encoder(sequences: list, alphabet: dict):
    """
    Encodes sequences provided an alphabet.

    Parameters:
        sequences (list): list of amino acid sequences.
        alphabet (dict): alphabet as dictionary.

    Returns:
        torch.Tensor: (number of sequences, maximum sequence length, size of the alphabet)
    """
    # Get the maximum sequence length
    max_sequence_length = max(len(sequence) for sequence in sequences)
    n_sequences = len(sequences)
    alphabet_size = len(alphabet)

    # Create an empty tensor of the right size
    tensor = torch.zeros((n_sequences, max_sequence_length, alphabet_size))

    # Fill the tensor
    for i, sequence in enumerate(sequences):
        for j, character in enumerate(sequence):
            # Get the index of the character in the alphabet
            char_index = alphabet[character]
            # Set the corresponding element of the tensor to 1
            tensor[i, j, char_index] = 1.0

    return tensor


def one_hot_decoder(tensor: torch.Tensor, alphabet: dict):
    """
    Decodes one-hot encoded sequences back to their original sequences.

    Parameters:
        tensor (torch.Tensor): one-hot encodings
        alphabet (dict): alphabet

    Returns:
        list: list of sequences
    """
    # Get the inverse of the alphabet (from indices to characters)
    inverse_alphabet = {v: k for k, v in alphabet.items()}

    # Convert the tensor to a list of indices
    sequences = torch.argmax(tensor, dim=2).tolist()

    # Convert the list of indices to a list of sequences
    decoded_sequences = []
    for sequence in sequences:
        decoded_sequence = [inverse_alphabet[index] for index in sequence if index != 0]
        decoded_sequences.append(''.join(decoded_sequence))

    return decoded_sequences


def blosum_encoding(sequence, matrix='BLOSUM62', canonical=True):
    '''
    Returns BLOSUM encoding for amino acid sequence. Unknown amino acids will be
    encoded with 0.5 at in entire row.

    Parameters:
    -----------
        sequence (str): Amino acid sequence
        blosum_matrix_choice (str): Choice of BLOSUM matrix. Can be 'BLOSUM50' or 'BLOSUM62'
        canonical (bool): only use canonical amino acids

    Returns:
    --------
        torch.Tensor: BLOSUM encoded sequence
    '''

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    ### Amino Acid codes
    alphabet_file = os.path.join(script_dir, "matrices/alphabet")
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    # Define BLOSUM matrices
    _blosum50 = np.loadtxt(os.path.join(script_dir, "matrices/BLOSUM50"), dtype=float).reshape((24, -1)).T
    _blosum62 = np.loadtxt(os.path.join(script_dir, "matrices/BLOSUM62"), dtype=float).reshape((24, -1)).T

    # Choose BLOSUM matrix
    if matrix == 'BLOSUM50':
        matrix = _blosum50
    elif matrix == 'BLOSUM62':
        matrix = _blosum62
    else:
        raise ValueError("Invalid BLOSUM matrix choice. Choose 'BLOSUM50' or 'BLOSUM62'.")

    blosum_matrix = {}
    for i, letter_1 in enumerate(alphabet):
        if canonical:
            blosum_matrix[letter_1] = matrix[i][:20]
        else:
            blosum_matrix[letter_1] = matrix[i]
    
    # create empty encoding vector
    encoding = np.zeros((len(sequence), len(blosum_matrix['A'])))
    
    # Convert each amino acid in sequence to BLOSUM encoding
    for i, aa in enumerate(sequence):
        if aa in alphabet:
            encoding[i, :] = blosum_matrix[aa]
        else:
            # Handle unknown amino acids with a default value of 0.5
            encoding[i, :] = 0.5

    # Convert numpy array to torch tensor
    encoding = torch.from_numpy(encoding)

    return encoding