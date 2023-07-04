import torch
import os
import numpy as np

def one_hot_encoder(sequences, alphabet=None, canonical=True):
    """
    Encodes sequences provided an alphabet.

    Parameters:
        sequences (list or str): list of amino acid sequences or a single sequence.
        alphabet (list or None): list of characters in the alphabet or None to load from file.
        canonical (bool): only use canonical amino acids.

    Returns:
        torch.Tensor: (number of sequences, maximum sequence length, size of the alphabet) for list input
                      (maximum sequence length, size of the alphabet) for string input
    """
    # Check if sequences is a string
    if isinstance(sequences, str):
        singular = True
        sequences = [sequences]  # Make it a list to use the same code below
    else:
        singular = False

    # Load the alphabet from a file if it's not provided
    if alphabet is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        ### Amino Acid codes
        alphabet_file = os.path.join(script_dir, "matrices/alphabet")
        alphabet = np.loadtxt(alphabet_file, dtype=str)

    # If canonical is True, only use the first 20 characters of the alphabet
    if canonical:
        alphabet = alphabet[:20]

    # Create a dictionary to map each character in the alphabet to its index
    alphabet_dict = {char: i for i, char in enumerate(alphabet)}

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
            char_index = alphabet_dict.get(character, -1)  # Return -1 if character is not in the alphabet
            if char_index != -1:
                # Set the corresponding element of the tensor to 1
                tensor[i, j, char_index] = 1.0

    # If the input was a string, return a tensor of shape (max_sequence_length, alphabet_size)
    if singular:
        tensor = tensor.squeeze(0)

    return tensor



def blosum_encoding(sequences, matrix='BLOSUM62', canonical=True):
    '''
    Returns BLOSUM encoding for amino acid sequence. Unknown amino acids will be
    encoded with 0.5 at in entire row.

    Parameters:
    -----------
        sequences (list or str): List of amino acid sequences or a single sequence
        blosum_matrix_choice (str): Choice of BLOSUM matrix. Can be 'BLOSUM50' or 'BLOSUM62'
        canonical (bool): only use canonical amino acids

    Returns:
    --------
        torch.Tensor: BLOSUM encoded sequence
    '''

    # Check if sequences is a string
    if isinstance(sequences, str):
        singular = True
        sequences = [sequences]  # Make it a list to use the same code below
    else:
        singular = False

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

    # Get the maximum sequence length
    max_sequence_length = max(len(sequence) for sequence in sequences)
    n_sequences = len(sequences)
    alphabet_size = len(blosum_matrix['A'])

    # Create an empty tensor of the right size
    tensor = torch.zeros((n_sequences, max_sequence_length, alphabet_size))

    # Convert each amino acid in sequence to BLOSUM encoding
    for i, sequence in enumerate(sequences):
        for j, aa in enumerate(sequence):
            if aa in alphabet:
                tensor[i, j, :] = torch.tensor(blosum_matrix[aa])
            else:
                # Handle unknown amino acids with a default value of 0.5
                tensor[i, j, :] = 0.5

    # If the input was a string, return a tensor of shape (max_sequence_length, alphabet_size)
    if singular:
        tensor = tensor.squeeze(0)

    return tensor
