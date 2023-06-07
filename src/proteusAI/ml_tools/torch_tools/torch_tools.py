import torch

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