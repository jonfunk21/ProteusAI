import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import gpytorch


def one_hot_encoder(sequences, alphabet=None, canonical=True, pbar=None, padding=None):
    """
    Encodes sequences provided an alphabet.

    Parameters:
        sequences (list or str): list of amino acid sequences or a single sequence.
        alphabet (list or None): list of characters in the alphabet or None to load from file.
        canonical (bool): only use canonical amino acids.
        padding (int or None): the length to which all sequences should be padded.
                               If None, no padding beyond the length of the longest sequence.

    Returns:
        torch.Tensor: (number of sequences, padding or maximum sequence length, size of the alphabet) for list input
                      (padding or maximum sequence length, size of the alphabet) for string input
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

    # Determine the length to which sequences should be padded
    max_sequence_length = max(len(sequence) for sequence in sequences)
    padded_length = padding if padding is not None else max_sequence_length

    n_sequences = len(sequences)
    alphabet_size = len(alphabet)

    # Create an empty tensor of the right size, with padding length
    tensor = torch.zeros((n_sequences, padded_length, alphabet_size))

    # Fill the tensor
    for i, sequence in enumerate(sequences):
        if pbar:
            pbar.set(
                i, message="Computing", detail=f"{i}/{len(sequences)} remaining..."
            )
        for j, character in enumerate(sequence):
            if j >= padded_length:
                break  # Stop if the sequence length exceeds the padded length
            # Get the index of the character in the alphabet
            char_index = alphabet_dict.get(
                character, -1
            )  # Return -1 if character is not in the alphabet
            if char_index != -1:
                # Set the corresponding element of the tensor to 1
                tensor[i, j, char_index] = 1.0

    # If the input was a string, return a tensor of shape (padded_length, alphabet_size)
    if singular:
        tensor = tensor.squeeze(0)

    return tensor


def blosum_encoding(
    sequences, matrix="BLOSUM62", canonical=True, pbar=None, padding=None
):
    """
    Returns BLOSUM encoding for amino acid sequences. Unknown amino acids will be
    encoded with 0.5 in the entire row.

    Parameters:
    -----------
        sequences (list or str): List of amino acid sequences or a single sequence.
        matrix (str): Choice of BLOSUM matrix. Can be 'BLOSUM50' or 'BLOSUM62'.
        canonical (bool): Only use canonical amino acids.
        padding (int or None): The length to which all sequences should be padded.
                               If None, no padding beyond the length of the longest sequence.
        pbar: Progress bar for shiny app.

    Returns:
    --------
        torch.Tensor: BLOSUM encoded sequence.
    """

    # Check if sequences is a string
    if isinstance(sequences, str):
        singular = True
        sequences = [sequences]  # Make it a list to use the same code below
    else:
        singular = False

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Load the alphabet
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

    # Create the BLOSUM encoding dictionary
    blosum_matrix = {}
    for i, letter_1 in enumerate(alphabet):
        if canonical:
            blosum_matrix[letter_1] = matrix[i][:20]
        else:
            blosum_matrix[letter_1] = matrix[i]

    # Determine the length to which sequences should be padded
    max_sequence_length = max(len(sequence) for sequence in sequences)
    padded_length = padding if padding is not None else max_sequence_length

    n_sequences = len(sequences)
    alphabet_size = len(blosum_matrix["A"])

    # Create an empty tensor of the right size, with padding length
    tensor = torch.zeros((n_sequences, padded_length, alphabet_size))

    # Convert each amino acid in sequence to BLOSUM encoding
    for i, sequence in enumerate(sequences):
        if pbar:
            pbar.set(
                i, message="Computing", detail=f"{i}/{len(sequences)} remaining..."
            )
        for j, aa in enumerate(sequence):
            if j >= padded_length:
                break  # Stop if the sequence length exceeds the padded length
            if aa in alphabet:
                tensor[i, j, :] = torch.tensor(blosum_matrix[aa])
            else:
                # Handle unknown amino acids with a default value of 0.5
                tensor[i, j, :] = 0.5

    # If the input was a string, return a tensor of shape (padded_length, alphabet_size)
    if singular:
        tensor = tensor.squeeze(0)

    return tensor


# Define the VHSE dictionary
vhse_dict = {
    "A": [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
    "R": [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.30, 0.83],
    "N": [-0.99, 0.00, -0.37, 0.69, -0.55, 0.85, 0.73, -0.80],
    "D": [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
    "C": [0.18, -1.67, -0.46, -0.21, 0.00, 1.20, -1.61, -0.19],
    "Q": [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.20, -0.41],
    "E": [-1.18, 0.40, 0.10, 0.36, -2.16, -0.17, 0.91, 0.02],
    "G": [-0.20, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
    "H": [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
    "I": [1.27, -0.14, 0.30, -1.80, 0.30, -1.61, -0.16, -0.13],
    "L": [1.36, 0.07, 0.26, -0.80, 0.22, -1.37, 0.08, -0.62],
    "K": [-1.17, 0.70, 0.70, 0.80, 1.64, 0.67, 1.63, 0.13],
    "M": [1.01, -0.53, 0.43, 0.00, 0.23, 0.10, -0.86, -0.68],
    "F": [1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.20],
    "P": [0.22, -0.17, -0.50, 0.05, -0.01, -1.34, -0.19, 3.56],
    "S": [-0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
    "T": [-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39],
    "W": [1.50, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85],
    "Y": [0.61, 1.60, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52],
    "V": [0.76, -0.92, -0.17, -1.91, 0.22, -1.40, -0.24, -0.03],
}


def vhse_encoder(sequences, padding=None, pbar=None):
    """
    Encodes sequences using VHSE descriptors.

    Parameters:
        sequences (list or str): List of amino acid sequences or a single sequence.
        padding (int or None): Length to which all sequences should be padded.
                               If None, no padding beyond the longest sequence.
        pbar: Progress bar for tracking (optional).

    Returns:
        torch.Tensor: VHSE encoded tensor of shape
                      (number of sequences, padding or max sequence length, 8)
    """
    # Ensure input is a list
    if isinstance(sequences, str):
        singular = True
        sequences = [sequences]
    else:
        singular = False

    # Determine maximum sequence length and apply padding
    max_sequence_length = max(len(sequence) for sequence in sequences)
    padded_length = padding if padding is not None else max_sequence_length

    n_sequences = len(sequences)
    vhse_size = 8  # VHSE descriptors have 8 components

    # Initialize output tensor with zeros
    tensor = torch.zeros((n_sequences, padded_length, vhse_size))

    # Encode each sequence
    for i, sequence in enumerate(sequences):
        if pbar:
            pbar.set(i, message="Encoding", detail=f"{i}/{n_sequences} completed...")
        for j, aa in enumerate(sequence):
            if j >= padded_length:
                break
            tensor[i, j] = torch.tensor(
                vhse_dict.get(aa, [0.5] * vhse_size)
            )  # Default for unknown AAs

    # Squeeze output for single sequence input
    if singular:
        tensor = tensor.squeeze(0)

    return tensor


def plot_attention(attention: list, layer: int, head: int, seq: Union[str, list]):
    """
    Plot the attention weights for a specific layer and head.

    Args:
        attention (list): List of attention weights from the model
        layer (int): Index of the layer to visualize
        head (int): Index of the head to visualize
        seq (str): Input sequence as a list of tokens
    """

    if isinstance(seq, str):
        seq = [char for char in seq]

    # Get the attention weights for the specified layer and head
    attn_weights = attention[layer][head].detach().cpu().numpy()

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 10))
    sns.heatmap(attn_weights, xticklabels=seq, yticklabels=seq, cmap="viridis")

    # Set plot title and labels
    plt.title(f"Attention weights - Layer {layer + 1}, Head {head + 1}")
    plt.xlabel("Input tokens")
    plt.ylabel("Output tokens")

    # Show the plot
    plt.show()


class GP(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, fix_mean=False
    ):  # special method: instantiate object
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()  # attribute
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module.constant.data.fill_(1)  # Set the mean value to 1
        if fix_mean:
            self.mean_module.constant.requires_grad_(False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predict_gp(model, likelihood, X):
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        predictions = likelihood(model(X))
        y_pred = predictions.mean
        y_std = predictions.stddev
        # lower, upper = predictions.confidence_region()

    return y_pred, y_std


def computeR2(y_true, y_pred):
    """
    Compute R2-values for to torch tensors.

    Args:
        y_true (torch.Tensor): true y-values
        y_pred (torch.Tensor): predicted y-values
    """
    # Ensure the tensors are 1-dimensional
    if y_true.dim() != 1 or y_pred.dim() != 1:
        raise ValueError("Both y_true and y_pred must be 1-dimensional tensors")

    # Compute the mean of true values
    y_mean = torch.mean(y_true)

    # Compute the total sum of squares (SS_tot)
    ss_tot = torch.sum((y_true - y_mean) ** 2)

    # Compute the residual sum of squares (SS_res)
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Compute the R² value
    r2 = 1 - (ss_res / ss_tot)

    return r2.item()  # Convert tensor to float
