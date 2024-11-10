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
