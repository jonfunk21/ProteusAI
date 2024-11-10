# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import os
import torch
from typing import Union


def load_embeddings(
    path: str, names: Union[list, None] = None, map_location: str = "cpu"
) -> tuple:
    """
    Loads all representations files from a directory, returns the names/ids and sequences as lists.

    Parameters:
        path (str): path to directory containing representations files
        names (list): list of file names in case files should be loaded in a specific order if names not None.
            Will go to provided path and load files by name order. Default None

    Returns:
        tuple: two lists containing the names and sequences as torch tensors

    Example:
        names, sequences = load('/path/to/representations')
    """

    tensors = []
    if names is None:
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt")]
        names = [f[:-3] for f in os.listdir(path) if f.endswith(".pt")]
        for f in files:
            t = torch.load(f, map_location=map_location)
            tensors.append(t)
    else:
        for name in names:
            t = torch.load(os.path.join(path, name), map_location=map_location)
            tensors.append(t)

    return names, tensors
