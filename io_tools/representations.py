import os
import torch

def load(path: str, map_location: str = 'cpu') -> list:
    """
    Loads all representations files from a directory, returns the names/ids and sequences as lists.

    Parameters:
        path (str): path to directory containing representations files

    Returns:
        tuple: two lists containing the names and sequences as torch tensors

    Example:
        names, sequences = load('/path/to/representations')
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]
    names = [f[:-3] for f in os.listdir(path) if f.endswith('.pt')]

    tensors = []
    for f in files:
        t = torch.load(f, map_location=map_location)
        tensors.append(t)

    return names, tensors