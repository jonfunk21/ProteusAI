import os
from biotite.sequence import ProteinSequence


def load_fastas(path):
    """
    Loads all fasta files from a directory, returns the names/ids and sequences as lists.

    Parameters:
        path (str): path to directory containing fasta files

    Returns:
        tuple: two lists containing the names and sequences as biotite.sequence.ProteinSequence object

    Example:
        names, sequences = load_fastas('/path/to/fastas')
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.fasta')]
    names = [f[:-6] for f in os.listdir(path) if f.endswith('.fasta')]

    sequences = []
    for file in files:
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    pass
                else:
                    sequences.append(ProteinSequence(line))

    return names, sequences