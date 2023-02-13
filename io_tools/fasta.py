import os
from biotite.sequence import ProteinSequence


def load(path: str) -> tuple:
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


def write(names: str, sequences: str, dest: str = None):
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
    assert len(names) == len(sequences), 'names and sequences must have the same length'

    with open(dest, 'w') as f:
        for i in range(len(names)):
            f.writelines('>' + names[i] + '\n')
            if i == len(names) - 1:
                f.writelines(sequences[i])
            else:
                f.writelines(sequences[i] + '\n')