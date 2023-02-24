import os
from biotite.sequence import ProteinSequence


def load_all(path: str, file_type: str = '.fasta', biotite: bool = False) -> tuple:
    """
    Loads all fasta files from a directory, returns the names/ids and sequences as lists.

    Parameters:
        path (str): path to directory containing fasta files
        file_type (str): some fastas are stored with different file endings. Default '.fasta'.
        biotite (bool): returns sequences as biotite.sequence.ProteinSequence object

    Returns:
        tuple: two lists containing the names and sequences

    Example:
        names, sequences = load_fastas('/path/to/fastas')
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(file_type)]

    names = []
    sequences = []
    for file in files:
        with open(file, 'r') as f:
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


def load(file: str, biotite: bool = False) -> tuple:
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
    with open(file, 'r') as f:
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