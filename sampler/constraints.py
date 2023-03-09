import numpy as np

def length_constraint(seqs: list, max_len: int = 200):
    """
    Constraint for the length of a seuqences.

    Parameters:
        seqs (list): sequences to be scored
        max_len (int): maximum length that a sequence is allowed to be. Default = 300

    Returns:
        float: Energy value
    """
    energies = np.zeros(len(seqs))

    for i, seq in enumerate(seqs):
        if len(seq) > max_len:
            energies[i] = float(len(seq) - max_len)
        else:
            energies[i] = 0.

    return energies
