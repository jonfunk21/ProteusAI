# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import numpy as np
import esm
import typing as T
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.structure.io.pdb import PDBFile
import biotite.structure as struc
from biotite.structure import sasa
import tempfile

#_____Sequence Constraints_____
def length_constraint(seqs: list, max_len: int = 200):
    """
    Constraint for the length of a seuqences.

    Parameters:
        seqs (list): sequences to be scored
        max_len (int): maximum length that a sequence is allowed to be. Default = 300

    Returns:
        np.array: Energy values
    """
    energies = np.zeros(len(seqs))

    for i, seq in enumerate(seqs):
        if len(seq) > max_len:
            energies[i] = float(len(seq) - max_len)
        else:
            energies[i] = 0.

    return energies


def seq_identity(seqs, ref, matrix="BLOSUM62", local=False):
    """
    Calculates sequence identity of sequences against a reference sequence based on alignment.
    By default a global alignment is performed using the BLOSUM62 matrix.

    Parameters:
        seq1 (str): reference sequence
        seq2 (str): query sequence
        matrix (str): alignement matrix {BLOSUM62, BLOSUM50, BLOSUM30}. Default BLOSUM62
        local (bool): Local alignment if True, else global alignment.

    Returns:
        numpy.ndarray: identity scores of sequences
    """
    alph = seq.ProteinSequence.alphabet
    matrix = align.SubstitutionMatrix(alph, alph, matrix)

    seqs = [seq.ProteinSequence(s) for s in seqs]
    ref = seq.ProteinSequence(ref)

    scores = np.zeros(len(seqs))
    for i, s in enumerate(seqs):
        alignments = align.align_optimal(s, ref, matrix, local=local)
        score = align.get_sequence_identity(alignments[0])
        scores[i] = score

    return scores


#_____Structure Constraints_____
def string_to_tempfile(data):
    # create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # write the string to the file
        temp_file.write(data.encode('utf-8'))
        # flush the file to make sure the data_tools is written
        temp_file.flush()
        # return the file object
        return temp_file


def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    """
    Taken from https://github.com/facebookresearch/esm/blob/main/scripts/esmfold_inference.py
    """
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences

def structure_prediction(
        sequences: list, names: list, chunk_size: int = 124,
        max_tokens_per_batch: int = 1024, num_recycles: int = None):
    """
    Predict the structure of proteins.

    Parameters:
        sequences (list): all sequences for structure prediction
        names (list): names of the sequences
        chunck_size (int): Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). Recommended values: 128, 64, 32.
        max_tokens_per_batch (int): Maximum number of tokens per gpu forward-pass. This will group shorter sequences together.
        num_recycles (int): Number of recycles to run. Defaults to number used in training 4.

    Returns:
        all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs
    """
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(chunk_size)
    all_sequences = list(zip(names, sequences))

    batched_sequences = create_batched_sequence_datasest(all_sequences, max_tokens_per_batch)
    all_headers = []
    all_sequences = []
    all_pdbs = []
    pTMs = []
    mean_pLDDTs = []
    for headers, sequences in batched_sequences:
        output = model.infer(sequences, num_recycles=num_recycles)
        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            all_headers.append(header)
            all_sequences.append(seq)
            all_pdbs.append(PDBFile.read(string_to_tempfile(pdb_string).name)) # biotite pdb file name
            mean_pLDDTs.append(mean_plddt.item())
            pTMs.append(ptm.item())

    return all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs


def globularity(pdbs):
    """
    globularity constraint

    Parameters:
        pdb (list): list of biotite pdb file

    Returns:
        np.array: variances of coordinates for each structure
    """
    variances = np.zeros(len(pdbs))

    for i, pdb in enumerate(pdbs):
        variance = pdb.get_coord().var()
        variances[i] = variance.item()

    return variances

def surface_exposed_hydrophobics(pdbs):
    """
    Calculate the surface exposed hydrophobics using the Shrake-Rupley (“rolling probe”) algorithm.

    Parameters:
        pdbs (list): list of biotite pdb file

    Returns:
        np.array: average sasa values for each structure
    """
    avrg_sasa_values = np.zeros(len(pdbs))
    for i, pdb in enumerate(pdbs):
        struc = pdb.get_structure()
        sasa_val = sasa(struc[0], probe_radius=1.4, atom_filter=None, ignore_ions=True,
                                      point_number=1000, point_distr='Fibonacci', vdw_radii='ProtOr')
        sasa_mean = sasa_val.mean()
        avrg_sasa_values[i] = sasa_mean.item()

    return avrg_sasa_values


def backbone_coordination(samples: list, refs: list):
    """
    Superimpose structures and calculate the RMSD of the backbones.
    sample structures will be aligned against reference structures.

    Parameters:
    -----------
        samples (list): list of sample structures
        refs (list): list of reference structures

    Returns:
    --------
        np.array: RMSD values of alignments
    """
    if len(samples) != len(refs):
        raise "samples and refs must have the same length"

    rmsds = np.zeros(len(samples))

    for i in range(len(samples)):
        _, _, rmsd = pdb.struc_align(refs[i], samples[i])
        rmsds[i] = rmsd.item()

    return rmsds


def all_atom_coordination(samples, refs, sample_consts, ref_consts):
    """
    Calculate the RMSD of residues with all atom constraints.
    All atomic positions will be taken into consideration.

    Parameters:
    -----------
        samples (list): list of biotite pdb files (mutated)
        refs (list): list of biotite pdb files (reference)
        sample_consts (list): list of constraints on the sample
            (potentially shifts over time due to deletions and insertions)
        ref_consts (list): list of constraints of the reference

    Returns:
    --------
        np.array (len(samples),): calculated RMSD values for each sequence
    """

    rmsds = np.zeros(len(samples))

    for i in range(len(samples)):
        sample = samples[i]
        ref = refs[i]
        sample_const = sample_consts[i]
        ref_const = ref_consts[i]

        # get structures
        sample_struc = sample.get_structure()[0]
        ref_struc = ref.get_structure()[0]

        # get indices for alignment
        sample_indices = np.where(np.isin(sample_struc.res_id, [i + 1 for i in sample_const['all_atm']]))
        ref_indices = np.where(np.isin(ref_struc.res_id, [i + 1 for i in ref_const['all_atm']]))

        sample_struc_common = sample_struc[sample_indices[0]]
        ref_struc_common = ref_struc[ref_indices[0]]

        sample_superimposed, transformation = struc.superimpose(
            ref_struc_common, sample_struc_common
        )

        sample_superimposed = struc.superimpose_apply(sample_struc, transformation)

        sample_pdb = PDBFile()
        PDBFile.set_structure(sample_pdb, sample_superimposed)
        sample_coord = sample_pdb.get_coord()[0][sample_indices]

        ref_pdb = PDBFile()
        PDBFile.set_structure(ref_pdb, ref_struc)
        ref_coord = ref_pdb.get_coord()[0][ref_indices]

        rmsd = struc.rmsd(sample_coord, ref_coord)

        rmsds[i] = rmsd

    return rmsds