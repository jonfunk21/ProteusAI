# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from biotite.structure.io.mol import MOLFile
from biotite.structure.io.pdb import PDBFile
import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
from Bio import SeqIO


def get_atom_array(file_path):
    """
    Returns atom array for a pdb file.

    Parameters:
        file_path (str): path to pdb file

    Returns:
        atom array
    """
    if file_path.endswith('pdb'):
        atom_mol = PDBFile.read(file_path)
        atom_array = atom_mol.get_structure()
    else:
        try:
            atom_mol = MOLFile.read(file_path)
            atom_array = atom_mol.get_structure()
        except:
            raise f'file: {file_path} invalid file format'

    return atom_array


def mol_contacts(mols, protein, dist=4.0):
    """
    Get residue ids of contacts of small molecule(s) to protein.

    Parameters:
        mols (str, list of strings): path to molecule file
        protein (str): path to protein file
        dist (float): distance to be considered contact

    Returns:
        set: res_ids of protein residues which are in contact with molecules
    """
    if isinstance(mols, list) or isinstance(mols, tuple):
        mols = [get_atom_array(m) for m in mols]
    else:
        mols = [get_atom_array(mols)]

    protein = get_atom_array(protein)

    res_ids = set()
    for mol in mols:
        cell_list = struc.CellList(mol, cell_size=dist)
        for prot in protein:
            contacts = cell_list.get_atoms(prot.coord, radius=dist)

            contact_indices = np.where((contacts != -1).any(axis=1))[0]

            contact_res_ids = prot.res_id[contact_indices]
            res_ids.update(contact_res_ids)

    res_ids = sorted(res_ids)
    return (res_ids)


def pdb_to_fasta(pdb_path: str):
    """
    Returns fasta sequence of pdb file.

    Parameters:
        pdb_path (str): path to pdb file

    Returns:
        str: fasta sequence string
    """
    seq = ""
    with open(pdb_path, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            seq = "".join(['>', str(record.id), '\n', str(record.seq)])

    return seq


def get_sse(pdb_path: str):
    """
    Returns the secondary structure of a protein given the pdb file.
    The secondary structure is infered using the P-SEA algorithm
    as imple-mented by the biotite Python package.

    Parameters:
        pdb_path (str): path to pdb
    """
    array = strucio.load_structure(pdb_path)
    sse = struc.annotate_sse(array)
    sse = ''.join(sse)
    return sse