from biotite.structure.io.mol import MOLFile
from biotite.structure.io.pdb import PDBFile
import numpy as np
import biotite.structure as struc


def get_atom_array(file_path):
    """
    returns atom array of file.
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
    Get residue ids of contacts of small molecule(s) to a protein.

    Parameters:
        mols (str, list of strings): path to molecule file
        protein (str): path to protein file
        dist (float): distance to be considered contact

    Returns:
        set: res_ids of protein residues which are in contact with molecules
    """
    if isinstance(mols, list) or isinstance(mols, tuple):
        single_element = False
        mols = [get_atom_array(m) for m in mols]
    else:
        mols = [get_atom_array(mols)]
        single_element = True

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