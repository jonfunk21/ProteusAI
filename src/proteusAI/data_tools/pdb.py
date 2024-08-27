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
import py3Dmol
from string import ascii_uppercase, ascii_lowercase


alphabet_list = list(ascii_uppercase + ascii_lowercase)
pymol_color_list = ["#33ff33", "#00ffff", "#ff33cc", "#ffff00", "#ff9999", "#e5e5e5", "#7f7fff", "#ff7f00",
                    "#7fff7f", "#199999", "#ff007f", "#ffdd5e", "#8c3f99", "#b2b2b2", "#007fff", "#c4b200",
                    "#8cb266", "#00bfbf", "#b27f7f", "#fcd1a5", "#ff7f7f", "#ffbfdd", "#7fffff", "#ffff7f",
                    "#00ff7f", "#337fcc", "#d8337f", "#bfff3f", "#ff7fff", "#d8d8ff", "#3fffbf", "#b78c4c",
                    "#339933", "#66b2b2", "#ba8c84", "#84bf00", "#b24c66", "#7f7f7f", "#3f3fa5", "#a5512b"]


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


### Visualization
def show_pdb(pdb_path, color='confidence', vmin=50, vmax=90, chains=None, Ls=None, size=(800, 480), show_sidechains=False,
         show_mainchains=False, highlight=None):
    """
    This function displays the 3D structure of a protein from a given PDB file in a Jupyter notebook.
    The protein structure can be colored by chain, rainbow, pLDDT, or confidence value. The size of the
    display can be changed. The sidechains and mainchains can be displayed or hidden.
    Parameters:
        pdb_path (str): The filename of the PDB file that contains the protein structure.
        color (str, optional): The color scheme for the protein structure. Can be "chain", "rainbow", "pLDDT", or "confidence". Defaults to "rainbow".
        vmin (float, optional): The minimum value of pLDDT or confidence value. Defaults to 50.
        vmax (float, optional): The maximum value of pLDDT or confidence value. Defaults to 90.
        chains (int, optional): The number of chains to be displayed. Defaults to None.
        Ls (list, optional): A list of the chains to be displayed. Defaults to None.
        size (tuple, optional): The size of the display window. Defaults to (800, 480).
        show_sidechains (bool, optional): Whether to display the sidechains. Defaults to False.
        show_mainchains (bool, optional): Whether to display the mainchains. Defaults to False.
    Returns:
        view: The 3Dmol view object that displays the protein structure.
    """

    with open(pdb_path) as ifile:
        system = "".join([x for x in ifile])

    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', width=size[0], height=size[1])

    if chains is None:
        chains = 1 if Ls is None else len(Ls)

    view.addModelsAsFrames(system)
    if color == "pLDDT" or color == 'confidence':
        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'rwb', 'min': vmin, 'max': vmax}}})
    elif color == "rainbow":
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    elif color == "chain":
        for n, chain, color in zip(range(chains), alphabet_list, pymol_color_list):
            view.setStyle({'chain': chain}, {'cartoon': {'color': color}})

    if show_sidechains:
        BB = ['C', 'O', 'N']
        view.addStyle({'and': [{'resn': ["GLY", "PRO"], 'invert': True}, {'atom': BB, 'invert': True}]},
                      {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        view.addStyle({'and': [{'resn': "GLY"}, {'atom': 'CA'}]},
                      {'sphere': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        view.addStyle({'and': [{'resn': "PRO"}, {'atom': ['C', 'O'], 'invert': True}]},
                      {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
    if show_mainchains:
        BB = ['C', 'O', 'N', 'CA']
        view.addStyle({'atom': BB}, {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        view.zoomTo()

    view.zoomTo()

    return view