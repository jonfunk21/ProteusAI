# This source code is part of the proteusAI package and is distributed
# under the MIT License.

"""
A subpackage for mining_tools.
"""

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import biotite.structure.io as strucio
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import py3Dmol
from string import ascii_uppercase, ascii_lowercase


alphabet_list = list(ascii_uppercase + ascii_lowercase)
pymol_color_list = ["#33ff33", "#00ffff", "#ff33cc", "#ffff00", "#ff9999", "#e5e5e5", "#7f7fff", "#ff7f00",
                    "#7fff7f", "#199999", "#ff007f", "#ffdd5e", "#8c3f99", "#b2b2b2", "#007fff", "#c4b200",
                    "#8cb266", "#00bfbf", "#b27f7f", "#fcd1a5", "#ff7f7f", "#ffbfdd", "#7fffff", "#ffff7f",
                    "#00ff7f", "#337fcc", "#d8337f", "#bfff3f", "#ff7fff", "#d8d8ff", "#3fffbf", "#b78c4c",
                    "#339933", "#66b2b2", "#ba8c84", "#84bf00", "#b24c66", "#7f7f7f", "#3f3fa5", "#a5512b"]

amino_acid_mapping = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Adding a generic mapping for unknowns
    'UNK': 'X',  # Unknown
    'SEC': 'U',  # Selenocysteine, sometimes considered the 21st amino acid
    # Additional mappings as needed
}

def load_struc(prot):
    """
    Load a protein structure.

    Args:
        prot biotite.structure.AtomArray or path to pdb (str).

    Returns:
        biotite.structure.AtomArray
    """
    if type(prot) == struc.AtomArray:
        prot = prot

    elif type(prot) == str:
        try:
            prot = strucio.load_structure(prot)
        except:
            raise ValueError("prot 1 has an unexpected format")
    
    else:
        raise ValueError("pdb file has an unexpected format")

    return prot


def align(prot1, prot2):
    """
    Superimpose protein 2 on protein 1.

    Args:
        prot1 biotite.structure.AtomArray or path to pdb (str).
        prot2 biotite.structure.AtomArray or path to pdb (str).

    Returns:
        prot1, prot2_superimposed as biotite.structure.AtomArray's
    """
    prot1 = load_struc(prot1)
    prot2 = load_struc(prot2)

    prot1 = prot1[(prot1.chain_id == "A") | (prot1.chain_id == "B")]
    prot1 = prot1[~struc.filter_solvent(prot1)]
    prot2 = prot2[~struc.filter_solvent(prot2)]

    prot1_common = prot1[struc.filter_intersection(prot1, prot2)]
    prot2_common = prot2[struc.filter_intersection(prot2, prot1)]

    # Superimpose
    prot2_superimposed, transformation = struc.superimpose(
        prot1_common, prot2_common, (prot2_common.atom_name == "CA")
    )

    # We do not want the cropped structures
    prot2_superimposed = struc.superimpose_apply(prot2, transformation)

    

    # Write PDBx files as input for PyMOL
    cif_file = pdbx.PDBxFile()
    pdbx.set_structure(cif_file, prot1, data_block="prot1")
    cif_file.write("test1.cif")

    cif_file = pdbx.PDBxFile()
    pdbx.set_structure(cif_file, prot2_superimposed, data_block="prot2")
    cif_file.write("test2.cif")

    return prot1, prot2_superimposed

def compute_rmsd(prot1, prot2):
    """
    Compute rmsd between two proteins.

    Args:
        prot1 biotite.structure.AtomArray or path to pdb (str).
        prot2 biotite.structure.AtomArray or path to pdb (str).

    Returns:
        rmsd (float)
    """

    prot1 = load_struc(prot1)
    prot2 = load_struc(prot2)

    prot1_common = prot1[struc.filter_intersection(prot1, prot2)]
    prot2_common = prot2[struc.filter_intersection(prot2, prot1)]

    rmsd = struc.rmsd(prot1_common, prot2_common)
    print("{:.3f}".format(rmsd))
    return rmsd

def chain_parser(pdb_file):
    """
    Parse chains from pdb file.

    Args:
        pdb_file: path to pdb file (str) or AtomArray
    """

    prot = load_struc(pdb_file)
    
    chains = list(set(prot.chain_id)) # type: ignore
    return chains

def get_sequences(prot_f):
    """
    Get the sequence from a protein structure.

    Args:
        prot biotite.structure.AtomArray or path to pdb (str).

    Return:
        dictionary of chains and sequences
    """

    prot = load_struc(prot_f)

    chains = chain_parser(prot)

    sequences = {}
    for chain in chains:
        res_ids = list(set(prot[prot.chain_id==chain].res_id))
        residues = [amino_acid_mapping.get(prot[prot.res_id==r].res_name[0], 'X') for r in res_ids]
        sequences[chain] = ''.join(residues)

    return sequences

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