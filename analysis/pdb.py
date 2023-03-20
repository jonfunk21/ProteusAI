import py3Dmol
from string import ascii_uppercase, ascii_lowercase
from biotite.structure.io.pdb import PDBFile
from Bio.PDB.Polypeptide import is_aa
from Bio import SeqIO
import biotite.structure as struc
import biotite.structure.io as strucio

alphabet_list = list(ascii_uppercase + ascii_lowercase)
pymol_color_list = ["#33ff33", "#00ffff", "#ff33cc", "#ffff00", "#ff9999", "#e5e5e5", "#7f7fff", "#ff7f00",
                    "#7fff7f", "#199999", "#ff007f", "#ffdd5e", "#8c3f99", "#b2b2b2", "#007fff", "#c4b200",
                    "#8cb266", "#00bfbf", "#b27f7f", "#fcd1a5", "#ff7f7f", "#ffbfdd", "#7fffff", "#ffff7f",
                    "#00ff7f", "#337fcc", "#d8337f", "#bfff3f", "#ff7fff", "#d8d8ff", "#3fffbf", "#b78c4c",
                    "#339933", "#66b2b2", "#ba8c84", "#84bf00", "#b24c66", "#7f7f7f", "#3f3fa5", "#a5512b"]


def show(pdb_str, color='rainbow', vmin=50, vmax=90, chains=None, Ls=None, size=(800, 480), show_sidechains=False,
         show_mainchains=False):
    """
    This function displays the 3D structure of a protein from a given PDB file in a Jupyter notebook.
    The protein structure can be colored by chain, rainbow, pLDDT, or confidence value. The size of the
    display can be changed. The sidechains and mainchains can be displayed or hidden.

    Parameters:
        pdb_str (str): The filename of the PDB file that contains the protein structure.
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

    with open(pdb_str) as ifile:
        system = "".join([x for x in ifile])

    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', width=size[0], height=size[1])

    if chains is None:
        chains = 1 if Ls is None else len(Ls)

    view.addModelsAsFrames(system)
    if color == "pLDDT" or color == 'confidence':
        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': vmin, 'max': vmax}}})
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


def get_sequence_length(ref_model):
    """
    Get sequence length of a Bio.PDB.Structure.Structure object
    """
    seq_len = 0
    for chain in ref_model:
        for residue in chain:
            if is_aa(residue):
                seq_len += 1
    return seq_len


def struc_align(pdb1: str, pdb2: str, atms1: list = None, atms2: list = None):
    """
    Alignes superimposes protein structures, either based on common residues
    or alignes a set of atoms. superimpose struc2 on struc1

    Parameters:
    -----------
        pdb1 (str): biotite pdb file
        pdb2 (str): biotite pdb file
        atms1, atms2 (list): list of atoms which should be superimposed.
            atms1 are atoms of struc1 which are common in struc2 (atms2)

    Returns:
    --------
        pdb_file1, pdb_file2: biotite pdb file
        rmsd: np.float32
    """
    struc1 = pdb1.get_structure()[0]
    struc2 = pdb2.get_structure()[0]

    if atms1 == None or atms2 == None:
        intersection1 = struc.filter_intersection(struc1, struc2)
        intersection2 = struc.filter_intersection(struc2, struc1)
        struc1_common = struc1[intersection1]
        struc2_common = struc2[intersection2]

    # write a test if the atoms are valid
    else:
        struc1_common = atms1
        struc2_common = atms2

    struc2_superimposed, transformation = struc.superimpose(
        struc1_common, struc2_common, (struc1_common.atom_name == "CA")
    )

    struc2_superimposed = struc.superimpose_apply(struc2, transformation)

    pdb_file1 = PDBFile()
    struc1 = PDBFile.set_structure(pdb_file1, struc1)
    coord1 = pdb_file1.get_coord()[0][intersection1]

    pdb_file2 = PDBFile()
    struc2 = PDBFile.set_structure(pdb_file2, struc2_superimposed)
    coord2 = pdb_file2.get_coord()[0][intersection2]

    rmsd = struc.rmsd(coord1, coord2)

    return pdb_file1, pdb_file2, rmsd


def to_fasta(pdb_path: str):
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