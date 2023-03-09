import py3Dmol
from string import ascii_uppercase, ascii_lowercase
import Bio.PDB
from Bio.PDB.Polypeptide import is_aa

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


def align_proteins(ref_filename: str, sample_filename: str, outfile: str, start_id: int = 1, end_id: int = None):
    """
    Alignes sample structure to reference structure, saves pdb aligned sample structure as pdb. Returns RMSD.

    Parameters:
        ref_filename (str): reference structure file path
        sample_filename (str): sample structure file path
        outfile (str): outfile name
        start_id (int, optional): beginning residue for alignement
        end_id (int, optional): end residue for alignement, if None it will take all residues of the shorter protein. Default None
    """
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)

    ref_structure = pdb_parser.get_structure("reference", ref_filename)
    sample_structure = pdb_parser.get_structure("sample", sample_filename)

    ref_model = ref_structure[0]
    sample_model = sample_structure[0]

    if end_id == None:
        end_id = min(get_sequence_length(ref_model), get_sequence_length(sample_model))

    atoms_to_be_aligned = range(start_id, end_id + 1)

    ref_atoms = []
    sample_atoms = []
    for ref_chain in ref_model:
        print(ref_chain)
        for ref_res in ref_chain:
            if ref_res.get_id()[1] in atoms_to_be_aligned:
                if 'CA' in ref_res:
                    ref_atoms.append(ref_res['CA'])

    for sample_chain in sample_model:
        for sample_res in sample_chain:
            if sample_res.get_id()[1] in atoms_to_be_aligned:
                if 'CA' in sample_res:
                    sample_atoms.append(sample_res['CA'])

    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms[:end_id], sample_atoms[:end_id])
    super_imposer.apply(sample_model.get_atoms())

    print(f"RMSD: {super_imposer.rms:.3f}")

    io = Bio.PDB.PDBIO()
    io.set_structure(sample_structure)
    io.save(outfile)

    return super_imposer.rms