import py3Dmol
from string import ascii_uppercase, ascii_lowercase

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