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
from openmm import unit
import openmm as mm
from openmm import app
from openmm.unit import *  # noqa: F403
from openmm.app import PDBFile
import tempfile
import os
import numpy as np


alphabet_list = list(ascii_uppercase + ascii_lowercase)
pymol_color_list = [
    "#33ff33",
    "#00ffff",
    "#ff33cc",
    "#ffff00",
    "#ff9999",
    "#e5e5e5",
    "#7f7fff",
    "#ff7f00",
    "#7fff7f",
    "#199999",
    "#ff007f",
    "#ffdd5e",
    "#8c3f99",
    "#b2b2b2",
    "#007fff",
    "#c4b200",
    "#8cb266",
    "#00bfbf",
    "#b27f7f",
    "#fcd1a5",
    "#ff7f7f",
    "#ffbfdd",
    "#7fffff",
    "#ffff7f",
    "#00ff7f",
    "#337fcc",
    "#d8337f",
    "#bfff3f",
    "#ff7fff",
    "#d8d8ff",
    "#3fffbf",
    "#b78c4c",
    "#339933",
    "#66b2b2",
    "#ba8c84",
    "#84bf00",
    "#b24c66",
    "#7f7f7f",
    "#3f3fa5",
    "#a5512b",
]

amino_acid_mapping = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Adding a generic mapping for unknowns
    "UNK": "X",  # Unknown
    "SEC": "U",  # Selenocysteine, sometimes considered the 21st amino acid
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
    if isinstance(prot, struc.AtomArray):
        prot = prot

    elif isinstance(prot, str):
        try:
            prot = strucio.load_structure(prot)
        except Exception as e:
            raise ValueError(f"prot 1 has an unexpected format. Error {e}")

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

    chains = list(set(prot.chain_id))  # type: ignore
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
        res_ids = list(set(prot[prot.chain_id == chain].res_id))
        residues = [
            amino_acid_mapping.get(prot[prot.res_id == r].res_name[0], "X")
            for r in res_ids
        ]
        sequences[chain] = "".join(residues)

    return sequences


def get_contacts(structure, chain=None, target="protein", dist=7.0):
    """
    Get contacts within a protein structure or between a protein and ligands, based on the specified chain.

    Args:
        structure (biotite.structure.AtomArray): The complete protein structure.
        chain (str, optional): Specific chain for which to compute the contacts.
            Default is None, which will use the first protein chain if not specified.
        target (str): Specify 'protein' for protein-protein contacts or 'ligand'
            for protein-ligand contacts. Default is 'protein'.
        dist (float): Specified distance threshold in Angstroms. Default is 7.

    Returns:
        list: Unique residue IDs in contact from the specified chain.
    """
    if chain is None:
        chain = structure.chain_id[0]  # Default to the first chain if none specified

    if target == "protein":
        # Select atoms from all other chains, excluding heteroatoms if needed
        target_atoms = structure[(structure.chain_id != chain) & (~structure.hetero)]
    elif target == "ligand":
        # Assuming ligands are identified as heteroatoms
        target_atoms = structure[structure.hetero]
    else:
        raise ValueError("Invalid target specified. Use 'protein' or 'ligand'.")

    # Select the atoms in the target chain, excluding heteroatoms if needed
    chain_atoms = structure[(structure.chain_id == chain) & (~structure.hetero)]

    # Initialize cell list with target atoms
    cell_list = struc.CellList(target_atoms, cell_size=dist)
    contact_indices = cell_list.get_atoms(chain_atoms.coord, radius=dist)

    # Determine contact residue IDs
    contact_residue_indices = np.where((contact_indices != -1).any(axis=1))[0]
    contact_res_ids = chain_atoms.res_id[contact_residue_indices]

    # Removing duplicate IDs by converting to a set, then back to sorted list
    unique_contact_res_ids = sorted(set(contact_res_ids))

    return unique_contact_res_ids


def compute_chi_angles(protein, res_ids):
    """
    Compute the chi angles for specified residues in a protein by chain.

    Args:
        protein: biotite.structure.AtomArray or path to pdb (str).
        res_ids (dict): Dictionary where keys are chain identifiers and values are lists of residue IDs
                        to compute chi angles for in each chain.
        chi_atom_names (dict): Dictionary where keys are residue names and values are lists of lists of
                               atom names involved in each chi angle.

    Returns:
        A dictionary where keys are tuples of (chain identifier, residue name, residue ID) and values
        are lists of chi angles.
    """

    # Dictionary containing atom names involved in chi angle calculations for each amino acid
    chi_atom_names = {
        "ALA": [],  # Alanine has no chi angles
        "ARG": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "NE"],
            ["CG", "CD", "NE", "CZ"],
        ],
        "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
        "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
        "CYS": [["N", "CA", "CB", "SG"]],
        "GLN": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "OE1"],
        ],
        "GLU": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "OE1"],
        ],
        "GLY": [],  # Glycine has no chi angles
        "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
        "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
        "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "LYS": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "CE"],
            ["CG", "CD", "CE", "NZ"],
        ],
        "MET": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "SD"],
            ["CB", "CG", "SD", "CE"],
        ],
        "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
        "SER": [["N", "CA", "CB", "OG"]],
        "THR": [["N", "CA", "CB", "OG1"]],
        "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "VAL": [["N", "CA", "CB", "CG1"]],
    }

    if isinstance(protein, str):
        protein = load_struc(protein)  # Assuming load_struc loads a structure

    chi_angles = {}
    for chain, residues in res_ids.items():
        for res_id in residues:
            # Filter protein by chain and residue ID
            mask = (protein.chain_id == chain) & (protein.res_id == res_id)
            res_protein = protein[mask]
            res_name = res_protein.res_name[0] if len(res_protein) > 0 else None

            if res_name and res_name in chi_atom_names:
                chi_list = []
                for atom_names in chi_atom_names[res_name]:
                    if all(atom in res_protein.atom_name for atom in atom_names):
                        atoms = [
                            res_protein[res_protein.atom_name == atom][0]
                            for atom in atom_names
                        ]
                        chi = struc.dihedral(*atoms)
                        chi_list.append(chi)
                if chi_list:
                    chi_angles[(chain, res_name, res_id)] = chi_list

    return chi_angles


def delta_chi(chi_angles_1, chi_angles_2):
    """
    Compute the delta between two sets of chi angles.

    Args:
        chi_angles_1 (dict): Dictionary with keys as (residue name, residue ID)
                             and values as lists of chi angles from the first structure.
        chi_angles_2 (dict): Dictionary with keys as (residue name, residue ID)
                             and values as lists of chi angles from the second structure.

    Returns:
        float: Sum of absolute differences between corresponding chi angles.
    """
    total_difference = 0.0

    for key in chi_angles_1:
        if key in chi_angles_2:
            angles_1 = chi_angles_1[key]
            angles_2 = chi_angles_2[key]
            # Ensure both lists of angles have the same length
            if len(angles_1) == len(angles_2):
                for a1, a2 in zip(angles_1, angles_2):
                    total_difference += abs(a1 - a2)
            else:
                raise ValueError(f"Mismatch in number of chi angles for residue {key}.")
        else:
            raise ValueError(f"Residue {key} not found in both structures.")

    return total_difference


def show_pdb(
    pdb_path,
    color="confidence",
    vmin=50,
    vmax=90,
    chains=None,
    Ls=None,
    size=(800, 480),
    show_sidechains=False,
    show_mainchains=False,
    highlight=None,
    sticks=None,
):
    """
    This function displays the 3D structure of a protein from a given PDB file in a Jupyter notebook.
    The protein structure can be colored by chain, rainbow, pLDDT, or confidence value. The size of the
    display can be changed. The sidechains and mainchains can be displayed or hidden.
    Additional functionality includes highlighting specific residues within specific chains by passing
    a dictionary with chains as keys and lists of residue numbers as values, and displaying ligands
    as sticks and ions as spheres.
    Parameters:
        pdb_path (str): The filename of the PDB file that contains the protein structure.
        color (str, optional): The color scheme for the protein structure. Can be "chain", "rainbow", "pLDDT", "confidence". Defaults to "rainbow".
        vmin (float, optional): The minimum value of pLDDT or confidence value. Defaults to 50.
        vmax (float, optional): The maximum value of pLDDT or confidence value. Defaults to 90.
        chains (int, optional): The number of chains to be displayed. Defaults to None.
        Ls (list, optional): A list of the chains to be displayed. Defaults to None.
        size (tuple, optional): The size of the display window. Defaults to (800, 480).
        show_sidechains (bool, optional): Whether to display the sidechains. Defaults to False.
        show_mainchains (bool, optional): Whether to display the mainchains. Defaults to False.
        highlight (dict, optional): A dictionary with chains as keys and lists of residue numbers to be highlighted as values. Defaults to None.
        sticks (list, optional): A list of residues that should be displayed as sticks. Default to None.
    Returns:
        view: The 3Dmol view object that displays the protein structure.
    """

    with open(pdb_path) as ifile:
        system = "".join([x for x in ifile])

    view = py3Dmol.view(
        js="https://3dmol.org/build/3Dmol.js", width=size[0], height=size[1]
    )

    view.addModelsAsFrames(system)

    # Apply color styles based on function arguments
    if color == "pLDDT" or color == "confidence":
        view.setStyle(
            {},
            {
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "rwb",
                        "min": vmin,
                        "max": vmax,
                    }
                }
            },
        )
    elif color == "rainbow":
        view.setStyle({}, {"cartoon": {"color": "spectrum"}})
    else:
        # Set default style as white
        view.setStyle({}, {"cartoon": {"color": "white"}})

    # Highlight specific residues per chain
    if highlight and isinstance(highlight, dict):
        for chain, residues in highlight.items():
            for resi in residues:
                highlight_style = {
                    "stick" if resi in (sticks or []) else "cartoon": {
                        "colorscheme": "blueCarbon",
                        "radius": 0.3,
                    }
                }
                view.addStyle({"chain": chain, "resi": str(resi)}, highlight_style)

    if sticks:
        stick_style = {"stick": {"radius": 0.3}}
        for resi in sticks:
            view.addStyle({"resi": str(resi)}, stick_style)

    # Display ligands as sticks and ions as spheres
    view.addStyle(
        {"hetflag": True, "bonds": 0, "atom": "not O"},
        {"sphere": {"colorscheme": "ionic", "radius": 0.5}},
    )
    view.addStyle(
        {"hetflag": True}, {"stick": {"colorscheme": "organic", "radius": 0.3}}
    )  # Display organic ligands as sticks

    if show_sidechains:
        BB = ["C", "O", "N"]
        view.addStyle(
            {
                "and": [
                    {"resn": ["GLY", "PRO"], "invert": True},
                    {"atom": BB, "invert": True},
                ]
            },
            {"stick": {"colorscheme": "WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "GLY"}, {"atom": "CA"}]},
            {"sphere": {"colorscheme": "WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "PRO"}, {"atom": ["C", "O"], "invert": True}]},
            {"stick": {"colorscheme": "WhiteCarbon", "radius": 0.3}},
        )

    if show_mainchains:
        BB = ["C", "O", "N", "CA"]
        view.addStyle(
            {"atom": BB}, {"stick": {"colorscheme": "WhiteCarbon", "radius": 0.3}}
        )

    view.zoomTo()

    return view


def relax_pdb(file, dest="outputs/struc/relaxed"):
    """
    Processes and minimizes a protein structure file using molecular dynamics.

    The function takes a PDB file, corrects its structure by adding missing residues and atoms,
    and then performs an energy minimization. The minimization is carried out under specified
    conditions using a Langevin integrator and a given force field. If CUDA is available,
    the function attempts to use it for the computations; otherwise, it falls back to CPU.

    Parameters:
        file (str): The path to the PDB file to be processed.
        dest (str, optional): The directory where the relaxed PDB file will be saved.
                              Default is 'outputs/struc/relaxed'.

    Returns:
        str: The path to the relaxed PDB file, saved in the specified destination directory.

    Raises:
        EnvironmentError: If the CUDA platform is not available and the function falls back to CPU,
                           this is not an error per se but might be relevant for performance-sensitive applications.

    Example:
        relaxed_path = relax_pdb('input/protein.pdb')
        print(f'Relaxed structure saved to {relaxed_path}')

    Note:
        This function uses the Amber14 force field and assumes a pH of 7.0 for protonation states.
    """

    try:
        from pdbfixer import PDBFixer
    except Exception as e:
        raise ValueError(
            f"Relaxation of protein structures requires PDBFixer: Please install through conda:\nconda install conda-forge::pdbfixer. Error {e}"
        )

    name = file.split("/")[-1].split(".")[0]

    fixer = PDBFixer(filename=file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(
        7.0
    )  # pH value to decide protonation state of HIS, ASP, GLU

    # Use a temporary file instead of a fixed file name
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdb", mode="w"
    ) as temp_file:
        PDBFile.writeFile(fixer.topology, fixer.positions, temp_file)
        temp_file_path = temp_file.name  # Store the temporary file name to use it later

    # Now read the temporary PDB file
    pdb = app.PDBFile(temp_file_path)

    # Prepare the force field
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # Create a system
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
    )

    # Create an integrator
    integrator = mm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds
    )

    # Try to use CUDA, otherwise fallback to CPU
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {"CudaPrecision": "mixed"}
    except Exception:
        # Fallback to CPU if CUDA is not available
        platform = mm.Platform.getPlatformByName("CPU")
        properties = {}  # CPU does not need special properties

    # Create a simulation context
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

    # Set the initial positions
    simulation.context.setPositions(pdb.positions)

    # Minimize the energy
    simulation.minimizeEnergy()

    # Get the final positions after minimization
    positions = simulation.context.getState(getPositions=True).getPositions()

    # Save the minimized structure to a new PDB file
    relaxed_pdb = os.path.join(dest, f"{name}_relaxed.pdb")
    app.PDBFile.writeFile(pdb.topology, positions, open(relaxed_pdb, "w"))

    return relaxed_pdb
