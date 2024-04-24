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


def load_struc(prot):
    """
    Load a protein structure.

    Args:
        prot biotite.structure.AtomArray or path to pdb (str).
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