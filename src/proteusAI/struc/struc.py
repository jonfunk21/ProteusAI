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


def align(prot1, prot2):
    """
    Superimpose protein 2 on protein 1.

    Args:
        prot1 biotite.structure.AtomArray or path to pdb (str).
        prot2 biotite.structure.AtomArray or path to pdb (str).

    Returns:
        prot1, prot2_superimposed as biotite.structure.AtomArray's
    """
    if type(prot1) == "str":
        try:
            prot1 = strucio.load_structure(prot1)
        except:
            raise ValueError("prot 1 has an unexpected format")
        
    if type(prot2) == "str":
        try:
            prot2 = strucio.load_structure(prot2)
        except:
            raise ValueError("prot 2 has an unexpected format")

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

def rmsd(prot1, prot2):
    """
    Compute rmsd between two proteins.

    Args:
        prot1 biotite.structure.AtomArray or path to pdb (str).
        prot2 biotite.structure.AtomArray or path to pdb (str).

    Returns:
        rmsd (float)
    """

    if type(prot1) == "str":
        try:
            prot1 = strucio.load_structure(prot1)
        except:
            raise ValueError("prot 1 has an unexpected format")
        
    if type(prot2) == "str":
        try:
            prot2 = strucio.load_structure(prot2)
        except:
            raise ValueError("prot 2 has an unexpected format")

    prot1_common = prot1[struc.filter_intersection(prot1, prot2)]
    prot2_common = prot2[struc.filter_intersection(prot2, prot1)]

    rmsd = struc.rmsd(prot1_common, prot2_common)
    print("{:.3f}".format(rmsd))
    return rmsd