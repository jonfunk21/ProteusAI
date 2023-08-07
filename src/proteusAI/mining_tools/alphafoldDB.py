# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import requests
import os

def get_AF2_pdb(protein_id: str, out_path: str) -> bool:
    """
    This function takes in a UniProt ID and an output path and downloads the corresponding AlphaFold model
    from the EBI AlphaFold database in PDB format.

    Parameters:
    protein_id (str): The UniProt ID of the protein
    out_path (str): The path to save the PDB file to. The directory containing the output file will be created if it does not exist.

    Returns:
    bool: True if the PDB file was downloaded successfully, False otherwise.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    requestURL = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v3.pdb"
    r = requests.get(requestURL)

    if r.status_code == 200:
        with open(out_path, 'wb') as f:
            f.write(r.content)
            return True
    else:
        return False
