# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from Bio import SeqIO
import requests
from io import StringIO
from hashlib import md5

def get_protein_sequence(uniprot_id: str) -> str:
    """
    This function takes a UniProt ID as input and returns the corresponding sequence record.

    Parameters:
        uniprot_id (str): The UniProt identifier of the protein of interest

    Returns:
        list: List with results as Bio.SeqRecord.SeqRecord object

    Example:
        ASMT_representations = uniprot.get_protein_sequence('P46597')
        sequence_string = str(ASMT_representations[0].seq)
    """
    base_url = "http://www.uniprot.org/uniprot/"
    current_url = base_url + uniprot_id + ".fasta"
    response = requests.post(current_url)
    c_data = ''.join(response.text)

    seq = StringIO(c_data)
    p_seq = list(SeqIO.parse(seq, 'fasta'))
    return p_seq


def get_uniprot_id(sequence: str) -> str:
    """
    This function takes in a protein sequence string and returns the corresponding UniProt ID.

    Parameters:
        sequence (str): The protein sequence string

    Returns:
        str: The UniProt ID of the protein, if it exists in UniProt. None otherwise.
    """
    h = md5(sequence.encode()).digest().hex()
    requestURL = f"https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size=100&md5={h}"  # 5e2c446cc1c54ee4406b9f6683b7f98d
    r = requests.get(requestURL, headers={"Accept": "application/json"})

    if not r.ok:
        return None
    data = r.json()

    if len(data) == 0:
        return None
    else:
        return data[0]["accession"]