from Bio import SeqIO
import requests as r
from io import StringIO


def get_protein_sequence(uniprot_id: str):
    """
    This function takes a UniProt ID as input and returns the corresponding sequence record.

    Parameters:
        uniprot_id (str): The UniProt identifier of the protein of interest

    Returns:
        list: List with results as Bio.SeqRecord.SeqRecord object

    Example:
        ASMT = uniprot.get_protein_sequence('P46597')
        sequence_string = str(ASMT[0].seq)
    """
    base_url = "http://www.uniprot.org/uniprot/"
    current_url = base_url + uniprot_id + ".fasta"
    response = r.post(current_url)
    c_data = ''.join(response.text)

    seq = StringIO(c_data)
    p_seq = list(SeqIO.parse(seq, 'fasta'))
    return p_seq