from Bio.Blast import NCBIWWW
from Bio import SeqIO
from Bio.Blast import NCBIXML
from Bio import Entrez
import os


def search_related_sequences(query_sequence, db: str = "nr"):
    """
    Search for related protein sequences using blast.

    Parameters:
        query_sequence (Bio.SeqRecord.SeqRecord): query sequence
        db (str): database, default nr (non redundant)

    Returns:
        list of hits

    Examples:
        query_sequence = SeqIO.read("example.fasta", "fasta")
        hits = search_related_sequences(query_sequence, "nr")
    """
    # Run a BLAST search against the specified database
    result_handle = NCBIWWW.qblast("blastp", db, query_sequence.format("fasta"))

    # Parse the BLAST results
    blast_records = NCBIXML.parse(result_handle)

    # Extract the IDs and descriptions of the top 10 hits
    hits = []
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                hits.append(alignment)

    return hits


def fastas_from_blast(hits: list, dest: str, email: str = "your@email.com", db: str = "protein"):
    """
    Download fastas from blast search result.

    Parameters:
        hits (list): list of alignements (alignments type = : Bio.Blast.Record.Alignment)
        email (str): email adress
        dest (str): destination where files will be saved

    Examples:
        query_sequence = SeqIO.read("example.fasta", "fasta")
        hits = search_related_sequences(query_sequence, "nr")
        fastas_from_blast(hits, './results/', 'your@email.com')
    """

    assert email != 'your@email.com', "Error: provide a valid email"
    hit_ids = [hit.hit_id for hit in hits]

    # Retrieve the sequences of the hits from the database
    Entrez.email = email
    handle = Entrez.efetch(db=db, id=hit_ids, rettype="fasta")
    records = list(SeqIO.parse(handle, "fasta"))

    if not os.path.exists(dest):
        os.makedirs(dest)

    # Save the sequences to FASTA files
    for record in records:
        out_f = os.path.join(dest, f"{record.id}.fasta")
        with open(out_f, "w") as f:
            SeqIO.write(record, f, "fasta")