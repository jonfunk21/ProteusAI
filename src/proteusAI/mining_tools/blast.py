# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from tempfile import gettempdir
import biotite.sequence.io.fasta as fasta
import biotite.application.blast as blast
import biotite.database.entrez as entrez


# TODO: change email
def search_related_sequences(
    query: str,
    program: str = "blastp",
    database: str = "nr",
    obey_rules: bool = True,
    mail: str = "johnyfunk@gmail.com",
):
    """
    Search for related sequences using the plast web app.

    Parameters:
        query (str):
            Query sequence
        program (str, optional):
            The specific BLAST program. One of 'blastn', 'megablast', 'blastp', 'blastx', 'tblastn' and 'tblastx'.
        database (str, optional):
            The NCBI sequence database to blast against. By default it contains all sequences (`database`='nr'`).
        obey_rules (bool, optional):
            If true, the application raises an :class:`RuleViolationError`, if the server is contacted too often,
            based on the NCBI BLAST usage rules. (Default: True)
        mail : str, optional
            If a mail address is provided, it will be appended in the
            HTTP request. This allows the NCBI to contact you in case
            your application sends too many requests.

    Returns:
        tuple: two lits, the first containing the hits and the second the hit sequences

    Example:
        hits, hit_seqs = blast_related_sequences(query=sequence, database='swissprot')
    """
    # Search only the UniProt/SwissProt database
    blast_app = blast.BlastWebApp(
        program=program,
        query=query,
        database=database,
        obey_rules=obey_rules,
        mail=mail,
    )
    blast_app.start()
    blast_app.join()
    alignments = blast_app.get_alignments()
    # Get hit IDs for hits with score > 200
    hits = []
    for ali in alignments:
        if ali.score > 200:
            hits.append(ali.hit_id)
    # Get the sequences from hit IDs
    hit_seqs = []
    for hit in hits:
        file_name = entrez.fetch(hit, gettempdir(), "fa", "protein", "fasta")
        fasta_file = fasta.FastaFile.read(file_name)
        hit_seqs.append(fasta.get_sequence(fasta_file))

    return hits, hit_seqs
