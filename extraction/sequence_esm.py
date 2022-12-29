import torch
import esm
import os

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()  # disables dropout for deterministic results
batch_converter = alphabet.get_batch_converter()

def compute_representations(data: tuple, dest: str = None):
    '''
    generate sequence representations using esm2_t33_650M_UR50D.
    The representation are of size 1280.

    Parameters:
        data : tuple of label for sequence and protein sequence (label, sequence)

    Returns: representations (list)

    Example:
        data = [("protein1", "AGAVCTGAKLI"), ("protein2", "AGHRFLIKLKI")]
        representations = get_sequence_representations(data)

    '''
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    if dest is not None:
        for i in range(len(sequence_representations)):
            _dest = os.path.join(dest, batch_labels[i])
            torch.save(sequence_representations[i], _dest + '.pt')

    return sequence_representations


def extract_from_fasta(fasta_path: str, batch_size: int, dest: str = None):
    """
    Get sequence representation for every sequence in fasta file.
    Fasta files are expected to follow this format:

    >{index}|{mutation_id}|{effect}
    {seq}

    If a destination is provided, the representations will be saved at the
    destination based on their mutation_id.

    Parameters:
        fasta_path : path to fasta file
        batch_size : batch size for calculation of representations
        dest : destination for saved embeddings. If dest=None embeddings won't be saved

    Returns: three list objects
        [(sequence_label, sequence)], activities, sequence_representations

    Example:
        example.fasta

            >0|P2A|0.5
            AAGLIKHIVDSEE

            >0|I5H|0.7
            APGLHKHIVDSEE

        data, activities, sequence_representations = extract_from_fasta(
                                                        fasta_path="./example.fasta",
                                                        batch_size=128,
                                                        dest="./")
    """
    with open(fasta_path, 'r') as f:
        activities = []
        data = []
        sequence_representations = []
        batch = []
        for line in f:
            if line.startswith('>'):
                label = line.split('|')[1]
                activity = line.split('|')[2]
            else:
                fasta = line
                activities.append(activity)
                data.append((label, fasta))

        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # Process each chunk
        for batch in batches:
            representations = compute_representations(batch, dest)
            sequence_representations += representations

        return data, activities, sequence_representations