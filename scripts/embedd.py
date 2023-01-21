import torch
import esm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
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


def divide_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def batch_fasta(fasta_path: str, batch_size: int, dest: str = None):
    """
    Get batches from fasta file with format:
    >{index}|{id}|{activity}
    {Sequence}

    Parameters:
        fasta_path : path to fasta file
        batch_size : batch size for calculation of representations
        dest : destination for saved embeddings. If dest=None embeddings won't be saved

    Returns: three list objects
        [(sequence_label, sequence)], activities
    """
    with open(fasta_path, 'r') as f:
        activities = []
        data = []
        batch = []
        for line in f:
            if line.startswith('>'):
                label = line.split('|')[1]
                activity = float(line.split('|')[2].replace('\n', ''))
            else:
                fasta = line.replace('\n', '')
                activities.append(activity)
                data.append((label, fasta))

        batches = []
        for chunk in divide_list(data, batch_size):
            batches.append(chunk)

        _activities = []
        for chunk in divide_list(activities, batch_size):
            _activities.append(chunk)
        # batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        # activities = []
        return batches, _activities

FASTA_PATH = "../esm/examples/data/P62593.fasta"
DEST = "../example_data/representations/P62593"

batches, activities = batch_fasta(fasta_path=FASTA_PATH, batch_size=10, dest=DEST)

seq_rep = compute_representations(batches[0], dest='.')