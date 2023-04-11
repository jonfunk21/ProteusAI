import torch
import esm
import os
import time
import argparse
import sys
sys.path.append('../')

def compute_representations(data: list, dest: str = None, device: str = 'cuda', rep_layer: int = 33, seq_rep_only=True):
    '''
    generate sequence representations using esm2_t33_650M_UR50D.
    The representation are of size 1280.

    Parameters:
        data (list): list of tuples containing sequence labels (str) and protein sequences 
                     (str or biotite.sequence.ProteinSequence) (label, sequence)
        dest (str): destination where embeddings are saved. Default None (won't save if dest is None).
        device (str): device used for calculation or representations. Default "cuda". 
                      other options are "cpu", or "mps" for M1/M2 chip
        rep_layer (int): representation layer from which the sequence is extracted. Default 33 (final layer)
        seq_rep_only (bool): if seq_rep_only then only sequence representations will be saved, else
            logits, representations, attentions and contacts will also be saved. If seq_rep_only is False,
            directories will be created for logits, representations, attentions and contacts at destination.

    Returns: representations (list) of sequence representation if seq_rep_only.
        [seq_rep1, ..., seq_repN]
        Else:
        [(seq_rep1, logits1, attentions1, contacts1),...(seq_repN, logitsN, attentionsN, contactsN)]

    Example:
        data = [("protein1", "AGAVCTGAKLI"), ("protein2", "AGHRFLIKLKI")]
        representations = get_sequence_representations(data)

    '''
    # check datatype of data
    if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
        pass # all elements are strings
    else:
        data = [(x[0], str(x[1])) for x in data]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=True)

    token_representations = results["representations"][rep_layer]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

    if not seq_rep_only:
        logits = results["logits"][rep_layer]
        attentions = ["attentions"][rep_layer]
        contacts = ["contacts"][rep_layer]
        if not os.path.exists(os.path.join(dest, 'logits')):
            os.mkdir(os.path.join(dest, 'logits'))
        if not os.path.exists(os.path.join(dest, 'representations')):
            os.mkdir(os.path.join(dest, 'representations'))
        if not os.path.exists(os.path.join(dest, 'attentions')):
            os.mkdir(os.path.join(dest, 'attentions'))
        if not os.path.exists(os.path.join(dest, 'contacts')):
            os.mkdir(os.path.join(dest, 'contacts'))

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    if dest is not None:
        for i in range(len(sequence_representations)):
            if seq_rep_only:
                _dest = os.path.join(dest, batch_labels[i])
                torch.save(sequence_representations[i], _dest + '.pt')
            else:
                _dest = os.path.join(dest, "representations", batch_labels[i])
                torch.save(sequence_representations[i], _dest + '.pt')
                _dest = os.path.join(dest, "logits", batch_labels[i])
                torch.save(logits[i], _dest + '.pt')
                _dest = os.path.join(dest, "attentions", batch_labels[i])
                torch.save(attentions[i], _dest + '.pt')
                _dest = os.path.join(dest, "contacts", batch_labels[i])
                torch.save(contacts[i], _dest + '.pt')

    if seq_rep_only:
        return sequence_representations
    else:
        return list(zip(sequence_representations, logits, attentions, contacts))


def divide_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def batchify_fasta(fasta_path: str, batch_size: int):
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

        return batches, _activities

def extract_sequences(file_name):
    """
    Given a fasta file with the format:

    >name1
    sequence1
    >name2
    sequence2

    this function will return the names and sequences of the fasta as lists.

    Parameters:
        file_name (str): name and path to file

    Returns:
        tuple: (names [list], sequences [list])
    """
    names = []
    sequences = []
    with open(file_name, 'r') as f:
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:
                    sequences.append(current_sequence)
                names.append(line[1:])
                current_sequence = ""
            else:
                current_sequence += line
        sequences.append(current_sequence)
    return names, sequences
    

def batch_embedd(fasta_path: str, dest: str, batch_size: int = 10, rep_layer: int = 33):
    """
    takes fasta files in a specific format and embedds all the sequences using the 
    esm2_t33_650M_UR50D model. The embeddings are saved in the destination folder.
    Fasta format:
    
        >{index}|{id}|{activity}
        {Sequence}
        >{index}|{id}|{activity}
        {Sequence}
    
    Parameters:
        fasta_path (str): path to fasta file
        dest (str): destination
        batch_size (int): batch size for computation
        rep_layer (int) : representation layer from which the sequence is extracted. Default 33 (final layer)

    Returns:
        list of sequence representations
    """

    if not os.path.exists(dest):
        os.makedirs(dest)

    start_time = time.time()
    batches, activities = batchify_fasta(fasta_path=fasta_path, batch_size=batch_size)

    for batch in batches:
        _ = compute_representations(batch, dest=dest, device=device, rep_layer=rep_layer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
     '''
     Description:
     Creates embeddings for a fasta file using esm.pretrained.esm2_t33_650M_UR50D() 
     and saves them at the destination.
     
     if the activity flag (-a or --activity) flag is set, the fasta should have the
     following format:
         >{index}|{id}|{activity}
         {Sequence}
     
     saves files as:
     <path to dest>/{id}.pt
     
     example:
     python3 embedd_esm.py -f input.fasta -d out_dir -b 26
     ''',
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-f', '--fasta', help='path to fasta file', required=True, type=str, default=None)
    parser.add_argument('-d', '--dest', help='path to destination', required=True, type=str,default=None)
    parser.add_argument('-b', '--batch_size', help='batch size for computation of representations', type=int, default=26)
    parser.add_argument('-a', '--activity', help='Set true if activity values are provided', type=bool, default=False)
    parser.add_argument('-m', '--model', help='select model for embedding {esm2, esm1v}, Uses esm1v if esm2 not selected', type=str, default=None)
    parser.add_argument('-r', '--rep_layer', help='choose representation layer of model. Default 33 (finla layer)', type=int, default=33)
    parser.add_argument('--seq_rep_only', help='Only save sequence representation. If false save logits, attentions and contacts in addition', type=bool, default=True)
    args = parser.parse_args()


    FASTA_PATH = args.fasta
    DEST = args.dest
    BATCH_SIZE = int(args.batch_size)
    MODEL = args.model
    REP_LAYER = args.rep_layer
    ACTIVITY = args.activity
    SEQ_REP_ONLY = args.seq_rep_only

    if DEST and not os.path.exists(DEST):
        os.mkdir(DEST)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # on M1 if mps available
    if device == torch.device(type='cpu'):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    print('Using device:', device)

    # THIS SEEMS TO MAKE PROBLEMS
    if MODEL == 'esm2':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()


    model.to(device)
    model.eval()  # disables dropout for deterministic results

    batch_converter = alphabet.get_batch_converter()

    if ACTIVITY:
        batch_converter = alphabet.get_batch_converter()
        batch_embedd(FASTA_PATH, DEST, BATCH_SIZE, REP_LAYER)
    else:
        names, seqs = extract_sequences(FASTA_PATH)
        data = list(zip(names, seqs))
        for i in range(0, len(data), BATCH_SIZE):
            r = compute_representations(data[i:i + BATCH_SIZE], dest=DEST, device=str(device), rep_layer=REP_LAYER, seq_rep_only=SEQ_REP_ONLY)