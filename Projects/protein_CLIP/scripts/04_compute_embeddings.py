import sys
sys.path.append('../../../src/')
from proteusAI.ml_tools.esm_tools.esm_tools import *
import os
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model

# natural language model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2 = GPT2Model.from_pretrained('gpt2-large')
gpt2.eval()

# protein language model
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
df_path = os.path.join(data_dir, 'processed/03_filtered_enzyme_dat.csv')
fasta_dir = os.path.join(data_dir, 'fastas')

df = pd.read_csv(df_path)
filtered = df['protein'].to_list()
fasta_files = [os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) if (f.endswith('.fasta') and f[:-6] in filtered)]

# calculate embeddings of name
drop_ec = []
for ec in set(df['EC'].to_list()):
    text_embedding_path = f'../data/embeddings/descriptions/EC_{ec.replace(".","_")}.pt'
    if not os.path.exists(text_embedding_path):  # check if file already exists
        rows = df[df['EC']==ec]
        text = '; '.join(f'{key}: {value}' for key, value in rows.iloc[0].items() if key in ['EC', 'DE', 'AN', 'CA', 'CF', 'CC'])
        if len(text) > 1024:
            drop_ec.append(ec)
        else:
            try:
                encoded_input = tokenizer(text, return_tensors='pt')
                output = gpt2(**encoded_input)
                text_embedding = output['last_hidden_state'][0, :].mean(0)
                torch.save(text_embedding, text_embedding_path)
            except:
                drop_ec.append(ec)

df = df[~df['EC'].isin(drop_ec)]

batch_size = 1
for i in range(0, len(fasta_files), batch_size):
    names = [n.split('/')[-1][:-6] for n in fasta_files[i:i + batch_size]]
    seqs = []
    for f in fasta_files[i:i + batch_size]:
        _, s = fasta.load_fasta(f)
        seqs.append(s[0])

    results, batch_lens, batch_labels, alphabet = esm_compute(seqs, model=model)
    sequence_representations = get_seq_rep(results, batch_lens)
    for j, n in enumerate(names):  # we need to use enumerate here to get the correct name for each sequence representation
        seq_rep_path = f'../data/embeddings/proteins/{n}.pt'
        if not os.path.exists(seq_rep_path):  # check if file already exists
            torch.save(sequence_representations[j], seq_rep_path)

df.to_csv(os.path.join(data_dir, 'processed/04_dataset.csv'), index=False, sep=',')