import sys
sys.path.append('../../../src/')
from proteusAI.ML.plm.esm_tools import *
import os
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2 = GPT2Model.from_pretrained('gpt2-large')
gpt2.eval()

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
df_path = os.path.join(data_dir, 'processed/02_enzyme_dat_reduced.csv')
fasta_dir = os.path.join(data_dir, 'fastas')

df = pd.read_csv(df_path)

fasta_files = [os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) if f.endswith('.fasta')]

batch_size = 2
for i in range(0, len(fasta_files), batch_size):
    names = [n.split('/')[-1][:-6] for n in fasta_files[i:i + batch_size]]
    seqs = []
    for f in fasta_files[i:i + batch_size]:
        _, s = fasta.load_fasta(f)
        seqs.append(s[0])

    # calculate embeddings of name
    text_embeddings = []
    for n in names:
        row = df[df['protein'] == n]
        text = '; '.join(f'{key}: {value}' for key, value in row.items() if key in ['EC', 'DE', 'AN', 'CA', 'CF', 'CC'])
        encoded_input = tokenizer(text, return_tensors='pt')
        output = gpt2(**encoded_input)
        text_embedding = output['last_hidden_state'][0, :].mean(0)
        text_embeddings.append(text_embedding)
        torch.save(text_embedding, f'../data/embeddings/descriptions/{n}.pt')

    results, batch_lens, batch_labels, alphabet = esm_compute(seqs)
    sequence_representations = get_seq_rep(results, batch_lens)
    for j in range(len(batch_lens)):
        torch.save(sequence_representations[j], f'../data/embeddings/proteins/{n}.pt')

df.to_csv(os.path.join(data_dir, 'processed/03_filtered_enzyme_dat.csv'), index=None, sep=',')