import sys
sys.path.append('../../../src/')
from proteusAI.io_tools import *
import os
import pandas as pd

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
df_path = os.path.join(data_dir, 'processed/02_enzyme_dat_reduced.csv')
fasta_dir = os.path.join(data_dir, 'fastas')

df = pd.read_csv(df_path)

fasta_files = [os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) if f.endswith('.fasta')]
names = [n.split('/')[-1][:-6] for n in fasta_files]

for i, fasta in enumerate(fasta_files):
    _, seq = load_fasta(fasta)

    # drop if sequence is to long
    if len(seq[0]) > 1024:
        df = df[df['protein'] != names[i]]

max_entries_per_ec = 100
df = df.groupby('EC').head(max_entries_per_ec)

df.to_csv(os.path.join(data_dir, 'processed/03_filtered_enzyme_dat.csv'), index=None, sep=',')