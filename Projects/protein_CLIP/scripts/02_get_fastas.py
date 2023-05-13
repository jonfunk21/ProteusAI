import requests as r
from Bio import SeqIO
from io import StringIO
import os
import pandas as pd
from tqdm import tqdm

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
uniprot_files = os.path.join(processed_data_dir, '01_enzyme_dat.csv')
dest = os.path.join(data_dir, 'fastas')

if not os.path.exists(dest):
    os.makedirs(dest)

def get_fasta(cID, dest):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+cID+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)

    Seq=StringIO(cData)
    pSeq=list(SeqIO.parse(Seq,'fasta'))
    fasta = pSeq[0].format('fasta')

    file_name = os.path.join(dest, cID + '.fasta')

    with open(file_name, 'w') as f:
        seq = ''
        lines = fasta.split('\n')
        for i, line in enumerate(lines):
            if i == 0:
                pass
            else:
                seq = seq + line
        f.write(seq)

data = pd.read_csv(uniprot_files)
cIDs = data.protein.to_list()

for i in range(1):
    # multiple attempts of downloading all files in case of server errors
    downloaded = os.listdir(dest)
    downloaded = [file[:-6] for file in downloaded]
    cIDs = [x for x in cIDs if x not in downloaded]
    for cID in tqdm(cIDs):
        if cID not in downloaded:
            try:
                get_fasta(cID, dest)
            except:
                pass

data = data[~data['protein'].isin(cIDs)]
data.to_csv(f'{processed_data_dir}/02_enzyme_dat_reduced.csv', index=None, sep=',')

print(f'downloaded sequences: {len(downloaded)}')
print(f'sequences failed to download: {len(cIDs)}')