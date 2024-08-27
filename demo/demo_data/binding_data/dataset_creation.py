import os
import pandas as pd

# Load the dataframes
df1 = pd.read_csv('demo/demo_data/binding_data/replicate_1.csv')
df2 = pd.read_csv('demo/demo_data/binding_data/replicate_2.csv')
df3 = pd.read_csv('demo/demo_data/binding_data/replicate_3.csv')

# Merge the dataframes on CDR1H_AA and CDR3H_AA to find common rows
merged_df = pd.merge(df1, df2, on=['CDR1H_AA', 'CDR3H_AA'], suffixes=('_df1', '_df2'))
merged_df = pd.merge(merged_df, df3, on=['CDR1H_AA', 'CDR3H_AA'])

# Calculate the average of fit_KD from the three dataframes
merged_df['fit_KD_avg'] = merged_df[['fit_KD_df1', 'fit_KD_df2', 'fit_KD']].mean(axis=1)

# Select the relevant columns for the final dataframe
result_df = merged_df[['CDR1H_AA', 'CDR3H_AA', 'fit_KD_avg']]

# Group by CDR1H_AA and CDR3H_AA to remove duplicates and average the fit_KD_avg
final_df = result_df.groupby(['CDR1H_AA', 'CDR3H_AA']).agg({'fit_KD_avg': 'mean'}).reset_index()

# Display the final result
final_df

wt_seq = 'EVKLDETGGGLVQPGRPMKLSCVASGFTFSDYWMNWVRQSPEKGLEWVAQIRNKPYNYETYYSDSVKGRFTISRDDSKSSVYLQMNNLRVEDMGIYYCTGSYYGMDYWGQGTSVTVSSAKTTAPSVYPLAPVCGDTTGSSVTLGCLVKGYFPEPVTLTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVTSSTWPSQSITCNVAHPASSTKVDKKIEPRG'

# CDRH1 and CDR3H wild-type sequences
cdr1h_wt = wt_seq[27:37]  # TFSDYWMNWV
cdr3h_wt = wt_seq[99:109] # GSYYGMDYWG

seqs = []
mutant_names = []
for i, row in final_df.iterrows():
    seq = list(wt_seq)
    CDR1H_AA = row['CDR1H_AA']
    CDR3H_AA = row['CDR3H_AA']
    seq[27:37] = list(CDR1H_AA)
    seq[99:109] = list(CDR3H_AA)
    mutated_seq = ''.join(seq)
    seqs.append(mutated_seq)
    
    # Determine the mutations for naming
    mutations = []
    
    # Compare CDR1H sequence
    for j in range(len(cdr1h_wt)):
        if CDR1H_AA[j] != cdr1h_wt[j]:
            mutations.append(f"{cdr1h_wt[j]}{27+j+1}{CDR1H_AA[j]}")
    
    # Compare CDR3H sequence
    for j in range(len(cdr3h_wt)):
        if CDR3H_AA[j] != cdr3h_wt[j]:
            mutations.append(f"{cdr3h_wt[j]}{99+j+1}{CDR3H_AA[j]}")
    
    # Combine mutations into a mutant name
    if mutations:
        mutant_name = ':'.join(mutations)
    else:
        mutant_name = 'WT'
    
    mutant_names.append(mutant_name)

data = {
    'mutant': mutant_names,
    'mutated_sequence': seqs,
    'DMS_score': final_df['fit_KD_avg'].to_list(),
    'DMS_score_bin': [None] * len(mutant_names)
}

results_df = pd.DataFrame(data)
results_df.to_csv('demo/demo_data/DMS/SCFV_HUMAN_Adams_2016_affinity.csv', index=False)

with open("demo/demo_data/DMS/SCFV_HUMAN_Adams_2016_affinity.fasta", "w") as f:
    f.write(f'>SCFV_HUMAN_Adams_2016_affinity\n')
    f.write(f'{wt_seq}')
