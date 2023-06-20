import os
import pandas as pd
sys.path.insert(0, '../../src')
import proteusAI.io_tools as io_tools

# encoding type ohe vs blosum
encoding_type = 'ohe'

# script path
script_path = os.path.dirname(os.path.realpath(__file__))
msa_path = os.path.join(script_path, MSA)

msa_results = io_tools.load_all_fastas(msa_path)

encodings = {}
for key, value in msa_results.items():
    sequences = value[1]
    e = torch_tools.one_hot_encoder(sequences, alphabet)
    encoded_sequences = [encoding for encoding in e]
    encodings[key] = pd.DataFrame({
        'label':value[0], 
        'x':encoded_sequences
    })