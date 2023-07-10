import sys
sys.path.insert(0, '../../src')
from proteusAI.ml_tools.esm_tools import *
import os
import pandas as pd

# paths
script_path = os.path.dirname(os.path.realpath(__file__))
results_path = os.path.join(script_path, 'zero_shot')
plots_path = os.path.join(script_path, 'plots')
DMS_path = os.path.join(script_path, '../data/DMS_enzymes')
datasets_path = os.path.join(DMS_path, 'datasets')

# make destinations
os.makedirs(results_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

# specific plotting functions
def plot_entropy_with_highlighted_mutations(name: str, seq:str, entropy:torch.Tensor, data: pd.DataFrame, dest:str, show: bool=False):
    # Find positions with positive effect
    mutated_positions = []
    for index, row in data.iterrows():
        # binary score of one is an improved mutant
        if row["DMS_score_bin"] == 1:
            position = int(row["mutant"][1:-1]) - 1  # 0-indexed
            mutated_positions.append(position)

    # only unique positions
    mutated_positions = list(set(mutated_positions))

    # title and destination for plot
    title = f"Per position entropy of {name}"
    dest = os.path.join(dest, f"{name}_entropy_highlighted.png")

    # Plot the per position entropy with highlighted mutated positions
    plot_per_position_entropy(entropy, seq, highlight_positions=mutated_positions, show=show, title=title, dest=dest)



def plot_heatmap_with_highlighted_mutations(name: str, data: pd.DataFrame, alphabet: esm.data.Alphabet, heatmap: torch.Tensor, dest: str, heatmap_type: str = "mmp", show: bool=False):
    # Select color scheme and title based on the heatmap type
    if heatmap_type == "p":
        color_sheme = "b"
        title = f"Per probability distribution probability of {name}"
    elif heatmap_type == "mmp":
        color_sheme = "rwb"
        title = f"Per position masked marginal probability of {name}"
    else:
        raise ValueError("Invalid heatmap_type. Accepted values are 'pd' or 'mmp'.")

    # Find positions with positive effect
    mutations = {}
    for index, row in data.iterrows():
        # binary score of one is an improved mutant
        if row["DMS_score_bin"] == 1:
            position = int(row["mutant"][1:-1]) - 1  # 0-indexed
            mutated_residue = row["mutant"][-1]
            mutations[position] = mutated_residue

    # Only unique positions
    mutations = {k: v for k, v in sorted(mutations.items())}

    # Destination for plot
    dest = os.path.join(dest, f"{name}_heatmap_highlighted.png")

    plot_heatmap(heatmap, alphabet, highlight_positions=mutations, show=show, color_sheme=color_sheme,
                 title=title, dest=dest)

if __name__ == '__main__':

    #read metadata
    meta_data = pd.read_csv(os.path.join(DMS_path, "enzyme_metadata.csv"))

    # extract names and sequences
    seqs = meta_data['target_seq']
    names = meta_data['DMS_id']

    # load datasets
    datasets = [pd.read_csv(os.path.join(datasets_path, name + '.csv')) for name in names]

    # load esm alphabet
    alphabet = alphabet.to_dict()

    i = 0
    # compute and plot
    for name, seq in zip(names, seqs):
        # create a folder for every gene
        dest = os.path.join(results_path, name)
        if not os.path.exists(dest):
            os.mkdir(dest)

        # compute embeddings
        results, batch_lens, batch_labels, alphabet = esm_compute([seq])
        seq_rep = get_seq_rep(results, batch_lens)

        # LLM major computations
        logits, alphabet = get_mutant_logits(seq, batch_size=1)
        _, _, pdbs, _, _ = structure_prediction(seqs=[seq], names=[name])

        # calculations
        p = get_probability_distribution(logits)
        mmp = masked_marginal_probability(p, seq, alphabet)
        entropy = per_position_entropy(p)
        pdb = entropy_to_bfactor(pdbs[0], entropy)

        # save tensors
        torch.save(p, os.path.join(dest, f"prob_dist.pt"))
        torch.save(mmp, os.path.join(dest, f"masked_marginal_probability.pt"))
        torch.save(entropy, os.path.join(dest, f"per_position_entropy.pt"))
        torch.save(results, os.path.join(dest, f"results.pt"))
        torch.save(logits, os.path.join(dest, f"masked_logits.pt"))

        # save visualizations
        prb_dist_path = os.path.join(dest, f"prob_dist.png")
        log_odds_path = os.path.join(dest, f"log_odds.png")
        per_pos_entropy_path = os.path.join(dest, f"per_pos_entropy.png")
        pdb_path = os.path.join(dest, f"{name}.pdb")

        plot_heatmap(p=p, alphabet=alphabet, remove_tokens=True, dest=prb_dist_path, show=False, title=f"{name} probability distribution", color_sheme="b")
        plot_heatmap(p=mmp, alphabet=alphabet, dest=log_odds_path, show=True, title=f"{name} per position log-odds", color_sheme="rwb")
        plot_per_position_entropy(entropy, seq, show=False, dest=per_pos_entropy_path)
        pdb.write(pdb_path)

        # highlighted plots
        plot_entropy_with_highlighted_mutations(name=name, seq=seq, entropy=entropy, data=datasets[i], dest=dest)
        plot_heatmap_with_highlighted_mutations(name=name, data=datasets[i], alphabet=alphabet, heatmap=mmp, heatmap_type='mmp')
        i += 1

