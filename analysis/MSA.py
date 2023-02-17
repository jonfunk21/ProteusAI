import biotite.sequence.graphics as graphics
import biotite.application.muscle as muscle
import matplotlib.pyplot as plt


# Perform a multiple sequence alignment using MUSCLE
def MSA(hits: list, hit_seqs: list, plot_results: bool = False, plt_range: tuple = (0, 200), muscle_version: str = '5',
        save_fig: str = None, save_fasta:str = None, figsize: tuple = (10.0, 8.0)):
    """
    performs multiple sequence alignement given a list of blast hits and the corresponding sequences.

    Parameters:
        hits (list): list of sequence names/ids
        hits_seqs (list): list of sequences corresponding to names/ids for MSA
        plot_results (bool, optional): plot results.
        plt_range (tuple, optional): range of sequence which is plotted. Default first 200 amino acids.
        muscle_version (str, optional): which muscle version is installed on your machine. Default 5.
        save_fig (str): saves fig if path is provided. Default None - won't save the figure.
        save_fasta (str): saves fasta if path is provided. Default None - won't save fasta.
        figsize (tuple): dimensions of the figure.

    Returns:
        dict: MSA results of sequence names/ids and gapped sequences
    """

    if muscle_version == '5':
        app = muscle.Muscle5App(hit_seqs)
    elif muscle_version == '3':
        app = muscle.MuscleApp(hit_seqs)
    else:
        raise 'Muscle version must be either 3 or 5'

    app.start()
    app.join()
    alignment = app.get_alignment()

    # Print the MSA with hit IDs
    gapped_seqs = alignment.get_gapped_sequences()

    MSA_results = {}
    for i in range(len(gapped_seqs)):
        MSA_results[hits[i]] = gapped_seqs[i]

    # Reorder alignments to reflect sequence distance
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    order = app.get_alignment_order()
    graphics.plot_alignment_type_based(
        ax, alignment[plt_range[0]:plt_range[1], order.tolist()], labels=[hits[i] for i in order],
        show_numbers=True, color_scheme="clustalx"
    )
    fig.tight_layout()

    if save_fig != None: plt.savefig(save_fig)

    if plot_results: plt.show()

    if save_fasta != None:
        with open(save_fasta, 'w') as f:
            for i, key in enumerate(MSA_results.keys()):
                s = MSA_results[key]
                if i < len(MSA_results) - 1:
                    f.writelines(f'>{key}\n')
                    f.writelines(f'{s}\n')
                else:
                    f.writelines(f'>{key}\n')
                    f.writelines(f'{s}')

    return MSA_results