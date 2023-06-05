# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import biotite.sequence.graphics as graphics
import biotite.application.muscle as muscle
import matplotlib.pyplot as plt
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align.Applications import ClustalwCommandline
from Bio import AlignIO
from Bio import SeqIO
from collections import Counter
from biotite.sequence import ProteinSequence


def align_proteins(names: list, seqs: list, plot_results: bool = False, plt_range: tuple = (0, 200), muscle_version: str = '5',
        save_fig: str = None, save_fasta:str = None, figsize: tuple = (10.0, 8.0)):
    """
    performs multiple sequence alignement given a list of blast names and the corresponding sequences.

    Parameters:
        names (list): list of sequence names
        seqs (list): list of sequences for MSA
        plot_results (bool, optional): plot results.
        plt_range (tuple, optional): range of sequence which is plotted. Default first 200 amino acids.
        muscle_version (str, optional): which muscle version is installed on your machine. Default 5.
        save_fig (str): saves fig if path is provided. Default None - won't save the figure.
        save_fasta (str): saves fasta if path is provided. Default None - won't save fasta.
        figsize (tuple): dimensions of the figure.

    Returns:
        dict: MSA results of sequence names/ids and gapped sequences
    """

    # Convert sequences to ProteinSequence objects if they are strings
    seqs = [ProteinSequence(seq) if isinstance(seq, str) else seq for seq in seqs]

    if muscle_version == '5':
        app = muscle.Muscle5App(seqs)
    elif muscle_version == '3':
        app = muscle.MuscleApp(seqs)
    else:
        raise ValueError('Muscle version must be either 3 or 5')

    app.start()
    app.join()
    alignment = app.get_alignment()

    # Print the MSA with hit IDs
    gapped_seqs = alignment.get_gapped_sequences()

    MSA_results = {}
    for i in range(len(gapped_seqs)):
        MSA_results[names[i]] = gapped_seqs[i]

    # Reorder alignments to reflect sequence distance
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    order = app.get_alignment_order()
    graphics.plot_alignment_type_based(
        ax, alignment[plt_range[0]:plt_range[1], order.tolist()], labels=[names[i] for i in order],
        show_numbers=True, color_scheme="clustalx"
    )
    fig.tight_layout()

    if save_fig != None:
        plt.savefig(save_fig)

    if plot_results:
        plt.show()
    else:
        plt.close()

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


def MSA_results_to_fasta(MSA_results: dict, fname: str):
    """
    Takes MSA results from the align proteins function and writes then into a fasta format.

    Parameters:
        MSA_results (dict): Dictionary of MSA results.
        fname (str): file name.

    Returns:
        None
    """
    with open(fname, 'w') as f:
        for i, key in enumerate(MSA_results.keys()):
            s = MSA_results[key]
            if i < len(MSA_results) - 1:
                f.writelines(f'>{key}\n')
                f.writelines(f'{s}\n')
            else:
                f.writelines(f'>{key}\n')
                f.writelines(f'{s}')

def align_dna(dna_sequences: list, verbose: bool = False):
    """
    performs multiple sequence alignement given a list of blast names and the corresponding sequences.
    The function uses the clustalw2 app which uses the global needleman-wunsch algorithm.

    Parameters:
        seqs (list): list of DNA sequences for MSA
        verbose (bool): print std out and std error if True. Default False

    Returns:
        list: MSA list of sequences
    """
    # Create SeqRecord objects for each DNA sequence
    seq_records = [SeqRecord(Seq(seq), id=f"seq{i + 1}") for i, seq in enumerate(dna_sequences)]

    # Write the DNA sequences to a temporary file in FASTA format
    temp_input_file = "temp_input.fasta"
    with open(temp_input_file, "w") as handle:
        SeqIO.write(seq_records, handle, "fasta")

    # Run ClustalW to perform the multiple sequence alignment
    temp_output_file = "temp_output.aln"
    clustalw_cline = ClustalwCommandline("clustalw2", infile=temp_input_file, outfile=temp_output_file)
    stdout, stderr = clustalw_cline()

    if verbose:
        print('std out:')
        print('-------')
        print(stdout)
        print('std error:')
        print('----------')
        print(stderr)

    # Read in the aligned sequences from the output file
    aligned_seqs = []
    with open(temp_output_file) as handle:
        alignment = AlignIO.read(handle, "clustal")
        for record in alignment:
            aligned_seqs.append(str(record.seq))

    return aligned_seqs


def get_consensus_sequence(dna_sequences: list):
    """
    Calculates the consensus sequence of multiple sequence alignements.
    It uses the most common character of every sequence in a list. All
    sequences need to be the same length.

    Parameters:
        dna_sequences (list): list of DNA sequences.

    Returns:
        str: consensus sequence.
    """
    # Get the length of the sequences
    sequence_length = len(dna_sequences[0])

    # Iterate over each position in the sequences
    consensus_sequence = ""
    for i in range(sequence_length):
        # Create a list of the characters at this position
        char_list = [seq[i] for seq in dna_sequences]

        # Count the occurrences of each character
        char_counts = Counter(char_list)

        # Get the most common character
        most_common_char = char_counts.most_common(1)[0][0]

        # Add the most common character to the consensus sequence
        consensus_sequence += most_common_char

    return consensus_sequence