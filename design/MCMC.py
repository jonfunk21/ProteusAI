import os
import random
import numpy as np
import sys
sys.path.append('')
from design import constraints


class ProteinDesign:
    """
    Optimizes a protein sequence based on a custom energy function. Set weights of constraints to 0 which you don't want to use.

    Parameters:
        native_seq (str): native sequence to be optimized
        sampler (str): choose between simulated_annealing and substitution design. Default simulated annealing
        n_traj (int): number of independent trajectories.
        n_iter (int): number of sampling intervals per trajectory. For simulated annealing, the number of iterations is often chosen in the range of [1,000, 10,000].
        mut_p (tuple): probabilities for substitution, insertion and deletion. Default [0.6, 0.2, 0.2]
        T (float): sampling temperature. For simulated annealing, T0 is often chosen in the range [1, 100]. default 10
        M (float): rate of temperature decay. or simulated annealing, a is often chosen in the range [0.01, 0.1] or [0.001, 0.01]. Default 0.01
        length_constraint (tuple): constraint on length. (maximum_allowed_length(int), weight_of_constraint(float)). Default (200, 0.2)
        identity (bool): Use sequence identity constraint.
        w_identity (float): Weight of sequence identity constraint. Positive values reward low sequence identity to native sequence.
        pred_struc (bool): if True predict the structure of the protein at every step and use structure based constraints in the energy function. Default True.
        w_ptm (float): weight for ptm. Default 0.4
        w_plddt (float): weight for plddt. Default 0.4
        outdir (str): path to output directory. Default None
    """

    def __init__(self, native_seq: str = None, sampler: str = 'simulated_annealing',
                 n_traj: int = 5, n_iter: int = 1000,
                 mut_p: tuple = (0.6, 0.2, 0.2),
                 T: float = 10., M: float = 0.01,
                 length_constraint: tuple = (200, 0.2),
                 pred_struc: bool = True,
                 w_ptm: float = 0.2, w_plddt: float = 0.2,
                 identity=True, w_identity = 0.2,
                 w_globularity: float = 0.002,
                 outdir = None
                 ):

        self.native_seq = native_seq
        self.sampler = sampler
        self.n_traj = n_traj
        self.n_iter = n_iter
        self.mut_p = mut_p
        self.T = T
        self.M = M
        self.pred_struc = pred_struc
        self.length_constraint = length_constraint
        self.max_len = int(length_constraint[0])
        self.w_max_len = float(length_constraint[1])
        self.identity = identity
        self.w_identity = w_identity
        self.w_ptm = w_ptm
        self.w_plddt = w_plddt
        self.w_globularity = w_globularity
        self.outdir = outdir


    def __str__(self):
        l = ['ProteusAI.MCMC.Hallucination class: \n',
             '---------------------------------------\n',
             'When Hallucination.run() sequences will be hallucinated using this seed sequence:\n\n',
             f'{self.native_seq}\n',
             '\nThe following variables were set:\n\n',
             'variable\t|value\n',
             '----------------+-------------------\n',
             f'algorithm: \t|{self.sampler}\n',
             f'n_traj: \t|{self.n_traj}\n',
             f'n_iter: \t|{self.n_iter}\n',
             f'mut_p: \t\t|{self.mut_p}\n',
             f'T: \t\t|{self.T}\n',
             f'M: \t\t|{self.M}\n\n',
             f'The energy function is a linear combination of the following constraints:\n\n',
             f'constraint\t|value\t|weight\n',
             '----------------+-------+------------\n',
             f'length \t\t|{self.max_len}\t|{self.w_max_len}\n',
             ]
        s = ''.join(l)
        if self.identity:
            l = [
                s,
                f'identity\t|\t|{self.w_identity}\n'
            ]
            s = ''.join(l)
        if self.pred_struc:
            l = [
                s,
                f'pTM\t\t|\t|{self.w_ptm}\n'
                f'pLDDT\t\t|\t|{self.w_plddt}\n'
            ]
            s = ''.join(l)
        return s

    ### SAMPLERS
    def mutate(self, seqs, mut_p: tuple = (0.6, 0.2, 0.2)):
        """
        mutates input sequences.

        Parameters:
            seqs (tuple): list of peptide sequences

        Returns:
            list: mutated sequences
        """
        AAs = ('A', 'C', 'D', 'E', 'F', 'G', 'H',
               'I', 'K', 'L', 'M', 'N', 'P', 'Q',
               'R', 'S', 'T', 'V', 'W', 'Y')

        mut_types = ('substitution', 'insertion', 'deletion')

        mutated_seqs = []
        for seq in seqs:
            mut_type = random.choices(mut_types, mut_p)[0]
            if mut_type == 'substitution':
                pos = random.randint(0, len(seq) - 1)
                replacement = random.choice(AAs)
                mut_seq = ''.join([seq[:pos], replacement, seq[pos + 1:]])
                mutated_seqs.append(mut_seq)

            elif mut_type == 'insertion':
                pos = random.randint(0, len(seq) - 1)
                insertion = random.choice(AAs)
                mut_seq = ''.join([seq[:pos], insertion, seq[pos:]])
                mutated_seqs.append(mut_seq)

            elif mut_type == 'deletion' and len(seq) > 1:
                pos = random.randint(0, len(seq) - 1)
                l = list(seq)
                del l[pos]
                mut_seq = ''.join(l)
                mutated_seqs.append(mut_seq)

            else:
                # will perform insertion if length is to small
                pos = random.randint(0, len(seq) - 1)
                insertion = random.choice(AAs)
                mut_seq = ''.join([seq[:pos], insertion, seq[pos:]])
                mutated_seqs.append(mut_seq)

        return mutated_seqs

    ### ENERGY FUNCTION and ACCEPTANCE CRITERION
    def energy_function(self, seqs: list, i):
        """
        Combines constraints into an energy function.

        Parameters:
            seqs (list): list of sequences

        Returns:
            list: Energy value
        """
        energies = np.zeros(len(seqs))

        # structure prediction



        # rescale pLDDT and make values negative. The higher the confidence is better


        energies += self.w_max_len * constraints.length_constraint(seqs=seqs, max_len=self.max_len)
        energies += self.w_identity * constraints.seq_identity(seqs=seqs, ref=self.native_seq)

        pdbs = []
        if self.pred_struc:
            names = [f'sequence_{j}_cycle_{i}' for j in range(len(seqs))]
            headers, sequences, pdbs, pTMs, pLDDTs = constraints.structure_prediction(seqs, names)
            pTMs = [1 - val for val in pTMs]
            pLDDTs = [1 - val / 100 for val in pLDDTs]
            energies += self.w_ptm * np.array(pTMs)
            energies += self.w_plddt * np.array(pLDDTs)
            energies += self.w_globularity * constraints.globularity(pdbs)

            # just a line to peak into some of the progress
            with open('peak', 'w') as f:
                for i in range(len(seqs)):
                    line = [
                        'header:', '\n',
                        headers[i], '\n',
                        'sequence:\n',
                        sequences[i], '\n',
                        'pTMs:', str(pTMs[i]), '\n',
                        'pLDDT:', str(pLDDTs[i]), '\n',
                        'energy:', str(energies[i]), '\n'
                    ]
                    f.writelines(line)

        return energies, pdbs

    def p_accept(self, E_x_mut, E_x_i, T, i, M):
        """
        Decides to accep or reject changes.
        """

        T = T / (1 + M * i)
        dE = E_x_i - E_x_mut
        exp_val = np.exp(1 / (T * dE))
        p_accept = np.minimum(exp_val, np.ones_like(exp_val))
        return p_accept

    ### RUN
    def run(self):
        """
        Runs MCMC-sampling based on user defined inputs. Returns optimized sequences.
        """
        native_seq = self.native_seq
        n_traj = self.n_traj
        n_iter = self.n_iter
        sampler = self.sampler
        energy_function = self.energy_function
        T = self.T
        M = self.M
        p_accept = self.p_accept
        mut_p = self.mut_p
        outdir = self.outdir

        if outdir != None:
            if not os.path.exists(outdir):
                os.mkdir(outdir)

        if sampler == 'simulated_annealing':
            mutate = self.mutate

        if native_seq is None:
            raise 'The optimizer needs a sequence to run. Define a sequence by calling SequenceOptimizer(native_seq = <your_sequence>)'

        seqs = [native_seq for _ in range(n_traj)]

        E_x_i, pdbs = energy_function(seqs, 0)
        for i in range(n_iter):
            mut_seqs = mutate(seqs, mut_p)
            E_x_mut, pdbs_mut = energy_function(mut_seqs, i)

            # accept or reject change
            p = p_accept(E_x_mut, E_x_i, T, i, M)
            num = '{:04d}'.format(i)
            for n in range(len(p)):
                if p[n] > random.random():
                    E_x_i[n] = E_x_mut[n]
                    seqs[n] = mut_seqs[n]
                    if self.pred_struc and outdir != None:
                        pdbs[n] = pdbs_mut[n]
                        with open(f'hallucinations/{num}_hallucination_{n}.pdb', 'w') as f:
                            f.writelines(pdbs[n])
                elif p[n] <= random.random() and (self.pred_struc and outdir != None):
                    with open(f'rejected/{num}_hallucination_{n}.pdb', 'w') as f:
                        f.writelines(pdbs_mut[n])

        return (seqs)