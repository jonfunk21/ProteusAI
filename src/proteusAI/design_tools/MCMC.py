# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import os
import random
import numpy as np
from proteusAI.design_tools import Constraints
import pandas as pd


class ProteinDesign:
    """
    Optimizes a protein sequence based on a custom energy function. Set weights of constraints to 0 which you don't want to use.

    Parameters:
    -----------
        native_seq (str): native sequence to be optimized
        constraints (dict): constraints on sequence.
            Keys describe the kind of constraint and values the position on which they act.
        sampler (str): choose between simulated_annealing and substitution design_tools.
            Default 'simulated annealing'
        n_traj (int): number of independent trajectories per sampling step. Lowest energy mutant will be selected when
            multiple are viable. Default 16
        steps (int): number of sampling steps per trajectory.
            For simulated annealing, the number of iterations is often chosen in the range of [1,000, 10,000].
        T (float): sampling temperature.
            For simulated annealing, T0 is often chosen in the range [1, 100]. default 1
        M (float): rate of temperature decay.
            or simulated annealing, a is often chosen in the range [0.01, 0.1] or [0.001, 0.01]. Default 0.01
        mut_p (tuple): probabilities for substitution, insertion and deletion.
            Default [0.6, 0.2, 0.2]
        pred_struc (bool): if True predict the structure of the protein at every step and use structure
            based constraints in the energy function. Default True.
        max_len (int): maximum length sequence length for lenght constraint.
            Default 300.
        w_len (float): weight of length constraint.
            Default 0.01
        w_identity (float): Weight of sequence identity constraint. Positive values reward low sequence identity to native sequence.
            Default 0.04
        w_ptm (float): weight for ptm. pTM is calculated as 1-pTM, because lower energies should be better.
            Default 1.
        w_plddt (float): weight for plddt.  The mean pLDDT is calculated as 1-mean_pLDDT, because lower energies should be better.
            Default 1.
        w_globularity (float): weight of globularity constraint
            Default 0.001
        w_bb_coord (float): weight on backbone coordination constraint. Constraints backbone to native structure.
            Default 0.02
        w_all_atm (float): weight on all atom coordination constraint. Acts on all atoms which are constrained
            Default 0.15
        w_sasa (float): weight of surface exposed hydrophobics constraint
            Default 0.02
        outdir (str): path to output directory.
            Default None
        verbose (bool): if verbose print information
    """

    def __init__(self,
                 native_seq: str = None,
                 constraints=None,
                 sampler: str = 'simulated_annealing',
                 n_traj: int = 16,
                 steps: int = 1000,
                 T: float = 10.,
                 M: float = 0.01,
                 mut_p: list = (0.6, 0.2, 0.2),
                 pred_struc: bool = True,
                 max_len: int = 300,
                 w_len: float=0.01,
                 w_identity: float = 0.1,
                 w_ptm: float = 1,
                 w_plddt: float = 1,
                 w_globularity: float = 0.001,
                 w_bb_coord: float = 0.02,
                 w_all_atm: float = 0.15,
                 w_sasa: float = 0.02,
                 outdir: str = None,
                 verbose: bool = False,
                 ):

        if constraints is None:
            constraints = {'no_mut': [], 'all_atm': []}
        self.native_seq = native_seq
        self.sampler = sampler
        self.n_traj = n_traj
        self.steps = steps
        self.mut_p = mut_p
        self.T = T
        self.M = M
        self.pred_struc = pred_struc
        self.max_len = max_len
        self.w_max_len = w_len
        self.w_identity = w_identity
        self.w_ptm = w_ptm
        self.w_plddt = w_plddt
        self.w_globularity = w_globularity
        self.outdir = outdir
        self.verbose = verbose
        self.constraints = constraints
        self.w_sasa = w_sasa
        self.w_bb_coord = w_bb_coord
        self.w_all_atm = w_all_atm

        # Parameters
        self.ref_pdbs = None
        self.ref_constraints = None
        self.initial_energy = None


    def __str__(self):
        l = ['ProteusAI.MCMC.Design class: \n',
             '---------------------------------------\n',
             'When Hallucination.run() sequences will be hallucinated using this seed sequence:\n\n',
             f'{self.native_seq}\n',
             '\nThe following variables were set:\n\n',
             'variable\t|value\n',
             '----------------+-------------------\n',
             f'algorithm: \t|{self.sampler}\n',
             f'steps: \t\t|{self.steps}\n',
             f'n_traj: \t|{self.n_traj}\n',
             f'mut_p: \t\t|{self.mut_p}\n',
             f'T: \t\t|{self.T}\n',
             f'M: \t\t|{self.M}\n\n',
             f'The energy function is a linear combination of the following constraints:\n\n',
             f'constraint\t|value\t|weight\n',
             '----------------+-------+------------\n',
             f'length \t\t|{self.max_len}\t|{self.w_max_len}\n',
             f'identity\t|\t|{self.w_identity}\n',
             ]
        s = ''.join(l)
        if self.pred_struc:
            l = [
                s,
                f'pTM\t\t|\t|{self.w_ptm}\n',
                f'pLDDT\t\t|\t|{self.w_plddt}\n',
                f'bb_coord\t\t|{self.w_bb_coord}\n',
                f'all_atm\t\t|\t|{self.w_all_atm}\n',
                f'sasa\t\t|\t|{self.w_sasa}\n',
            ]
            s = ''.join(l)
        return s

    ### SAMPLERS
    def mutate(self, seqs, mut_p: list = None, constraints: list = None):
        """
        mutates input sequences.

        Parameters:
            seqs (list): list of peptide sequences
            mut_p (list): mutation probabilities
            constraints (list): dictionary of constraints

        Returns:
            list: mutated sequences
        """

        if mut_p is None:
            mut_p = [0.6, 0.2, 0.2]

        AAs = ('A', 'C', 'D', 'E', 'F', 'G', 'H',
               'I', 'K', 'L', 'M', 'N', 'P', 'Q',
               'R', 'S', 'T', 'V', 'W', 'Y')

        mut_types = ('substitution', 'insertion', 'deletion')

        mutated_seqs = []
        mutated_constraints = []
        mutations = []
        for i, seq in enumerate(seqs):
            mut_constraints = {}

            # loop until allowed mutation has been selected
            mutate = True
            while mutate:
                pos = random.randint(0, len(seq) - 1)
                mut_type = random.choices(mut_types, mut_p)[0]
                if pos in constraints[i]['no_mut'] or pos in constraints[i]['all_atm']:
                    pass
                # secondary structure constraint disallows deletion
                # insertions between two secondary structure constraints will have the constraint of their neighbors
                else:
                    break

            if mut_type == 'substitution':
                replacement = random.choice(AAs)
                mut_seq = ''.join([seq[:pos], replacement, seq[pos + 1:]])
                for const in constraints[i].keys():
                    positions = constraints[i][const]
                    mut_constraints[const] = positions
                mutations.append(f'sub:{seq[pos]}{pos}{replacement}')

            elif mut_type == 'insertion':
                insertion = random.choice(AAs)
                mut_seq = ''.join([seq[:pos], insertion, seq[pos:]])
                # shift constraints after insertion
                for const in constraints[i].keys():
                    positions = constraints[i][const]
                    positions = [i if i < pos else i + 1 for i in positions]
                    mut_constraints[const] = positions
                mutations.append(f'ins:{pos}{insertion}')

            elif mut_type == 'deletion' and len(seq) > 1:
                l = list(seq)
                del l[pos]
                mut_seq = ''.join(l)
                # shift constraints after deletion
                for const in constraints[i].keys():
                    positions = constraints[i][const]
                    positions = [i if i < pos else i - 1 for i in positions]
                    mut_constraints[const] = positions
                mutations.append(f'del:{seq[pos]}{pos}')

            else:
                # will perform insertion if length is to small
                insertion = random.choice(AAs)
                mut_seq = ''.join([seq[:pos], insertion, seq[pos:]])
                # shift constraints after insertion
                for const in constraints[i].keys():
                    positions = constraints[i][const]
                    positions = [i if i < pos else i + 1 for i in positions]
                    mut_constraints[const] = positions
                mutations.append(f'ins:{pos}{insertion}')

            mutated_seqs.append(mut_seq)
            mutated_constraints.append(mut_constraints)


        return mutated_seqs, mutated_constraints, mutations

    ### ENERGY FUNCTION and ACCEPTANCE CRITERION
    def energy_function(self, seqs: list, i: int, constraints: list):
        """
        Combines constraints into an energy function. The energy function
        returns the energy values of the mutated files and the associated pdb
        files as temporary files. In addition it returns a dictionary of the different
        energies.

        Parameters:
            seqs (list): list of sequences
            i (int): current iteration in sampling
            constraints (list): list of constraints

        Returns:
            tuple: Energy value, pdbs, energy_log
        """
        # reinitialize energy
        energies = np.zeros(len(seqs))
        energy_log = dict()

        e_len = self.w_max_len * Constraints.length_constraint(seqs=seqs, max_len=self.max_len)
        e_identity = self.w_identity * Constraints.seq_identity(seqs=seqs, ref=self.native_seq)

        energies += e_len
        energies += e_identity

        energy_log[f'e_len x {self.w_max_len}'] = e_len
        energy_log[f'e_identity x {self.w_identity}'] = e_identity

        pdbs = []
        if self.pred_struc:
            # structure prediction
            names = [f'sequence_{j}_cycle_{i}' for j in range(len(seqs))]
            headers, sequences, pdbs, pTMs, mean_pLDDTs = Constraints.structure_prediction(seqs, names)
            pTMs = [1 - val for val in pTMs]
            mean_pLDDTs = [1 - val / 100 for val in mean_pLDDTs]

            e_pTMs = self.w_ptm * np.array(pTMs)
            e_mean_pLDDTs = self.w_plddt * np.array(mean_pLDDTs)
            e_globularity = self.w_globularity * Constraints.globularity(pdbs)
            e_sasa = self.w_sasa * Constraints.surface_exposed_hydrophobics(pdbs)

            energies += e_pTMs
            energies += e_mean_pLDDTs
            energies += e_globularity
            energies += e_sasa

            energy_log[f'e_pTMs x {self.w_ptm}'] = e_pTMs
            energy_log[f'e_mean_pLDDTs x {self.w_plddt}'] = e_mean_pLDDTs
            energy_log[f'e_globularity x {self.w_globularity}'] = e_globularity
            energy_log[f'e_sasa x {self.w_sasa}'] = e_sasa

            # there are now ref pdbs before the first calculation
            if self.ref_pdbs is not None:
                e_bb_coord = self.w_bb_coord * Constraints.backbone_coordination(pdbs, self.ref_pdbs)
                e_all_atm = self.w_all_atm * Constraints.all_atom_coordination(pdbs, self.ref_pdbs, constraints, self.ref_constraints)

                energies += e_bb_coord
                energies += e_all_atm

                energy_log[f'e_bb_coord x {self.w_bb_coord}'] = e_bb_coord
                energy_log[f'e_all_atm x {self.w_all_atm}'] = e_all_atm
            else:
                energy_log[f'e_bb_coord x {self.w_bb_coord}'] = []
                energy_log[f'e_all_atm x {self.w_all_atm}'] = []

        energy_log['iteration'] = i + 1

        return energies, pdbs, energy_log

    def p_accept(self, E_x_mut, E_x_i, T, i, M):
        """
        Decides to accep or reject changes. Changes which have a lower energy
        than the previous state will always be accepted. Changes which have
        higher energies will be accepted with a probability p_accept. The
        acceptance probability for bad states decreases over time.

        Parameters:
        -----------
            E_x_mut (np.array): energies of mutated sequences
            E_x_i (np.array): energies of initial sequences
            T (float): Temperature
            i (int): current itteration
            M (float): decay constant

        Returns:
        --------
            np.array: accept probabilities
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
        constraints = self.constraints
        n_traj = self.n_traj
        steps = self.steps
        sampler = self.sampler
        energy_function = self.energy_function
        T = self.T
        M = self.M
        p_accept = self.p_accept
        mut_p = self.mut_p
        outdir = self.outdir
        pdb_out = os.path.join(outdir, 'pdbs')
        png_out = os.path.join(outdir, 'pngs')
        data_out = os.path.join(outdir, 'data_tools')

        if outdir != None:
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            if not os.path.exists(pdb_out):
                os.mkdir(pdb_out)
            if not os.path.exists(png_out):
                os.mkdir(png_out)
            if not os.path.exists(data_out):
                os.mkdir(data_out)

        if sampler == 'simulated_annealing':
            mutate = self.mutate

        if native_seq is None:
            raise 'The optimizer needs a sequence to run. Define a sequence by calling SequenceOptimizer(native_seq = <your_sequence>)'

        seqs = [native_seq for _ in range(n_traj)]
        constraints = [constraints for _ in range(n_traj)]
        self.ref_constraints = constraints.copy() # THESE ARE CORRECT

        # for initial calculation don't use the full sequences, unecessary
        # calculation of initial state
        E_x_i, pdbs, energy_log = energy_function([seqs[0]], -1, [constraints[0]])
        E_x_i = [E_x_i[0] for _ in range(n_traj)]
        pdbs = [pdbs[0] for _ in range(n_traj)]

        # empty energies dictionary for the first run
        for key in energy_log.keys():
            energy_log[key] = []
        energy_log['T'] = []
        energy_log['M'] = []
        energy_log['mut'] = []
        energy_log['description'] = []

        self.initial_energy = E_x_i.copy()
        self.ref_pdbs = pdbs.copy()

        if self.pred_struc and outdir is not None:
            # saves the n th structure
            num = '{:0{}d}'.format(len(energy_log['iteration']), len(str(self.steps)))
            pdbs[0].write(os.path.join(pdb_out, f'{num}_design.pdb'))

        # write energy_log in data_out
        if outdir is not None:
            df = pd.DataFrame(energy_log)
            df.to_csv(os.path.join(data_out, f'energy_log.pdb'), index=False)

        for i in range(steps):
            mut_seqs, _constraints, mutations = mutate(seqs, mut_p, constraints)
            E_x_mut, pdbs_mut, _energy_log = energy_function(mut_seqs, i, _constraints)
            # accept or reject change
            p = p_accept(E_x_mut, E_x_i, T, i, M)

            new_struc_found = False
            accepted_ind = [] # indices of accepted structures
            for n in range(n_traj):
                if p[n] > random.random():
                    accepted_ind.append(n)
                    E_x_i[n] = E_x_mut[n]
                    seqs[n] = mut_seqs[n]
                    constraints[n] = _constraints[n]
                    new_struc_found = True

            if new_struc_found:
                # get index of lowest energie sructure out of the newly found structures
                min_E = accepted_ind[0]
                for a in accepted_ind:
                    if a < min_E:
                        min_E = a

                # update all to lowest energy structure
                E_x_i = [E_x_i[min_E] for _ in range(n_traj)]
                seqs = [seqs[min_E] for _ in range(n_traj)]
                constraints = [constraints[min_E] for _ in range(n_traj)]
                pdbs = [pdbs_mut[min_E] for _ in range(n_traj)]

                for key in energy_log.keys():
                    # skip skalar values in this step
                    if key not in ['T', 'M', 'iteration', 'mut', 'description']:
                        e = _energy_log[key]
                        energy_log[key].append(e[min_E].item())

                energy_log['iteration'].append(i)
                energy_log['T'].append(T)
                energy_log['M'].append(M)
                energy_log['mut'].append(mutations[min_E])

                num = '{:0{}d}'.format(len(energy_log['iteration']), len(str(self.steps)))
                energy_log['description'].append(f'{num}_design')

                if self.pred_struc and outdir is not None:
                    # saves the n th structure
                    pdbs[0].write(os.path.join(pdb_out, f'{num}_design.pdb'))

                # write energy_log in data_out
                if outdir is not None:
                    df = pd.DataFrame(energy_log)
                    df.to_csv(os.path.join(data_out, f'energy_log.pdb'), index=False)

        return (seqs)