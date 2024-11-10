# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import os
import numpy as np
from proteusAI.design_tools import Constraints
import pandas as pd


class ZeroShot:
    """
    ZeroShot inference for single mutational effects. Mutates every single position to every possible amino acid at
    that position. Then calculate differnt metrics of structure and embeddings for the mutants. Observe which mutations
    have the greatest effect on the mutants.

    Parameters:
    -----------
        seq (str): native sequence to be optimized
        constraints (dict): constraints on sequence.
            Keys describe the kind of constraint and values the position on which they act.
        sampler (str): choose between simulated_annealing and substitution design_tools.
            Default 'simulated annealing'
        batch_size (int): number of independent trajectories per sampling step. Lowest energy mutant will be selected when
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

    def __init__(
        self,
        seq: str = None,
        name="Prot",
        constraints=None,
        batch_size: int = 20,
        pred_struc: bool = True,
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
            constraints = {"all_atm": []}
        self.seq = seq
        self.name = name
        self.constraints = constraints
        self.batch_size = batch_size
        self.w_ptm = w_ptm
        self.w_plddt = w_plddt
        self.w_globularity = w_globularity
        self.outdir = outdir
        self.verbose = verbose
        self.w_sasa = w_sasa
        self.w_bb_coord = w_bb_coord
        self.w_all_atm = w_all_atm

        # Parameters
        self.ref_pdbs = None

    def __str__(self):
        lines = [
            "ProteusAI.MCMC.ZeroShot class: \n",
            "---------------------------------------\n",
            "When Hallucination.run() sequences will be hallucinated using this seed sequence:\n\n",
            f"{self.seq}\n",
            "\nThe following variables were set:\n\n",
            "variable\t|value\n",
            "----------------+-------------------\n",
            f"batch_size: \t|{self.batch_size}\n",
            "The energy function is a linear combination of the following constraints:\n\n",
            "constraint\t|value\t|weight\n",
            "----------------+-------+------------\n",
        ]
        s = "".join(lines)
        lines = [
            s,
            f"pTM\t\t|\t|{self.w_ptm}\n",
            f"pLDDT\t\t|\t|{self.w_plddt}\n",
            f"bb_coord\t\t|{self.w_bb_coord}\n",
            f"all_atm\t\t|\t|{self.w_all_atm}\n",
            f"sasa\t\t|\t|{self.w_sasa}\n",
        ]
        s = "".join(lines)
        return s

    ### SAMPLERS
    def mutate(self, seq, pos):
        """
        mutates input sequences. to all possible amino acids at position

        Parameters:
            seqs (str): native protein sequence
            pos (int): position to mutate

        Returns:
            list: mutated sequences
        """

        AAs = (
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        )

        native_aa = seq[pos]

        mut_seqs = []
        names = []
        for res in AAs:
            if res != native_aa:
                mut_seq = "".join([seq[:pos], res, seq[pos + 1 :]])
                mut_seqs.append(mut_seq)
                names.append(str(seq[pos]) + str(pos) + str(res))
        return mut_seqs, names

    ### ENERGY FUNCTION and ACCEPTANCE CRITERION
    def energy_function(self, seqs: list, pos: int, names):
        """
        Combines constraints into an energy function. The energy function
        returns the energy values of the mutated files and the associated pdb
        files as temporary files. In addition it returns a dictionary of the different
        energies.

        Parameters:
            seqs (list): list of sequences
            pos (int): position in sequence
            constraints (list): list of constraints

        Returns:
            tuple: Energy value, pdbs, energy_log
        """
        # reinitialize energy
        energies = np.zeros(len(seqs))
        energy_log = dict()
        constraints = self.constraints

        for i, c in enumerate(constraints["all_atm"]):
            if c == pos:
                del constraints["all_atm"][i]
        constraints = [constraints for i in range(len(seqs))]

        # structure prediction
        names = [f"{name}_{self.name}" for name in names]
        headers, sequences, pdbs, pTMs, mean_pLDDTs = Constraints.structure_prediction(
            seqs, names
        )
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

        energy_log[f"e_pTMs x {self.w_ptm}"] = e_pTMs
        energy_log[f"e_mean_pLDDTs x {self.w_plddt}"] = e_mean_pLDDTs
        energy_log[f"e_globularity x {self.w_globularity}"] = e_globularity
        energy_log[f"e_sasa x {self.w_sasa}"] = e_sasa

        # there are now ref pdbs before the first calculation
        if self.ref_pdbs is not None:
            e_bb_coord = self.w_bb_coord * Constraints.backbone_coordination(
                pdbs, self.ref_pdbs
            )
            e_all_atm = self.w_all_atm * Constraints.all_atom_coordination(
                pdbs, self.ref_pdbs, constraints, constraints
            )

            energies += e_bb_coord
            energies += e_all_atm

            energy_log[f"e_bb_coord x {self.w_bb_coord}"] = e_bb_coord
            energy_log[f"e_all_atm x {self.w_all_atm}"] = e_all_atm
        else:
            energy_log[f"e_bb_coord x {self.w_bb_coord}"] = [0]
            energy_log[f"e_all_atm x {self.w_all_atm}"] = [0]

        energy_log["position"] = [pos]

        return energies, pdbs, energy_log

    ### RUN
    def run(self):
        """
        Runs MCMC-sampling based on user defined inputs. Returns optimized sequences.
        """
        seq = self.seq
        batch_size = self.batch_size
        energy_function = self.energy_function
        outdir = self.outdir
        mutate = self.mutate
        pdb_out = os.path.join(outdir, "pdbs")
        png_out = os.path.join(outdir, "pngs")
        data_out = os.path.join(outdir, "data_tools")

        if outdir is not None:
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            if not os.path.exists(pdb_out):
                os.mkdir(pdb_out)
            if not os.path.exists(png_out):
                os.mkdir(png_out)
            if not os.path.exists(data_out):
                os.mkdir(data_out)

        if seq is None:
            raise "Provide a sequence(seq = <your_sequence>)"

        # for initial calculation don't use the full sequences, unecessary
        # calculation of initial state
        _, pdbs, energy_log = energy_function([seq], 0, ["native"])
        pdbs = [pdbs[0] for _ in range(batch_size - 1)]
        energy_log["mut"] = ["-"]
        energy_log["description"] = ["native"]

        # make energies to list
        for key in energy_log.keys():
            if not isinstance(energy_log[key], list):
                energy_log[key] = energy_log[key].tolist()

        self.ref_pdbs = pdbs.copy()

        if outdir is not None:
            # saves the n th structure
            pdbs[0].write(os.path.join(pdb_out, f"native_{self.name}.pdb"))

        # write energy_log in data_out
        if outdir is not None:
            df = pd.DataFrame(energy_log)
            df.to_csv(os.path.join(data_out, "energy_log.pdb"), index=False)

        for pos in range(len(seq)):
            seqs, names = mutate(seq, pos)
            E_x_i, pdbs, _energy_log = energy_function(seqs, pos, names)

            for n in range(len(seqs)):
                for key in energy_log.keys():
                    # skip skalar values in this step
                    if key not in ["position", "mut", "description"]:
                        e = _energy_log[key].tolist()
                        with open("test", "w") as f:
                            print("energy_log", file=f)
                            print(energy_log, file=f)
                            print("_energy_log", file=f)
                            print(_energy_log, file=f)
                            print("key", file=f)
                            print(key, file=f)
                            print("e", file=f)
                            print(e, file=f)
                        energy_log[key].append(e[n])

                energy_log["position"].append(pos)
                energy_log["mut"].append(names[n])

                energy_log["description"].append(f"{names[n]}_{self.name}")

                if outdir is not None:
                    # saves the n th structure
                    pdbs[n].write(os.path.join(pdb_out, f"{names[n]}_{self.name}.pdb"))

                # write energy_log in data_out
                if outdir is not None:
                    df = pd.DataFrame(energy_log)
                    df.to_csv(os.path.join(data_out, "energy_log.pdb"), index=False)
