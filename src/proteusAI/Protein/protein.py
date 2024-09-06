# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from typing import Union
import inspect
import os
import warnings
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(current_path, '..')
sys.path.append(root_path)
from proteusAI.ml_tools.esm_tools import *
from proteusAI.struc import *
import hashlib
folder_path = os.path.dirname(os.path.realpath(__file__))
USR_PATH = os.path.join(folder_path, '../../../usrs')

model_dict = {"rf":"Random Forrest", "knn":"KNN", "svm":"SVM", "vae":"VAE", "esm2":"ESM-2", "esm1v":"ESM-1v"}

class Protein:
    """
    The protein object contains information about a single protein,
    such as its sequence, name, and its mathematical representations.
    
    Attributes:
        name (str): Name/id of the protein.
        seq (str): Protein sequence.
        rep (list): List of available representations.
        rep_path (str): Path to representations directory.
        design (str): Output of design.
    """

    def __init__(self, name: Union[str, None] = None, seq: Union[str, None] = None, struc: Union[str, struc.AtomArray, None] = None, reps: Union[list, tuple] = [], 
    user: Union[str, None] = 'guest', y = None, y_pred = None, y_sigma = None, acq_score = None, source: Union[str,None] = None, fname: Union[str,None] = None):
        """
        Initialize a new protein object.

        Args:
            name (str): Name/id of the protein.
            seq (str): Protein sequence.
            struc (AtomArray): Protein structure.
            reps (list): List of available representations.
            user (str): Path to the user. Will create one if the path does not exist. Default guest.
            y (float, int, str): Label for the protein.
            y_pred (float, int, str): Predicted y_value.
            y_sigma (float, int, str): Predicted y_value.
            acq_score (float): acquisition score.
            source (str, or data): Source of data, either a file or a data package created from a diversification step.
            fname (str): Only relevant for the app - provides the real file name instead of temporary file name from shiny.
        
        Parameters:
            
        """

        # Arguments
        self.name = name
        self.seq = seq
        self.reps = list(reps)
        self.struc = struc
        self.source = source
        self.y = y
        self.y_pred = y_pred
        self.y_sigma = y_sigma
        self.acq_score = acq_score
        self.fname = fname
        self.user = os.path.join(USR_PATH, user)

        # Parameters
        self.pdb_file = None
        self.fasta_file = None
        self.reps = []
        self.source_path = None
        self.rep_path = None
        self.struc_path = None
        self.design = None
        self.chains = []
        self.zs_path = None
        self.class_dict = None

        # Create user if user does not exist
        if not os.path.exists(self.user):
            self.initialize_user()

        # Initialize library from file or from inheritance
        if type(self.source) == str:
            self.init_from_file()
        elif self.source is not None:
            self.init_from_inheritance()

    def __str__(self):
        if self.struc is not None:
            struc_loaded = "loaded"
        return f"proteusAI.Protein():\n____________________\nname\t: {self.name}\nseq\t: {self.seq}\nrep\t: {self.reps}\ny:\t{self.y}\ny_pred:\t{self.y_pred}\ny_sig:\t{self.y_sigma}\nstruc:\t{self.pdb_file}\n"
    
    __repr__ = __str__

    def initialize_user(self):
        """
        initializing a new library.
        """
        print(f"Initializing user '{self.user}'...")

        # handle app case
        if self.fname:
            fname = self.fname.split('.')[0]
            file_extension = self.fname.split('.')[-1]
        else:
            f = self.source.split('/')[-1]
            fname = f.split('.')[0]
            file_extension = f.split('.')[-1]

        # set paths
        self.source_path = os.path.join(USR_PATH, self.user, fname)
        self.rep_path = os.path.join(self.source_path, 'zero_shot/rep')
        self.struc_path = os.path.join(self.source_path, 'zero_shot/struc')
        
        # create user library if user does not exist
        if not os.path.exists(self.user):
            os.makedirs(self.user)
            if self.file and not os.path.exists(os.path.join(self.user, self.file.split('.')[0])):
                fname = self.file.split('.')[0]
                os.makedirs(os.path.join(self.user, f'{fname}'))
                os.makedirs(os.path.join(self.user, f'{fname}/library'))
                os.makedirs(os.path.join(self.user, f'{fname}/zero_shot'))
                os.makedirs(os.path.join(self.user, f'{fname}/design'))
            print(f"User created at {self.user}")

    def init_from_inheritance(self):
        pass

    # TODO: add protein specific loading functions of previously computed data (currently only handled in app)

    def init_from_file(self):

        # parse information
        if self.fname:
            fname = self.fname
        else:
            fname = self.source.split('/')[-1]

        # Check for representations
        if self.rep_path == None:
            fname = fname.split('.')[0]
            zs_path = os.path.join(self.user, f'{fname}/zero_shot')
            rep_path = os.path.join(zs_path, 'rep')
            struc_path = os.path.join(zs_path, 'struc')
            self.zs_path = zs_path
            self.rep_path = rep_path
            self.struc_path = struc_path
            self.name = fname
        else:
            rep_path = '/'.join(self.rep_path.split('/')[:-1])


        # If file is not None, then initialize load fasta
        if self.source.endswith('.fasta'):
            self.load_fasta(file=self.source)
        
        # If file is not None, then initialize load fasta
        if self.source.endswith('.pdb'):
            self.load_structure(self.source, self.name)

        
    
    def load_fasta(self, file: str):
        """
        Load protein sequence from a FASTA file.

        Args:
            file (str): Path to the FASTA file.
        """
        sequences = []
        headers = []
        header = None
        seq = []

        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if header:
                        sequences.append(''.join(seq))
                        seq = []
                    header = line[1:]
                    headers.append(header)
                else:
                    seq.append(line)

            if seq:
                sequences.append(''.join(seq))

        if not headers:
            raise ValueError(f"The file {file} is not in FASTA format or is empty.")
        
        if len(headers) > 1:
            warnings.warn("The provided FASTA file contains multiple entries. Using only the first entry.")

        # Set the header as self.name
        name = headers[0].strip()  # remove potential leading spaces

        # Store the first sequence in self.seq
        seq = sequences[0]

        # Create and return a new Protein instance
        if self.fname:
            fname = self.fname.split('.')[0]
            self.name = fname
        else:
            self.name = name
            
        self.seq = seq

    def load_structure(self, prot_f, name = None, filter_solvent=True):
        """
        Load a structure from a pdb or cif file or an AtomArray.

        Args:
            struc: path to structure file or AtomArray
            name: provide protein name, else name of the file is assumed.
            filter_solvent (bool): True
        """
        if name is None and type(prot_f) == str:
            name = prot_f.split('/')[-1].split('.')[0]

        prot = load_struc(prot_f)

        if filter_solvent:
            non_solvent_mask = ~struc.filter_solvent(prot)
            prot = prot[non_solvent_mask]

        seqs = get_sequences(prot_f)
        chains = chain_parser(prot_f)

        self.pdb_file = prot_f
        self.seq = seqs
        self.name = name
        self.struc = prot
        self.chains = chains

    def view_struc(self, color=None, highlight=None, sticks=None):
        """
        3D visualization of protein structure.

        Args:
            color (str): Choose different coloration options
        """
        view = show_pdb(self.pdb_file, color=color, highlight=highlight, sticks=sticks)
        return view
    
    
    ### Zero-shot prediction ###
    def zs_prediction(self, model='esm2', batch_size=100, pbar=None, device=None, chain=None):
        """
        Compute zero-shot scores

        Args:
            model (str): Model used to compute ZS scores
            batch_size (int): Batch size used to compute ZS-Scores
            pbar: App progress bar
            device (str): Choose hardware for computation. Default 'None' for autoselection
                        other options are 'cpu' and 'cuda'. 
        """

        # Set a default chain if none is provided and there are chains available
        if chain is None and len(self.chains) >= 1:
            chain = self.chains[0]
        
        # Now ensure chain has a value before proceeding
        if chain is not None and len(self.chains) >= 1:
            seq = self.seq[chain]
            dest = os.path.join(self.zs_path, "results", chain, model)
        else:
            seq = self.seq
            dest = os.path.join(self.zs_path, "results", model)

        # Check if results already exist
        if os.path.exists(dest):
            print(f"Results already computed. Loading from {dest}")
            p = torch.load(os.path.join(dest, "prob_dist.pt"))
            mmp = torch.load(os.path.join(dest, "masked_marginal_probability.pt"))
            entropy = torch.load(os.path.join(dest, "per_position_entropy.pt"))
            logits = torch.load(os.path.join(dest, "masked_logits.pt"))
            df = pd.read_csv(os.path.join(dest, "zs_scores.csv"))
        else:
            # Perform computation if results do not exist
            print("Computing logits")
            logits, alphabet = get_mutant_logits(seq, batch_size=batch_size, model=model, pbar=pbar, device=device)

            # Calculations
            p = get_probability_distribution(logits)
            mmp = masked_marginal_probability(p, seq, alphabet)
            entropy = per_position_entropy(p)

            # Create directory if it doesn't exist
            if not os.path.exists(dest):
                os.makedirs(dest)
                print(f"Directory created at {dest}")

            # Save tensors
            torch.save(p, os.path.join(dest, "prob_dist.pt"))
            torch.save(mmp, os.path.join(dest, "masked_marginal_probability.pt"))
            torch.save(entropy, os.path.join(dest, "per_position_entropy.pt"))
            torch.save(logits, os.path.join(dest, "masked_logits.pt"))

            self.p = p
            self.mmp = mmp
            self.entropy = entropy
            self.logits = logits

            df = zs_to_csv(seq, alphabet, p, mmp, entropy, os.path.join(dest, "zs_scores.csv"))

            # no true y_values
            ys = [None] * len(mmp)

        out = {
            'df':df, 'rep_path':self.rep_path, 'struc_path':self.struc_path, 'y_type':'num', 'y_pred_col':'mmp', 'seqs_col':'sequence', 
            'names_col':'mutant', 'reps':self.reps, 'class_dict':self.class_dict, 'pred_data': True
        }

        return out


    def zs_library(self, model="esm2", chain=None):
        """
        Generate zero-shot library.
        """

        if chain == None and len(self.chains) >= 1:
            chain = self.chains[0]
            wt_seq = self.seq[chain]
            zs_results_path = os.path.join(self.zs_path, "results", chain, model, "zs_scores.csv")
        elif len(self.chains) >= 1:
            wt_seq = self.seq[chain]
            zs_results_path = os.path.join(self.zs_path, "results", chain, model, "zs_scores.csv")
        else:
            wt_seq = self.seq
            zs_results_path = os.path.join(self.zs_path, "results", model, "zs_scores.csv")

        # load already computed zs scores
        if os.path.exists(zs_results_path):
            df = pd.read_csv(zs_results_path)

        # generate df with blank y-values
        else:
            #wt_seq = self.seq
            canonical_aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
            mutants, sequences = [], []
            for pos in range(len(wt_seq)):
                for aa in canonical_aas:
                    if wt_seq[pos] != aa:
                        mutants.append(wt_seq[pos] + str(pos+1) + aa)
                        sequences.append(wt_seq[:pos] + aa + wt_seq[pos+1:])
            ys = [None] * len(mutants)

            df = pd.DataFrame({"mutant":mutants, "sequence":sequences ,"p":ys, "mmp":ys, "entropy":ys})
        
        out = {
            'df':df, 'rep_path':self.rep_path, 'struc_path':self.struc_path, 'y_type':'num', 'y_col':'mmp', 
            'seqs_col':'sequence', 'names_col':'mutant', 'reps':self.reps, 'class_dict':self.class_dict
        }

        return out
    

    ### Structure prediction ###
    def esm_fold(self, batch_size=100, chain=None, dest=None, pbar=None): # If structure prediction will become available for libraries, set dest in library, create a protein dir under the file name
        """
        Compute zero-shot scores

        Args:
            batch_size (int): Batch size used to compute ZS-Scores
            dest (str): custom destination for file
            pbar: App progress bar
            
        """
        if chain == None:
            chain = self.chains[0]

        seq = self.seq[chain]

        # check scores for this protein and model have already been computed
        name = self.name

        user_path = self.user

        if self.name and not dest:
            dest = os.path.join(user_path, f"protein/")
            pdb_file = os.path.join(dest, {self.name})
        elif dest:
            pdb_file = os.path.join(dest, f"{self.name}.pdb")

        # Check if structure already exist
        if os.path.exists(pdb_file):
            self.pdb_file = pdb_file
            self.struc = strucio.load_structure(pdb_file)
            self.chains = chain_parser(self.struc)
            # get ptsm and plddts from loaded files
        else:
            os.makedirs(dest, exist_ok=True)
            all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs = structure_prediction(seqs = [self.seq], names=[self.name])
            pdb = all_pdbs[0]
            pdb.write(pdb_file)

            self.pdb_file = pdb_file
            self.struc = all_pdbs[0]
            self.pTMs = pTMs[0]
            self.pLDDT = mean_pLDDTs[0]
            self.chains = chain_parser(self.struc)
    
        return self.struc
    

    ### Inverse Folding ###
    def esm_if(self, fixed=[], chain=None, temperature=1.0, num_samples=100, model=None, alphabet=None, pbar=None, dest=None, noise=0.2):
        """
        Perform inverse folding using ESM-IF
        """
        user_path = self.user

        if dest:
            csv_path = os.path.join(dest, f"{self.name}.csv")
        else:
            dest = os.path.join(user_path, f"{self.name}/design/")
            csv_path = os.path.join(dest, f"{self.name}.csv")
            
        os.makedirs(dest, exist_ok=True)

        # define chain
        if chain == None:
            chain = self.chains[0]

        # return dataframe of results
        df = esm_design(self.pdb_file, chain, fixed=fixed, temperature=temperature, num_samples=num_samples, model=model, alphabet=alphabet, noise=noise, pbar=pbar)

        df.to_csv(csv_path)

        self.designs = csv_path

        rep_path = os.path.join(dest, "rep")
        struc_path = os.path.join(dest, "struc")

        out = {
            'df':df, 'rep_path':rep_path, 'struc_path':struc_path, 'y_type':'num', 'y_col':'log_likelihood', 
            'seqs_col':'sequence', 'names_col':'names', 'reps':[], 'class_dict':self.class_dict
        }

        return out
    
    # Plot 
    # Plot zero-shot entropy
    def plot_entropy(self, model='esm2', title=None, section=None, chain=None):
        if chain == None and len(self.chains) >= 1:
            chain = self.chains[0]
            seq = self.seq[chain]
        elif len(self.chains) >= 1:
            seq = self.seq[chain]
        else:
            seq = self.seq
            chain == None

        name = self.name

        if self.name:
            dest = os.path.join(self.user, f"{self.name}/zero_shot/results", model)
            if chain:
                dest = os.path.join(self.user, f"{self.name}/zero_shot/results/{chain}", model)
        else:
            dest = os.path.join(self.user, f"protein/zero_shot/results", model)
            if chain:
                dest = os.path.join(self.user, f"protein/zero_shot/results/{chain}", model)

        # Load required data
        self.p = torch.load(os.path.join(dest, "prob_dist.pt"))
        self.mmp = torch.load(os.path.join(dest, "masked_marginal_probability.pt"))
        self.entropy = torch.load(os.path.join(dest, "per_position_entropy.pt"))
        self.logits = torch.load(os.path.join(dest, "masked_logits.pt"))

        # Section handling
        seq_len = len(seq)
        if section is None:
            section = (0, seq_len)
        elif isinstance(section, tuple):
            if len(section) != 2 or any(type(i) != int for i in section):
                raise ValueError("Section must be a tuple of two integers.")
            if section[0] < 0 or section[0] >= seq_len or section[1] > seq_len:
                raise ValueError("Section indices are out of sequence range.")
            if section[1] < section[0]:
                raise ValueError("Section start index must be less than end index.")
        else:
            raise TypeError("Section must be a tuple or None.")

        # Set title
        if title is None:
            title = f"{model_dict[model]} per-position entropy"

        # Plot entropy
        fig = plot_per_position_entropy(per_position_entropy=self.entropy, sequence=seq, highlight_positions=None, dest=None, title=title, section=section)
        return fig


    def plot_scores(self, model='esm2', section=None, color_scheme=None, title=None, highlight_positions=None, chain=None):
        """
        Plot the zero-shot prediction scores for a given model and sequence.

        Args:
            model (str): The name of the model to use for plotting (default: 'esm2').
            section (tuple): Section of the sequence to be shown in the plot - low and high end of sequence to be displayed. Show entire sequence if None (default: None).
            color_scheme (str): Color scheme for the heatmap ('rwb' for red-white-blue, 'r' for reds, 'b' for blues) (default: None).
            title (str): Title of the plot (default: None).
            highlight_positions (dict): Dictionary specifying positions to highlight with the format {position: residue} (default: None).

        Returns:
            fig (matplotlib.figure.Figure): The created matplotlib figure.
        """
        
        if chain == None and len(self.chains) >= 1:
            chain = self.chains[0]
            seq = self.seq[chain]
        elif len(self.chains) >= 1:
            seq = self.seq[chain]
        else:
            seq = self.seq
            chain == None

        name = self.name

        if self.name:
            dest = os.path.join(self.user, f"{self.name}/zero_shot/results", model)
            if chain:
                dest = os.path.join(self.user, f"{self.name}/zero_shot/results/{chain}", model)
        else:
            dest = os.path.join(self.user, f"protein/zero_shot/results", model)
            if chain:
                dest = os.path.join(self.user, f"protein/zero_shot/results/{chain}", model)

        # Load required data
        self.p = torch.load(os.path.join(dest, "prob_dist.pt"))
        self.mmp = torch.load(os.path.join(dest, "masked_marginal_probability.pt"))
        self.entropy = torch.load(os.path.join(dest, "per_position_entropy.pt"))
        self.logits = torch.load(os.path.join(dest, "masked_logits.pt"))

        # Section handling
        seq_len = len(seq)
        if section is None:
            section = (0, seq_len)
        elif isinstance(section, tuple):
            if len(section) != 2 or any(type(i) != int for i in section):
                raise ValueError("Section must be a tuple of two integers.")
            if section[0] < 0 or section[0] >= seq_len or section[1] > seq_len:
                raise ValueError("Section indices are out of sequence range.")
            if section[1] < section[0]:
                raise ValueError("Section start index must be less than end index.")
        else:
            raise TypeError("Section must be a tuple or None.")

        # Set color scheme
        if color_scheme is None:
            color_scheme = "rwb"

        # Set title
        if title is None:
            title = f"{model_dict[model]} Zero-shot prediction scores"

        # Plot heatmap
        fig = plot_heatmap(p=self.mmp, alphabet=alphabet, dest=None, title=title, show=False, remove_tokens=True, color_sheme=color_scheme, section=section, highlight_positions=highlight_positions)
        return fig

    # structure utils
    def get_contacts(self, chain: Union[str, None] = None, target: str = 'protein', dist=7.):
        """
        Get protein protein contacts for a specific chain in a protein. 

        Args:
            chain (str): specify chain for which to compute the contacts. Default 'None' will take the first chain.
            target (str): Specify protein-'protein' contacts or protein-'ligand' contacts, Default 'protein'
        """
        return get_contacts(self.struc, chain, target, dist)


    ### getters and setters ###
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str) and value is not None:
            raise TypeError(f"Expected 'name' to be of type 'str', but got '{type(value).__name__}'")
        self._name = value

    # For seq
    @property
    def seq(self):
        return self._seq

    @seq.setter
    def seq(self, value):
        if not isinstance(value, (str, dict)) and value is not None:
            raise TypeError(f"Expected 'seq' to be of type 'str', but got '{type(value).__name__}'")
        self._seq = value

    # For rep
    @property
    def reps(self):
        return self._reps

    @reps.setter
    def reps(self, value):
        if not isinstance(value, (list, tuple)) and value is not None:
            raise TypeError(f"Expected 'rep' to be of type 'list' or 'tuple', but got '{type(value).__name__}'")
        self._reps = list(value)

    # For y
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if not isinstance(value, (str, int, float)) and value is not None:
            raise TypeError(f"Expected 'rep' to be of type 'int', 'float', or 'str', but got '{type(value).__name__}'")
        self._y = value