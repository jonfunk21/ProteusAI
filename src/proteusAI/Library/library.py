# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(current_path, '..')
sys.path.append(root_path)
from proteusAI.Protein.protein import Protein
import proteusAI.ml_tools.esm_tools.esm_tools as esm_tools
import proteusAI.ml_tools.torch_tools as torch_tools
import proteusAI.io_tools as io_tools
import proteusAI.visual_tools as vis
import proteusAI.struc as pai_struc
import pandas as pd
from typing import Union, Optional
import torch
from sklearn.preprocessing import LabelEncoder
folder_path = os.path.dirname(os.path.realpath(__file__))
USR_PATH = os.path.join(folder_path, '../../../usrs')


class Library:
    """
    The Library object holds information about proteins, labels and representations.
    It is also used to create mathematical representations of proteins.

    The library object serves as input to Model objects, to train machine learning models.

    Attributes:
        usr (str): User name - determines where data will be stored. default: Guest
        data (str): Path to data file ().
        proteins (list): List of proteins.
    """

    # TODO: add VAEs too

    representation_types = ['esm1v', 'esm2', 'ohe', 'blosum62', 'blosum50', 'vae']
    _allowed_y_types = ['class', 'num']
    in_memory = ['ohe', 'blosum62', 'blosum50']

    def __init__(self, user: str = 'guest', source: Union[str,dict,None] = None, seqs_col: Union[str,None] = None, names_col: Union[str,None] = None,
                 y_col: Union[str,None] = None, y_type: Union[str,None] = None, sheet: Union[str,None] = None, fname: Union[str, None] = None):
        """
        Initialize a new library.

        Args:
            user (str): User name.
            source (str, or dict): Source of data, either a file or a data package created from a diversification step.
            seqs_col (str): Column name for the sequences, if the source is a '.csv' or Excel file.
            names_col (str): Column name for the names (sequence descriptions), if the source is a '.csv' or Excel file.
            y_col (str): Column name for the y-values (e.g. fitness, stability, activity), if the source is a '.csv' or Excel file.
            y_type (str): Specify the data type of y-values, either categorical or numerical. Will parse automatically if none is provided.
            sheet (str): Specify the excel sheet name, if the source is an Excel file.
            fname (str): Only relevant for the app - provides the real file name instead of temporary file name from shiny.

        Parameters:
            data (df): dataframe of uploaded data (raw data).
            seqs (list): list of sequence.
            y (list): list of y values.
            y_pred (list): list of predicted y values.
            y_type (str): Type of y values ('class' or 'num').
            names (list): names of sequences.
            reps (list): list of computed representations.
            proteins (list): list of protein objects.
            pred_data (bool): Using predicted data, e.g. predicted ZS y-values with no real y-values
        """
        # Arguments
        self.user = os.path.join(USR_PATH, user)
        self.source = source
        self.seq_col = seqs_col
        self.names_col = names_col
        self.y_col = y_col
        self.y_type = y_type
        self.sheet = sheet
        self.fname = fname

        # Parameters
        self.data = None
        self.seqs = None
        self.y = None
        self.y_pred = None
        self.names = None
        self.reps = self.in_memory.copy()
        self.source_path = None
        self.rep_path = None
        self.strucs = None
        self.struc_path = None
        self.class_dict = None
        self.pred_data = False

        # Create user if user does not exist
        if not os.path.exists(self.user):
            self.initialize_user()

        # Initialize library from file or from inheritance
        if type(self.source) == str:
            self.init_from_file()
        else:
            self.init_from_inheritance()

    
    def init_from_file(self):
        """
        Initialize a library from a file.
        """
        
        # handle app case
        if self.fname:
            fname = self.fname.split('.')[0]
            file_extension = self.fname.split('.')[-1]
        else:
            f = self.source.split('/')[-1]
            fname = f.split('.')[0]
            file_extension = f.split('.')[-1]

        # set paths
        self.source_path = os.path.join(self.user, fname)
        self.rep_path = os.path.join(self.source_path, 'library/rep')
        self.struc_path = os.path.join(self.source_path, 'library/struc')
        
        # create user library if user does not exist
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(os.path.join(self.source_path, 'library/rep'), exist_ok=True)
        os.makedirs(os.path.join(self.source_path, 'library/struc'), exist_ok=True)

        # load the data
        if file_extension in ['xlsx', 'xls', 'csv']:
            self._read_tabular_data(in_file=self.source, seqs=self.seq_col, y=self.y_col, y_type=self.y_type, names=self.names_col, 
                                    sheet=self.sheet, file_ext=file_extension)
        elif file_extension in ['fasta', 'fa']:
            self._read_fasta(self.source)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        

    def init_from_inheritance(self):
        """
        Initialize a library from another process.

        Args:
            data (dict): Contains parent path and path to data file.
        """

        # For the future adapt this so it can handle multi-factor optimization
        # extract data from parent file
        data = self.source

        # Parsing parameters
        self.rep_path = data['rep_path']
        self.struc_path = data['struc_path']
        self.seq_col = data['seqs_col']
        self.names_col = data['names_col']
        self.reps = data['reps']
        self.class_dict = data['class_dict']

        # Parsing arguments
        df = data['df']
        self.seqs = df[self.seq_col].to_list()
        self.names = df[self.names_col].to_list()
        self.y_type = data['y_type']
        self.data = df

        # if true y values are there
        if 'y_col' in data.keys():
            self.y_col = data['y_col']
            self.y = df[self.y_col].to_list()
        else:
            self.y = [None] * len(self.seqs)

        # if predicted y values are there
        if 'y_pred_col' in data.keys():
            self.y_pred_col = data['y_pred_col']
            self.y_pred = df[self.y_pred_col].to_list()
        else:
            self.y_pred = [None] * len(self.seqs)
        
        # if predicted y values are there
        if 'y_sigma_col' in data.keys():
            self.y_sigma_col = data['y_sigma_col']
            self.y_sigma = df[self.y_sigma_col].to_list()
        else:
            self.y_sigma = [None] * len(self.y)

        # if predicted y values are there
        if 'acq_col' in data.keys():
            self.acq_col = data['acq_col']
            self.acq_score = df[self.acq_col].to_list()
        else:
            self.acq_score = [None] * len(self.y)

        if 'pred_data' in data.keys():
            self.pred_data = data['pred_data']

        # create proteins
        self.proteins = [Protein(name, seq, y=y, y_pred=y_pred, y_sigma=y_sigma, acq_score=acq_score, user=self.user) for name, seq, y, y_pred, y_sigma, acq_score in zip(self.names, self.seqs, self.y, self.y_pred, self.y_sigma, self.acq_score)]


    def initialize_user(self):
        """
        initializing a new library.
        """
        print(f"Initializing user '{self.user}'...")
        
        # create user library if user does not exist
        if not os.path.exists(self.user):
            os.makedirs(self.user)
            if self.fname and not os.path.exists(os.path.join(self.user, self.fname.split('.')[0])):
                fname = self.fname.split('.')[0]
                os.makedirs(os.path.join(self.user, f'{fname}'))
                os.makedirs(os.path.join(self.user, f'{fname}/library'))
                os.makedirs(os.path.join(self.user, f'{fname}/zero_shot'))
                os.makedirs(os.path.join(self.user, f'{fname}/design'))
            print(f"User created at {self.user}")

        # check if sequence have been provided
        if len(self.seqs) > 0:
            # create dummy names if no names are provided
            if len(self.seqs) != len(self.names):
                print(f"Number of sequences ({len(self.seqs)}), does not match the number of names ({len(self.names)})")
                print("Dummy names will be created")
                self.names = [f"protein_{i}" for i in range(len(self.seqs))]

            # create protein objects TODO: This is the slow step
            if len(self.y) == len(self.names):
                for name, seq, y in zip(self.names, self.seqs, self.y):
                    protein = Protein(name, seq, y=y)
                    self.proteins.append(protein)

            else:
                for name, seq in zip(self.names, self.seqs):
                    protein = Protein(name, seq)
                    self.proteins.append(protein)

        print('Done!')

            
    ### IO ###
    def load_library(self):
        """
        Checking the availability of precomputed representations.
        """
        print(f"Loading library '{self.user}'...")

        # Check for representations
        if self.rep_path == None:
            if self.file:
                fname = self.file.split('.')[0]
                rep_path = os.path.join(self.user, f'{fname}/library/rep')
                struc_path = os.path.join(self.user, f'{fname}/library/struc')
                self.rep_path = rep_path
                self.struc_path = struc_path
            else:
                rep_path = os.path.join(self.user, 'rep')
                struc_path = os.path.join(self.user, 'struc')
                self.rep_path = rep_path
                self.struc_path = struc_path
        else:
            rep_path = '/'.join(self.rep_path.split('/')[:-1])

        if os.path.exists(rep_path):
            for rep_type in self.representation_types:
                rep_type_path = os.path.join(rep_path, rep_type)
                if os.path.exists(rep_type_path):
                    self.reps.append(rep_type)
                    print(f"- Found representations of type '{rep_type}' in 'rep/{rep_type}'.")

        print("Loading done!")


    def read_data(self, data: str, seqs: Union[str, None] = None, y: Union[str, None] = None, y_type: str='num', 
                  names: Union[str, None] = None, sheet: Optional[str] = None):
        """
        Reads data from a CSV, Excel, or FASTA file and populates the Library object.

        Args:
            data (str): Path to the data file. Tabular data or fasta
            seqs (str): Column name for sequences in the data file.
            y (str): Column name for y values in the data file.
            y_type (str): Type of y values ('class' or 'num').
            names (str, optional): Column name for sequence names in the data file.
            sheet (str, optional): Name of the Excel sheet to read.
        """

        assert y_type in self._allowed_y_types
        self.y_type = y_type
        
        # Determine file type based on extension
        file_ext = os.path.splitext(data)[1].lower()
        self.file = data.split('/')[-1].split('.')[0]
        
        self.rep_path = os.path.join(self.user, self.file, "library/rep")
        self.struc_path = os.path.join(self.user, self.file, "library/struc")

        if file_ext in ['.xlsx', '.xls', '.csv']:
            self._read_tabular_data(in_file=data, seqs=seqs, y=y, y_type=y_type, names=names, sheet=sheet, file_ext=file_ext)
        elif file_ext in ['.fasta', '.fa']:
            self._read_fasta(data)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")


    def _read_tabular_data(self, in_file: str, seqs: str, y: Union[str, None], y_type: Union[str, None], 
                           names: Union[str, None], sheet: Optional[str], file_ext: Union[str, None] = None,
                           check_rep: bool = True, check_strucs: bool = True):
        """
        Reads data from a CSV, Excel, and populates the Library object. Called by read_data

            Args:
                data (str): Path to the data file.
                seqs (str): Column name for sequences in the data file.
                y (str): Column name for y values in the data file.
                y_type (str): Type of y values ('class' or 'num').
                names (str, optional): Column name for sequence names in the data file.
                sheet (str, optional): Name of the Excel sheet to read.
                file_ext (str): file extension
                check_rep (bool): Check if all representations have been computed.
                check_sturcs (bool): Check if structures have been computed.
        """
        if file_ext in ['xlsx', 'xls']:
            if sheet is None:
                df = pd.read_excel(in_file)
            else:
                df = pd.read_excel(in_file, sheet_name=sheet)
        elif file_ext == 'csv':
            df = pd.read_csv(in_file)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        self.data = df
        # Validate the columns exist
        if seqs not in df.columns or y not in df.columns:
            raise ValueError("The provided column names do not match the columns in the data file.")

        # If names are not provided, generate dummy names
        if names not in df.columns:
            self.names = [f"protein_{i}" for i in range(len(self.seqs))]
        else:
            self.names = df[names].tolist()

        self.seqs = df[seqs].tolist()

        # Handle y values
        if y is not None:
            self.y = df[y].tolist()

            if y_type == 'class':
                self.y, self.class_dict = self._encode_categorical_labels(self.y)

            # Create protein objects with y values
            self.proteins = [Protein(name, seq, y=y, user=self.user) for name, seq, y in zip(self.names, self.seqs, self.y)]
        else:
            # Create protein objects without y values
            self.proteins = [Protein(name, seq, user=self.user) for name, seq in zip(self.names, self.seqs)]

        if check_rep:
            self._check_reps()

        if check_strucs:
            self._check_strucs()


    def _read_fasta(self, in_file, check_rep: bool = False):
        """
        Read fasta file and create protein objects.

        Args:
            data (str): Path to data.
            check_rep (bool): Check if all representations have been computed.
        """
        names, sequences = io_tools.fasta.load_fasta(in_file)
        self.names = names
        self.seqs = sequences

        # Create protein objects from names and sequences
        self.proteins = [Protein(name, seq) for name, seq in zip(self.names, self.seqs)]
        self.y = [None] * len(self.proteins)
        df = pd.DataFrame({"names":names, "sequence":self.seqs, "y":self.y})
        self.data = df

        if check_rep:
            self._check_reps()


    def _check_reps(self):
        """
        Check for available representations, store in protein object if representation is found
        """
        reps = [r for r in os.listdir(self.rep_path) if not r.startswith('.')]
        if len(reps) > 0:
            for rep in reps:
                rep_names = [f for f in os.listdir(os.path.join(self.rep_path, rep)) if f.endswith('.pt')]
                if len(rep_names) >= len(set(self.names)):
                    self.reps.insert(0, rep)  # Insert representation at the front
                    for protein in self.proteins:
                        f_name = protein.name + '.pt'
                        if os.path.exists(os.path.join(self.rep_path, rep, f_name)):
                            protein._reps.insert(0, rep)


    def _check_strucs(self):
        """
        Check for available representations, store in protein object if representation is found
        """
        strucs = [r for r in os.listdir(self.struc_path) if not r.startswith('.')]
        if len(strucs) > 0:
            for struc in strucs:
                struc_names = [f for f in os.listdir(os.path.join(self.struc_path, struc)) if f.endswith('.pt')]
                if len(struc_names) == len(set(self.names)):
                    self.strucs.append(struc)
                    for protein in self.proteins:
                        f_name = protein.name + '.pdb'
                        protein._strucss.append(strucs) 


    def _encode_categorical_labels(self, ys):
        """
        Encode categorical labels into numerical format.
        """
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(ys).tolist()
        class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
        return encoded_labels, class_mapping


    ### Utility ###
    def rename_proteins(self, new_names: Union[list, tuple]):
        """
        Renames all proteins in the Library object with provided names.

        Args:
            new_names (list or tuple): List of new protein names.
        """
        
        if len(new_names) != len(self.proteins):
            raise ValueError(f"Number of provided names ({len(new_names)}) does not match number of proteins in the library ({len(self.proteins)}).")

        # Update names of Protein objects and in the self.names list
        for protein, new_name in zip(self.proteins, new_names):
            protein.name = new_name

        self.names = new_names

        self._check_reps()
        self._check_strucs()

    
    def set_y_values(self, y_values: list):
        """
        Sets the y values for all proteins.

        Args:
            y_values (list): List of y values.
        """

        if len(y_values) != len(self.proteins):
            raise ValueError(f"Number of provided y values ({len(y_values)}) does not match number of proteins in the library ({len(self.proteins)}).")

        for protein, y_value in zip(self.proteins, y_values):
            protein.y = y_value

    
    ### Representation builders ###
    def compute(self, method: str, batch_size: int = 100, dest: Union[str, None] = None, pbar=None, device=None, proteins=None):
        """
        Compute representations for proteins.

        Args:
            method (str): Method for computing representation
            batch_size (int, optional): Batch size for representation computation.
            dest (str): destination of representations
            pbar: Progress bar for shiny app.
            device (str): Choose hardware for computation. Default 'None' for autoselection
                          other options are 'cpu' and 'cuda'. 
            proteins (list): list of specific proteins. Optional
        """
        simple_rep_types = ['ohe', 'blosum62', 'blosum50']
        supported_methods = self.representation_types + simple_rep_types

        assert method in supported_methods, f"'{method}' is not a supported method"
        assert isinstance(batch_size, (int, type(None)))

        if method in ["esm2", "esm1v"]:
            self.esm_builder(model=method, batch_size=batch_size, dest=dest, pbar=pbar, device=device)
        elif method == 'ohe':
            reps = self.ohe_builder(dest=dest, pbar=pbar, proteins=proteins)
            return reps
        elif method in ['blosum62', 'blosum50']:
            reps = self.blosum_builder(matrix_type=method.upper(), dest=dest, pbar=pbar, proteins=proteins)
            return reps

    
    def esm_builder(self, model: str="esm2", batch_size: int=10, dest: Union[str, None] = None, pbar=None, device=None):
        """
        Computes esm representations.

        Args:
            model (str): Supports esm2 and esm1v.
            batch_size (int): Batch size for computation.
            dest (str): destination of representations
            pbar: Progress bar for shiny app.
            device (str): Choose hardware for computation. Default 'None' for autoselection
                          other options are 'cpu' and 'cuda'. 
        """
        
        dest = os.path.join(self.rep_path, model)

        if not os.path.exists(dest):
            os.makedirs(dest)

        # Filtering out proteins that have already computed representations
        proteins_to_compute = [protein for protein in self.proteins if not os.path.exists(os.path.join(dest, protein.name + '.pt'))]
        
        print(f"computing {len(proteins_to_compute)} proteins")
        
        if pbar:
            pbar.set(message=f"Computing {len(proteins_to_compute)} representations", detail=f"...")

        # get names for and sequences for computation
        names = [protein.name for protein in proteins_to_compute]
        seqs = [protein.seq for protein in proteins_to_compute]
        
        # compute representations
        esm_tools.batch_compute(seqs, names, dest=dest, model=model, batch_size=batch_size, pbar=pbar, device=device)
        
        for protein in proteins_to_compute:
            if model not in protein.reps:
                protein.reps.append(model)
        
        if model not in self.reps:
            self.reps.append(model)


    def ohe_builder(self, dest: Union[str, None] = None, pbar=None, proteins=None):
        """
        Computes one-hot encoding representations for proteins using one_hot_encoder method.
        Assumes all data fits in memory.

        Args:
            dest (str): destination of representations
        """
        if proteins:
            seqs = [prot.seq for prot in proteins]
        else:
            seqs = self.seqs
        
        # Determine the maximum sequence length for padding
        max_sequence_length = max(len(seq) for seq in self.seqs)
        
        # Compute the one-hot encoding with the calculated padding
        ohe_representations = torch_tools.one_hot_encoder(seqs, pbar=pbar, padding=max_sequence_length)
        
        return ohe_representations


    def blosum_builder(self, matrix_type="BLOSUM62", dest: Union[str, None] = None, pbar=None, proteins=None):
        """
        Computes BLOSUM representations for proteins using blosum_encoding method.
        Assumes all data fits in memory.

        Args:
            matrix_type (str): Type of BLOSUM matrix to use.
            dest (str): destination
        """
        if proteins:
            seqs = [prot.seq for prot in proteins]
        else:
            seqs = self.seqs
        
        # Determine the maximum sequence length for padding
        max_sequence_length = max(len(seq) for seq in self.seqs)
        
        # Compute the BLOSUM encoding with the calculated padding
        blosum_representations = torch_tools.blosum_encoding(seqs, matrix=matrix_type, pbar=pbar, padding=max_sequence_length)
        
        return blosum_representations


    def load_representations(self, rep: Union[str, None], proteins: Union[list, None] = None):
        """
        Loads representations for a list of proteins.

        Args:
            rep (str): type of representation to load
            proteins (list): list of proteins to load, load all if None

        Returns:
            list: List of representations.
        """

        if self.rep_path == None:
            rep_path = os.path.join(self.user, f"rep/{rep}")
        else:
            rep_path = os.path.join(self.rep_path, rep)

        if proteins == None:
            file_names = [protein.name + ".pt" for protein in self.proteins]
        else:
            file_names = [protein.name + ".pt" for protein in proteins]

        if rep in self.in_memory:
            reps = self.compute(method=rep, proteins=proteins)
        else:
            _, reps = io_tools.load_embeddings(path=rep_path, names=file_names)
        return reps
    
    ### Folding ###
    def fold(self, names, model: str = 'esm_fold', num_recycles: int = 0, pbar=None, relax: bool = False):
        """
        Fold sequences by their names. 

        Args:
            names (list): list of names to fold
            model (str): model used for folding, currently only esm_fold
            num_recycles (int): number of recycling steps, default 0
            pbar: progress bar for app
        """
        out = False

        seqs = self.data[self.data[self.names_col].isin(names)][self.seq_col].to_list()
        if len(seqs) > 0:
            all_headers, all_sequences, all_pdbs, pTMs, mean_pLDDTs = esm_tools.structure_prediction(names=names, seqs=seqs, num_recycles=num_recycles, pbar=pbar)

            if not os.path.exists(self.struc_path):
                os.makedirs(self.struc_path)

            for i, name in enumerate(names):
                dest = os.path.join(self.struc_path, name + '.pdb')
                f = all_pdbs[i]
                f.write(dest)

            if relax:
                if pbar:
                    pbar.set(message="Initiating energy minimization", detail=f"Minimizing {len(names)} structures...")
                for i, name in enumerate(names):
                    if pbar:
                        pbar.set(i+1, message="Minimizing energy", detail=f"{i+1}/{len(names)} remaining...")
                    self.relax_struc(name)
        
            df = pd.DataFrame({"name":all_headers, "sequence":all_sequences, "pLDDT":mean_pLDDTs, "pTM":pTMs})
            out = {
                'df':df, 'rep_path':self.rep_path, 'struc_path':self.struc_path, 'y_type':'num', 'y_col':'pLDDT', 
                'seqs_col':'sequence', 'names_col':'name', 'reps':self.reps, 'class_dict':self.library.class_dict
            }
            
        return out

    
    ### Structure ###
    def relax_struc(self, name: str):
        """
        Perform energy minimization on a protein structure.
        """
        f = os.path.join(self.struc_path, name + '.pdb')
        pai_struc.relax_pdb(f, dest=self.struc_path)

    
   
    def struc_geom(self, ref, residues: dict = {}):
        """
        Compute the difference in geometry of a dictionary of residues in a library compared to a reference structure.
        The residues must be present in both structures, or they won't be compared.

        Args:
            ref (proteusAI.Protein): Reference protein structure.
            residues (dict): Dictionary where keys are chain identifiers and values are lists of residues to compare.

        Returns:
            A DataFrame containing comparison results for each protein in the library.
        """

        ref_struc = ref.struc
        ref_chi = pai_struc.compute_chi_angles(ref_struc, residues)

        rmsds = []
        delta_chis = []

        for prot in self.proteins:
            target_path = os.path.join(self.struc_path, prot.name + '.pdb') 
            ref_struc, target = pai_struc.align(ref.pdb_file, target_path)
            rmsd = pai_struc.compute_rmsd(ref_struc, target)
            target_chi = pai_struc.compute_chi_angles(target, residues)

            # Calculate delta chi for each chain and residue specified
            delta_chi = {}
            for chain, res_list in residues.items():
                for res_id in res_list:
                    if (chain, res_id) in ref_chi and (chain, res_id) in target_chi:
                        # Compute difference for each chi angle
                        chi_differences = [
                            abs(ref_angle - tar_angle) for ref_angle, tar_angle in zip(ref_chi[(chain, res_id)], target_chi[(chain, res_id)])
                        ]
                        delta_chi[(chain, res_id)] = chi_differences

            delta_chis.append(delta_chi)
            rmsds.append(rmsd)

        results_path = os.path.join(self.rep_path, f'{prot.name}_analysis.csv')

        if not os.path.exists(self.rep_path):
            os.makedirs(self.rep_path, exist_ok=True)

        df = pd.DataFrame({"names": [prot.name for prot in self.proteins], "RMSD": rmsds, "delta_chi": delta_chis})
        df.to_csv(results_path)    

        return df


    def plot_tsne(self, rep: str, y_upper=None, y_lower=None, names=None, highlight_mask=None, highlight_label=None):
        """
        Plot representations with optional thresholds and point names.

        Args:
            rep (str): Representation type to plot.
            y_upper (float, optional): Upper threshold for special coloring.
            y_lower (float, optional): Lower threshold for special coloring.
            names (List[str], optional): List of names for each point.
            highlight_mask (list): List of 0s and 1s to highlight plot. Default None.
            highlight_label (str): Text for the legend entry of highlighted points.
        """

        x = self.load_representations(rep)
        y = self.y

        if self.y_type == 'class':
            y = [self.class_dict[i] for i in y]

        fig, ax, df = vis.plot_tsne(x, y, y_upper=y_upper, y_lower=y_lower, names=names, rep_type=rep, y_type=self.y_type, random_state=42,
                                    highlight_mask=highlight_mask, highlight_label=highlight_label)

        return fig, ax, df


    def plot_umap(self, rep: str, y_upper=None, y_lower=None, names=None, highlight_mask=None, highlight_label=None):
        """
        Plot representations with optional thresholds and point names.

        Args:
            rep (str): Representation type to plot.
            y_upper (float, optional): Upper threshold for special coloring.
            y_lower (float, optional): Lower threshold for special coloring.
            names (List[str], optional): List of names for each point.
            highlight_mask (list): List of 0s and 1s to highlight plot. Default None.
            highlight_label (str): Text for the legend entry of highlighted points.
        """

        x = self.load_representations(rep)
        y = self.y

        if self.y_type == 'class':
            y = [self.class_dict[i] for i in y]

        fig, ax, df = vis.plot_umap(x, y, y_upper=y_upper, y_lower=y_lower, names=names, rep_type=rep, y_type=self.y_type, random_state=42, 
                                    highlight_mask=highlight_mask, highlight_label=highlight_label)

        return fig, ax, df


    def plot_pca(self, rep: str, y_upper=None, y_lower=None, names=None, highlight_mask=None, highlight_label=None):
        """
        Plot representations with optional thresholds and point names.

        Args:
            rep (str): Representation type to plot.
            y_upper (float, optional): Upper threshold for special coloring.
            y_lower (float, optional): Lower threshold for special coloring.
            names (List[str], optional): List of names for each point.
            highlight_mask (list): List of 0s and 1s to highlight plot. Default None.
            highlight_label (str): Text for the legend entry of highlighted points.
        """

        x = self.load_representations(rep)
        y = self.y
        
        if self.y_type == 'class':
            y = [self.class_dict[i] for i in y]

        fig, ax, df = vis.plot_pca(x, y, y_upper=y_upper, y_lower=y_lower, names=names, rep_type=rep, y_type=self.y_type, random_state=42,
                                   highlight_mask=highlight_mask, highlight_label=highlight_label)

        return fig, ax, df
    
    def __len__(self):
        return len(self.seqs)


    def top_n(self, n: int = 10, ascending=False):
        """
        Returns the top n-proteins by y-value.

        Args:
            n (int): Number of top variants. Default 10
            ascending (bool): sort in ascending or descending order
        """
        
        # Get the list of proteins
        proteins = self.proteins
        
        # Sort the proteins by y-value
        sorted_proteins = sorted(proteins, key=lambda protein: protein.y, reverse=not ascending)
        
        # Return the top n proteins
        return sorted_proteins[:n]
