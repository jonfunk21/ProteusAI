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
import pandas as pd
from typing import Union, Optional
import torch



class Library:
    """
    The Library object holds information about proteins, labels and representations.
    It is also used to create mathematical representations of proteins.

    The library object serves as input to Model objects, to train machine learning models.

    Attributes:
        project (str): Path to the project. Will create one if the path does not exist.
        data (str): Path to data file ().
        proteins (list): List of proteins.
    """

    # TODO: add VAEs too

    representation_types = ['esm1v', 'esm2', 'ohe', 'blosum62', 'blosum50', 'vae']
    _allowed_y_types = ['class', 'num']

    def __init__(self, project: str, overwrite: bool = False, names: list = [], seqs: list = [], 
                 proteins: list = [], ys: list=[], y_type: Union[str, None] = None):
        """
        Initialize a new library.

        Args:
            name (str): Path to library.
            overwrite (bool): Allow to overwrite files if True.
            names (list): List of protein names.
            seqs (list): List of sequences as strings.
            ys (list): List of y values.
            proteins (Protein, optional): List of proteusAI protein objects.
            y_type: Type of y values class ('class') or numeric ('num') 
        """
        self.project = project
        self.overwrite = overwrite
        self.proteins = proteins
        self.names = names
        self.seqs = seqs
        self.ys = ys
        self.reps = []
        self.y_type = y_type  
        
        # handle case if library does not exist
        if not os.path.exists(self.project):
            self.initialize_library()

        # if the library already exists
        else:
            # load existing information
            print(f"Library {project} already exists. Loading existing library...")
            self.initialize_library()
            self.load_library()

    def initialize_library(self):
        """
        initializing a new library.
        """
        print(f"Initializing library '{self.project}'...")
        
        # create project library
        if not os.path.exists(self.project):
            os.makedirs(self.project)
            print(f"library created at {self.project}")

        # check if sequence have been provided
        if len(self.seqs) > 0:
            # create dummy names if no names are provided
            if len(self.seqs) != len(self.names):
                print(f"Number of sequences ({len(self.seqs)}), does not match the number of names ({len(self.names)})")
                print("Dummy names will be created")
                self.names = [f"protein_{i}" for i in range(len(self.seqs))]

            # create protein objects
            if len(self.ys) == len(self.names):
                for name, seq, y in zip(self.names, self.seqs, self.ys):
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
        Load an existing library.
        """
        print(f"Loading library '{self.project}'...")

        # Check for data
        data_path = os.path.join(self.project, 'data')
        if os.path.exists(data_path):
            data_files = os.listdir(data_path)
            if len(data_files) > 0:
                for dat in data_files:
                    print(f"- Found '{dat}' in 'data/'.")

        # Check for models
        models_path = os.path.join(self.project, 'models')
        if os.path.exists(models_path):
            cls_path = os.path.join(models_path, 'cls')
            if os.path.exists(cls_path):
                classifier_models = os.listdir(cls_path)
                print(f"- Found {len(classifier_models)} classifier models in 'models/cls'.")

            reg_path = os.path.join(models_path, 'reg')
            if os.path.exists(reg_path):
                regressor_models = os.listdir(reg_path)
                print(f"- Found {len(regressor_models)} regressor models in 'models/reg'.")

        # Check for representations
        rep_path = os.path.join(self.project, 'rep')
        if os.path.exists(rep_path):
            for rep_type in self.representation_types:
                rep_type_path = os.path.join(rep_path, rep_type)
                if os.path.exists(rep_type_path):
                    self.reps.append(rep_type)
                    print(f"- Found representations of type '{rep_type}' in 'rep/{rep_type}'.")

        print("Loading done!")


    def read_data(self, data: str, seqs: str, y: Union[str, None] = None, y_type: str='num', 
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

        if file_ext in ['.xlsx', '.xls', '.csv']:
            self._read_tabular_data(data=data, seqs=seqs, y=y, y_type=y_type, names=names, sheet=sheet, file_ext=file_ext)
        elif file_ext == '.fasta':
            self._read_fasta(data)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _read_tabular_data(self, data: str, seqs: str, y: Union[str, None], y_type: Union[str, None], 
                           names: Union[str, None], sheet: Optional[str], file_ext: Union[str, None] = None):
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
        """
        if file_ext in ['.xlsx', '.xls']:
            if sheet is None:
                df = pd.read_excel(data)
            else:
                df = pd.read_excel(data, sheet_name=sheet)
        elif file_ext == '.csv':
            df = pd.read_csv(data)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Validate the columns exist
        if seqs not in df.columns or y not in df.columns:
            raise ValueError("The provided column names do not match the columns in the data file.")

        

        # If names are not provided, generate dummy names
        if names not in df.columns:
            self.names = [f"protein_{i}" for i in range(len(self.seqs))]
        else:
            self.names = df[names].tolist()

        self.seqs = df[seqs].tolist()
        if y is not None:
            ys = df[y].tolist()

            # Create protein objects from names and sequences
            self.proteins = [Protein(name, seq, y=y) for name, seq, y in zip(self.names, self.seqs, ys)]

        else:
            # Create protein objects from names and sequences
            self.proteins = [Protein(name, seq) for name, seq in zip(self.names, self.seqs)]


        self._check_reps()


    def _read_fasta(self, data):
        """
        Read fasta file and create protein objects.

        Args:
            data (str): Path to data.
        """
        names, sequences = io_tools.fasta.load_fasta(data)
        self.names = names
        self.seqs = sequences

        # Create protein objects from names and sequences
        self.proteins = [Protein(name, seq) for name, seq in zip(self.names, self.seqs)]

        self._check_reps()


    def _check_reps(self):
        """
        Check for available representations, store in protein object if representation is found
        """
        reps = os.listdir(os.path.join(self.project, "rep"))
        if len(reps) > 0:
            computed = 0
            rep = None
            for rep in reps:
                rep_path = os.path.join(self.project, f"rep/{rep}")
                proteins = []
                rep_names = [f for f in os.listdir(rep_path) if f.endswith('.pt')]
                for protein in self.proteins:
                    f_name = protein.name + '.pt'
                    if f_name in rep_names:
                        protein._reps.append(rep)
                        computed += 1
                    proteins.append(protein)

                self.proteins = proteins
            print(f"{round(computed/len(self.proteins)*100,2)}% of {rep} computed.")

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
    def compute(self, method: str, batch_size: int = 1):
        """
        Compute representations for proteins.

        Args:
            method (str): Method for computing representation
            batch_size (int, optional): Batch size for representation computation.
        """
        simple_rep_types = ['ohe', 'blosum62', 'blosum50']
        supported_methods = self.representation_types + simple_rep_types

        assert method in supported_methods, f"'{method}' is not a supported method"
        assert isinstance(batch_size, (int, type(None)))

        if method in ["esm2", "esm1v"]:
            self.esm_builder(model=method, batch_size=batch_size)
        elif method == 'ohe':
            self.ohe_builder()
        elif method in ['blosum62', 'blosum50']:
            self.blosum_builder(matrix_type=method.upper())

    
    def esm_builder(self, model: str="esm2", batch_size: int=1):
        """
        Computes esm representations.

        Args:
            model (str): Supports esm2 and esm1v.
            batch_size (int): Batch size for computation.
        """

        dest = os.path.join(self.project, f"rep/{model}")
        if not os.path.exists(dest):
            os.makedirs(dest)

        # Filtering out proteins that have already computed representations
        proteins_to_compute = [protein for protein in self.proteins if not os.path.exists(os.path.join(dest, protein.name + '.pt'))]
        
        print(f"computing {len(proteins_to_compute)} proteins")

        # get names for and sequences for computation
        names = [protein.name for protein in proteins_to_compute]
        seqs = [protein.seq for protein in proteins_to_compute]
        
        # compute representations
        esm_tools.batch_compute(seqs, names, dest=dest, model=model, batch_size=batch_size)
        
        for protein in proteins_to_compute:
            if model not in protein.reps:
                protein.reps.append(model)


    def ohe_builder(self):
        """
        Computes one-hot encoding representations for proteins using one_hot_encoder method.
        Assumes all data fits in memory.
        """

        dest = os.path.join(self.project, "rep/ohe")
        if not os.path.exists(dest):
            os.makedirs(dest)

        proteins_to_compute = [protein for protein in self.proteins if not os.path.exists(os.path.join(dest, protein.name + '.pt'))]
        
        print(f"Computing {len(proteins_to_compute)} proteins")

        sequences = [protein.seq for protein in proteins_to_compute]
        if len(proteins_to_compute) > 0:
            ohe_representations = torch_tools.one_hot_encoder(sequences)

            for i, protein in enumerate(proteins_to_compute):
                torch.save(ohe_representations[i], os.path.join(dest, protein.name + '.pt'))
                if 'ohe' not in protein.reps:
                    protein.reps.append('ohe')


    def blosum_builder(self, matrix_type="BLOSUM62"):
        """
        Computes BLOSUM representations for proteins using blosum_encoding method.
        Assumes all data fits in memory.

        Args:
            matrix_type (str): Type of BLOSUM matrix to use.
        """

        dest = os.path.join(self.project, f"rep/{matrix_type.lower()}")
        if not os.path.exists(dest):
            os.makedirs(dest)

        proteins_to_compute = [protein for protein in self.proteins if not os.path.exists(os.path.join(dest, protein.name + '.pt'))]
        
        print(f"Computing {len(proteins_to_compute)} proteins")

        sequences = [protein.seq for protein in proteins_to_compute]

        if len(proteins_to_compute) > 0:
            blosum_representations = torch_tools.blosum_encoding(sequences, matrix=matrix_type)

            for i, protein in enumerate(proteins_to_compute):
                torch.save(blosum_representations[i], os.path.join(dest, protein.name + '.pt'))
                if matrix_type.lower() not in protein.reps:
                    protein.reps.append(matrix_type.lower())


    def load_representations(self, rep: Union[str, None], proteins: Union[list, None] = None):
        """
        Loads representations for a list of proteins.

        Args:
            rep (str): type of representation to load
            proteins (list): list of proteins to load, load all if None

        Returns:
            list: List of representations.
        """


        rep_path = os.path.join(self.project, f"rep/{rep}")

        if proteins == None:
            file_names = [protein.name + ".pt" for protein in self.proteins]
        else:
            file_names = [protein.name + ".pt" for protein in proteins]

        _, reps = io_tools.load_embeddings(path=rep_path, names=file_names)

        return reps
    

    def plot_tsne(self, rep: str):
        """
        Plot representations.

        rep (str): Representation type to plot
        """

        x = self.load_representations(rep)
        y = self.ys

        fig, ax = vis.plot_tsne(x, y, rep_type=rep)

        return fig, ax
