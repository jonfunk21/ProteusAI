# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from typing import Union
import inspect
import os
import warnings

class Protein:
    """
    The protein object contains information about a single protein,
    such as its sequence, name, and its mathematical representations.
    
    Attributes:
        name (str): Name/id of the protein.
        seq (str): Protein sequence.
        rep (list): List of available representations.
        rep_path (str): Path to representations directory.
    """

    def __init__(self, name: Union[str, None] = None, seq: Union[str, None] = None, reps: Union[list, tuple] = [], rep_path: Union[str, None] = None, y = None):
        """
        Initialize a new protein object.

        Args:
            name (str): Name/id of the protein.
            seq (str): Protein sequence.
            reps (list): List of available representations.
            rep_path (str): Path to representations directory. Default './rep/'.
            y (float, int, str): Label for the protein.
        """

        # assertions and checks
        assert isinstance(name, (str, type(None)))
        assert isinstance(seq, (str, type(None)))
        assert isinstance(reps, (list, tuple))
        assert isinstance(rep_path, (str, type(None)))

        self.name = name
        self.seq = seq
        self.reps = list(reps)
        self.y = y
        
        # If path is not provided, use the directory of the calling script
        if rep_path is None:
            caller_path = os.path.dirname(self.get_caller_path())
            self.path = os.path.join(caller_path, 'rep')
        else:
            assert isinstance(rep_path, str)
            self.path = rep_path

    def get_caller_path(self):
        """Returns the path of the script that called the function."""
        caller_frame = inspect.stack()[1]
        caller_path = caller_frame[1]
        return os.path.abspath(caller_path)

    def __str__(self):
        return f"proteusAI.Protein():\n____________________\nname\t: {self.name}\nseq\t: {self.seq}\nrep\t: {self.reps}\ny\t{self.y}"
    
    __repr__ = __str__

    ### Class methods ###
    @classmethod
    def load_fasta(cls, file: str):
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
        return cls(name=name, seq=seq)
    
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
        if not isinstance(value, str) and value is not None:
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

    # For path
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if not isinstance(value, str) and value is not None:
            raise TypeError(f"Expected 'path' to be of type 'str', but got '{type(value).__name__}'")
        self._path = value

    # For y
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if not isinstance(value, (str, int, float)) and value is not None:
            raise TypeError(f"Expected 'rep' to be of type 'int', 'float', or 'str', but got '{type(value).__name__}'")
        self._y = value