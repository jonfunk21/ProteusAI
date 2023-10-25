# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(current_path, '..')
sys.path.append(root_path)
import proteusAI.Library as Library
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import proteusAI.io_tools as io_tools
import random

class Model:
    """
    The Model object allows the user to create machine learning models, using 
    Library objects as input. 

    Attributes:
        model: Class variable holding the model.
        library (proteusAI.Library): Library object to train a model. Default None.
        model (str): Type of model to create. Default random forrest ('rf').
        x (str): Choose vector representation for 'x'. Default 'ohe'.
        split (str): Choose data split. Default random split.
        k_folds (int): Choose k for k-fold cross validation. Default None.
        grid_search (bool): Performe a grid search. Default 'False'.
        custom_params (dict): Provide a dictionary or json file with custom parameters. Default None.
        custom_model (torch.nn.Module): Provide a custom PyTorch model.
        optim (str): Optimizer for training PyTorch models. Default 'adam'.
        lr (float): Learning rate for training PyTorch models. Default 10e-4.
        seed (int): random seed. Default 21.
    """

    _sklearn_models = ['rf', 'knn', 'svm', 'ffnn']
    _in_memory_representations = ['ohe', 'blosum50', 'blosum62']
    
    def __init__(self,library: Library = None, model_type: str = 'rf', x: str = 'ohe', split: str = 'random',
                 k_folds: int = None, grid_search: bool = False, custom_params: dict = None,
                 custom_model: torch.nn.Module = None, optim: str = 'adam', lr: float = 10e-4, seed: int = 21
            ):
        """
        Initialize a new model.

        Args:
            model: Contains the trained model.
            library (proteusAI.Library): Library object to train a model. Default None.
            model_type (str): Type of model to create. Default random forrest ('rf').
            x (str): Choose vector representation for 'x'. Default 'ohe'.
            split (str): Choose data split. Default random split.
            k_folds (int): Choose k for k-fold cross validation. Default None.
            grid_search (bool): Performe a grid search. Default 'False'.
            custom_params (dict): Provide a dictionary or json file with custom parameters. Default None.
            custom_model (torch.nn.Module): Provide a custom PyTorch model.
            optim (str): Optimizer for training PyTorch models. Default 'adam'.
            lr (float): Learning rate for training PyTorch models. Default 10e-4.
            seed (int): random seed. Default 21.
        """
        self.model = None
        self.library = library
        self.model_type = model_type
        self.x = x
        self.split = split
        self.k_folds = k_folds
        self.grid_search = grid_search
        self.custom_params = custom_params
        self.custom_model = custom_model
        self.optim = optim
        self.lr = lr
        self.seed = seed

        # Attributes
        self.train_data = None
        self.test_data = None
        self.val_data = None
        
        # determine if the task is a classification or regression task
        if library is not None:
            self.y_type = library.y_type
        

    def train(self, library: Library = None, model_type: str = 'rf', x: str = 'ohe', split: str = 'random',
                 k_folds: int = None, grid_search: bool = False, custom_params: dict = None,
                 custom_model: torch.nn.Module = None, optim: str = 'adam', lr: float = 10e-4, seed: int = 21):
        """
        Train the model.
        """

        # split data
        self.train_data, self.test_data, self.val_data = self.split_data()

        # load model
        self.model = self.load_model()

        # train
        if self.model_type in self._sklearn_models:
            self.train_sklearn()
        else:
            raise ValueError(f"The training method for '{self.model_type}' models has not been implemented yet")

    
    ### Helpers ###
    def split_data(self):
        """
        Split data into train, test, and validation sets. Performs a 80:10:10 split.

            1. 'random': Randomly splits data points
            2. 'site': Splits data by sites in the protein,
                such that the same site cannot be found
                in the training, testing and validation set
            3. 'custom': Splits the data according to a custom pattern

        Returns:
            tuple: returns three lists of train, test and validation proteins.
        """

        proteins = self.library.proteins

        random.seed(self.seed)

        train_size = int(0.80 * len(proteins))
        test_size = int(0.10 * len(proteins))

        if self.split == 'random':
            random.shuffle(proteins)
            # Split the data
            train_data = proteins[:train_size]
            test_data = proteins[train_size:train_size + test_size]
            val_data = proteins[train_size + test_size:]

        # TODO: implement other splitting methods
        if self.split != 'random':
            raise ValueError(f"The {self.split} split has not been implemented yet...")
        
        return train_data, test_data, val_data
    

    def load_representations(self, proteins: list):
        """
        Loads representations for a list of proteins.

        Args:
            proteins (list): List of proteins.

        Returns:
            list: List of representations.
        """

        rep_path = os.path.join(self.library._rep_path, self.x)
        names = [protein.name for protein in proteins]

        _, reps = io_tools.load_embeddings(names=names)

        return reps
    

    def load_model(self):
        """
        Load model according to user specifications.
        """

        model_type = self.model_type

        if model_type in self._sklearn_models:
            if model_type in self._sklearn_models:
                if self.y_type == 'class':
                    if model_type == 'rf':
                        model = RandomForestClassifier()
                    if model_type == 'svm':
                        model = SVC()  # Support Vector Classifier for classification tasks
                    if model_type == 'knn':
                        model = KNeighborsClassifier()
                elif self.y_type == 'num':
                    if model_type == 'rf':
                        model = RandomForestRegressor()
                    if model_type == 'svm':
                        model = SVR()  # Support Vector Regressor for regression tasks
                    if model_type == 'knn':
                        model = KNeighborsRegressor()
        else:
            raise ValueError(f"Model type '{model_type}' has not been implemented yet")
        
        return model
    

    def train_sklearn(self):
        """
        Train sklearn models
        """

        train = self.load_representations(self.train_data)
        test = self.load_representations(self.test_data)
        val = self.load_representations(self.val_data)

        x_train = torch.stack(train).cpu().numpy()
        x_test = torch.stack(test).cpu().numpy()
        x_val = torch.stack(val).cpu().numpy()

        y_train = [protein.y for protein in train]
        y_test = [protein.y for protein in test]
        y_val = [protein.y for protein in val]

        if self.k_folds is None:
            self.model.fit(x_train, y_train)

            test_r2 = self.model.score(x_test, y_test)
            test_predictions = self.model.predict(x_test)

            val_r2 = self.model.score(x_val, y_val)
            val_predictions = self.model.predict(x_val)
        
        else:
            raise ValueError(f"K-fold cross validation has not been implemented yet")
    

    ### Getters and Setters ###
    # Getter and Setter for library
    @property
    def library(self):
        return self._library
    
    @library.setter
    def library(self, library):
        self._library = library
        if library is not None:
            self.y_type = library.y_type

    # Getter and Setter for model_type
    @property
    def model_type(self):
        return self._model_type
    
    @model_type.setter
    def model_type(self, model_type):
        self._model_type = model_type

    # Getter and Setter for x
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = x

    # Getter and Setter for split
    @property
    def split(self):
        return self._split
    
    @split.setter
    def split(self, split):
        self._split = split

    # Getter and Setter for k_folds
    @property
    def k_folds(self):
        return self._k_folds
    
    @k_folds.setter
    def k_folds(self, k_folds):
        self._k_folds = k_folds

    # Getter and Setter for grid_search
    @property
    def grid_search(self):
        return self._grid_search
    
    @grid_search.setter
    def grid_search(self, grid_search):
        self._grid_search = grid_search

    # Getter and Setter for custom_params
    @property
    def custom_params(self):
        return self._custom_params
    
    @custom_params.setter
    def custom_params(self, custom_params):
        self._custom_params = custom_params

    # Getter and Setter for custom_model
    @property
    def custom_model(self):
        return self._custom_model
    
    @custom_model.setter
    def custom_model(self, custom_model):
        self._custom_model = custom_model

    # Getter and Setter for optim
    @property
    def optim(self):
        return self._optim
    
    @optim.setter
    def optim(self, optim):
        self._optim = optim

    # Getter and Setter for lr
    @property
    def lr(self):
        return self._lr
    
    @lr.setter
    def lr(self, lr):
        self._lr = lr

    # Getter and Setter for seed
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        self._seed = seed