# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(current_path, '..')
sys.path.append(root_path)
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import proteusAI.io_tools as io_tools
import proteusAI.visual_tools as vis
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
        test_true (list): List of true values of the test dataset.
        test_predictions (list): Predicted values of the test dataset.
        val_true (list): List of true values of the validation dataset.
        val_predictions (list): Predicted values of the validation dataset.
        test_r2 (float): R-squared value of the model on the test set.
        val_r2 (float): R-squared value of the model on the validation dataset.
    """

    _sklearn_models = ['rf', 'knn', 'svm', 'ffnn']
    _in_memory_representations = ['ohe', 'blosum50', 'blosum62']
    
    def __init__(self, **kwargs):
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
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.test_true = None
        self.test_predictions = None
        self.val_true = None
        self.val_predictions = None
        self.test_r2 = None
        self.val_r2 = None

        # Set attributes using the provided kwargs
        self._set_attributes(**kwargs)

        # Determine if the task is a classification or regression task
        if self.library is not None:
            self.y_type = self.library.y_type

    
    ### args
    def _set_attributes(self, **kwargs):
        defaults = {
            'library': None,
            'model_type': 'rf',
            'x': 'ohe',
            'split': 'random',
            'k_folds': None,
            'grid_search': False,
            'custom_params': None,
            'custom_model': None,
            'optim': 'adam',
            'lr': 10e-4,
            'seed': 21
        }
        
        # Update defaults with provided keyword arguments
        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)
        

    def train(self, **kwargs):
        """
        Train the model.
        """
        # Update attributes if new values are provided
        self._set_attributes(**kwargs)

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
    

    def load_representations(self, proteins: list, rep_path: str = None):
        """
        Loads representations for a list of proteins.

        Args:
            proteins (list): List of proteins.
            rep_path (str): Path to representations. If None, the method assumes the
                library project path and the representation type used for training.

        Returns:
            list: List of representations.
        """
        if rep_path is None:
            rep_path = os.path.join(self.library.project, f"rep/{self.x}")

        file_names = [protein.name + ".pt" for protein in proteins]

        _, reps = io_tools.load_embeddings(path=rep_path, names=file_names)

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
        Train sklearn models.
        """

        train = self.load_representations(self.train_data)
        test = self.load_representations(self.test_data)
        val = self.load_representations(self.val_data)

        x_train = torch.stack(train).cpu().numpy()
        x_test = torch.stack(test).cpu().numpy()
        x_val = torch.stack(val).cpu().numpy()

        y_train = [protein.y for protein in self.train_data]
        self.y_test = [protein.y for protein in self.test_data]
        self.y_val = [protein.y for protein in self.val_data]

        if self.k_folds is None:
            self.model.fit(x_train, y_train)

            self.test_r2 = self.model.score(x_test, self.y_test)
            self.y_test_pred = self.model.predict(x_test)

            self.val_r2 = self.model.score(x_val, self.y_val)
            self.y_val_pred = self.model.predict(x_val)
        
        else:
            raise ValueError(f"K-fold cross validation has not been implemented yet")
    

    def predict(self, proteins: list, rep_path = None):
        """
        Scores the R-squared value for a list of proteins.

        Args:
            proteins (list): List of proteins to make predictions.
            rep_path (str): Path to representations for proteins in the list.
                If None, the library project path and representation type for training
                will be assumed

        Returns:
            list: Predictions generated by the model.
        """
        if self.model is None:
            raise ValueError(f"Model is 'None'")

        reps = self.load_representations(proteins, rep_path)
        x = torch.stack(reps).cpu().numpy()

        predictions = self.model.predict(x)

        return predictions
    

    def score(self, proteins: list, rep_path = None):
        """
        Make predictions for a list of proteins.

        Args:
            proteins (list): List of proteins to make predictions.
            rep_path (str): Path to representations for proteins in the list.
                If None, the library project path and representation type for training
                will be assumed

        Returns:
            list: Predictions generated by the model.
        """

        if self.model is None:
            raise ValueError(f"Model is 'None'")
        
        reps = self.load_representations(proteins, rep_path)
        x = torch.stack(reps).cpu().numpy()
        y = [protein.y for protein in proteins]

        scores = self.model.score(x, y)

        return scores
    

    def true_vs_predicted(self, y_true: list, y_pred: list, title: str = None, 
                          x_label: str = None, y_label: str = None , plot_grid: bool = True, 
                          file: str = None, show_plot: bool = True):
        """
        Predicts true values versus predicted values.

        Args:
            y_true (list): True y values.
            y_pred (list): Predicted y values.
            title (str): Set the title of the plot. 
            x_label (str): Set the x-axis label.
            y_label (str): Set the y-axis label.
            plot_grid (bool): Display a grid in the plot.
            file (str): Choose a file name.
            show_plot (bool): Choose to show the plot.
        """
        
        if file is not None:
            dest = os.path.join(self.library.project, f"plots/{self.model_type}")
            file = os.path.join(dest, file)
            if not os.path.exists(dest):
                os.makedirs(dest)

        vis.plot_predictions_vs_groundtruth(y_true, y_pred, title, x_label, 
                                            y_label, plot_grid, file, show_plot)


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