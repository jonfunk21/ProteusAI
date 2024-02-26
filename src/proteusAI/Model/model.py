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
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import proteusAI.io_tools as io_tools
import proteusAI.visual_tools as vis
import random
from typing import Union
import json
from joblib import dump
import csv
import torch
import pandas as pd


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
        test_r2 (float): R-squared value of the model on the test set.
        val_true (list): List of true values of the validation dataset.
        val_predictions (list): Predicted values of the validation dataset.
        val_r2 (float): R-squared value of the model on the validation dataset.
    """

    _sklearn_models = ['rf', 'knn', 'svm', 'ffnn'] # add GP
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
        self._model = None
        self.train_data = []
        self.test_data = []
        self.val_data = []
        self.test_true = []
        self.test_predictions = []
        self.val_true = []
        self.val_predictions = []
        self.test_r2 = []
        self.val_r2 = []
        self.rep_path = None

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
            'rep_path': None,
            'split': 'random',
            'k_folds': None,
            'grid_search': False,
            'custom_params': None,
            'custom_model': None,
            'optim': 'adam',
            'lr': 10e-4,
            'seed': 42
        }
        
        # Update defaults with provided keyword arguments
        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)
        

    def train(self, **kwargs):
        """
        Train the model.

        Args:
            library (proteusAI.Library): Data for training.
            model_type (str): choose the model type ['rf', 'svm', 'knn', 'ffnn'],
            x (str): choose the representation type ['esm2', 'esm1v', 'ohe', 'blosum50', 'blosum62'].
            rep_path (str): Path to representations. Default None - will extract from library object.
            split (str): Choose a method to split your data ['random'].
            k_folds (int): Number of folds for cross validation.
            grid_search: Enable grid search.
            custom_params: None. Not implemented yet.
            custom_model: None. Not implemented yet.
            optim (str): Choose optimizer for feed forward neural network. e.g. 'adam'.
            lr (float): Choose a learning rate for feed forward neural networks. e.g. 10e-4.
            seed (int): Choose a random seed. e.g. 42
        """
        # Update attributes if new values are provided
        self._set_attributes(**kwargs)

        # split data
        self.train_data, self.test_data, self.val_data = self.split_data()

        # load model
        self._model = self.model()

        # train
        if self.model_type in self._sklearn_models:
            self.train_sklearn(rep_path=self.rep_path)
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

        train_data, test_data, val_data = [], [], []
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
    

    def load_representations(self, proteins: list, rep_path: Union[str, None] = None):
        """
        Loads representations for a list of proteins.

        Args:
            proteins (list): List of proteins.
            rep_path (str): Path to representations. If None, the method assumes the
                library project path and the representation type used for training.

        Returns:
            list: List of representations.
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if rep_path is None:
            rep_path = os.path.join(self.library.project, f"rep/{self.x}")

        file_names = [protein.name + ".pt" for protein in proteins]

        _, reps = io_tools.load_embeddings(path=rep_path, names=file_names)

        return reps


    def model(self, **kwargs):
        """
        Load or create model according to user specifications and parameters.

        For a complete list and detailed explanation of model parameters, refer to the scikit-learn documentation.
        """

        model_type = self.model_type
        model = None

        # Define the path for the params.json and model
        params_path = f"{self.library.project}/models/{self.model_type}/params.json"

        # Check if params.json exists
        if not os.path.exists(params_path):
            os.makedirs(os.path.dirname(params_path), exist_ok=True)
            with open(params_path, 'w') as f:
                json.dump(kwargs, f)

        if model_type in self._sklearn_models:
            model_params = kwargs.copy()
            #model_params['seed'] = self.seed

            if self.y_type == 'class':
                if model_type == 'rf':
                    model = RandomForestClassifier(**model_params)
                elif model_type == 'svm':
                    model = SVC(**model_params)
                elif model_type == 'knn':
                    model = KNeighborsClassifier(**model_params)
            elif self.y_type == 'num':
                if model_type == 'rf':
                    model = RandomForestRegressor(random_state=self.seed, **model_params)
                elif model_type == 'svm':
                    model = SVR(**model_params)
                elif model_type == 'knn':
                    model = KNeighborsRegressor(**model_params)
            return model
        else:
            raise ValueError(f"Model type '{model_type}' has not been implemented yet")

    

    def train_sklearn(self, rep_path):
        """
        Train sklearn models and save the model.

        Args:
            rep_path (str): representation path
        """
        assert self._model is not None

        # This is for representations that are not stored in memory
        train = self.load_representations(self.train_data, rep_path=rep_path)
        test = self.load_representations(self.test_data, rep_path=rep_path)
        val = self.load_representations(self.val_data, rep_path=rep_path)

        # handle representations that are not esm
        if len(train[0].shape) == 2:
            train = [x.view(-1) for x in train]
            test = [x.view(-1) for x in test]
            val = [x.view(-1) for x in val]

        x_train = torch.stack(train).cpu().numpy()
        x_test = torch.stack(test).cpu().numpy()
        x_val = torch.stack(val).cpu().numpy()

        # TODO: For representations that are stored in memory the computation happens here:

        y_train = [protein.y for protein in self.train_data]
        self.y_test = [protein.y for protein in self.test_data]
        self.y_val = [protein.y for protein in self.val_data]
        self.val_names = [protein.name for protein in self.val_data]

        if self.k_folds is None:
            self._model.fit(x_train, y_train)

            self.test_r2 = self._model.score(x_test, self.y_test)
            self.y_test_pred = self._model.predict(x_test)

            self.val_r2 = self._model.score(x_val, self.y_val)
            self.y_val_pred = self._model.predict(x_val)

            # Save the model
            model_save_path = f"{self.library.project}/models/{self.model_type}/model.joblib"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            dump(self._model, model_save_path)

            if not os.path.exists(f"{self.library.project}/models/{self.model_type}/data/"):
                os.makedirs(f"{self.library.project}/models/{self.model_type}/data/")

            # Save the sequences, y-values, and predicted y-values to CSV
            def save_to_csv(proteins, y_values, y_pred_values, filename):
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['sequence', 'y-value', 'y-predicted'])  # CSV header
                    for protein, y, y_pred in zip(proteins, y_values, y_pred_values):
                        writer.writerow([protein.seq, y, y_pred])

            save_to_csv(self.train_data, y_train, [None]*len(y_train), f"{self.library.project}/models/{self.model_type}/data/train_data.csv")
            save_to_csv(self.test_data, self.y_test, self.y_test_pred, f"{self.library.project}/models/{self.model_type}/data/test_data.csv")
            save_to_csv(self.val_data, self.y_val, self.y_val_pred, f"{self.library.project}/models/{self.model_type}/data/val_data.csv")

            # Save results to a JSON file
            results = {
                'test_r2': self.test_r2,
                'val_r2': self.val_r2
            }
            with open(f"{self.library.project}/models/{self.model_type}/results.json", 'w') as f:
                json.dump(results, f)

        else:
            #kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            #for train_index, test_index in kf.split(X):
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
        if self._model is None:
            raise ValueError(f"Model is 'None'")

        reps = self.load_representations(proteins, rep_path)
        x = torch.stack(reps).cpu().numpy()

        predictions = self._model.predict(x)

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

        if self._model is None:
            raise ValueError(f"Model is 'None'")
        
        reps = self.load_representations(proteins, rep_path)
        x = torch.stack(reps).cpu().numpy()
        y = [protein.y for protein in proteins]

        scores = self._model.score(x, y)

        return scores
    

    def true_vs_predicted(self, y_true: list, y_pred: list, title: Union[str, None] = None, 
                          x_label: Union[str, None] = None, y_label: Union[str, None] = None , plot_grid: bool = True, 
                          file: Union[str, None] = None, show_plot: bool = False):
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

        fig, ax = vis.plot_predictions_vs_groundtruth(y_true, y_pred, title, x_label, 
                                            y_label, plot_grid, file, show_plot)
        
        return fig, ax
    
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