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

class Model:
    """
    The Model object allows the user to create machine learning models, using 
    Library objects as input. 

    Attributes:
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
    
    def __init__(self, library: Library = None, model: str = 'rf', x: str = 'ohe', split: str = 'random',
                 k_folds: int = None, grid_search: bool = False, custom_params: dict = None,
                 custom_model: torch.nn.Module = None, optim: str = 'adam', lr: float = 10e-4, seed: int = 21
            ):
        """
        Initialize a new model.

        Args:
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

        self.library = library
        self.model = model
        self.x = x
        self.split = split
        self.k_folds = k_folds
        self.grid_search = grid_search
        self.custom_params = custom_params
        self.custom_model = custom_model
        self.optim = optim
        self.lr = lr
        self.seed = seed

        # determine if the task is a classification or regression task
        if library is not None:
            self.y_type = library.y_type
