# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import csv
import os
import random
import sys
from typing import Union
import hashlib

import gpytorch
import hdbscan
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from scipy.stats import pearsonr, kendalltau

import proteusAI.ml_tools.bo_tools as BO
import proteusAI.visual_tools as vis
from proteusAI.Library import Library
from proteusAI.Protein import Protein
from proteusAI.ml_tools.torch_tools import GP, computeR2, predict_gp

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(current_path, "..")
sys.path.append(root_path)


class Model:
    """
    The Model object allows the user to create machine learning models, using
    Library objects as input.

    Attributes:
        model: Class variable holding the model.
        library (proteusAI.Library): Library object to train a model. Default None.
        model (str): Type of model to create. Default random forrest ('rf').
        rep (str): Choose protein representation. Default 'ohe'.
        x (str): Future warining: please use rep instead
        split (str): Choose data split. Default random split.
        k_folds (int): Choose k for k-fold cross validation. Default None.
        grid_search (bool): Performe a grid search. Default 'False'.
        custom_params (dict): Provide a dictionary or json file with custom parameters. Default None.
        custom_model (torch.nn.Module): Provide a custom PyTorch model.
        optim (str): Optimizer for training PyTorch models. Default 'adam'.
        lr (float): Learning rate for training PyTorch models. Default 10e-4.
        seed (int): random seed. Default 42.
        test_true (list): List of true values of the test dataset.
        test_predictions (list): Predicted values of the test dataset.
        test_r2 (float): R-squared value of the model on the test set.
        val_true (list): List of true values of the validation dataset.
        val_predictions (list): Predicted values of the validation dataset.
        val_r2 (float): R-squared value of the model on the validation dataset.
    """

    _clustering_algs = ["hdbscan"]
    _sklearn_models = ["rf", "knn", "svm", "ffnn", "ridge"]
    _pt_models = ["gp"]
    _in_memory_representations = ["ohe", "blosum50", "blosum62", "vhse"]
    defaults = {
        "library": None,
        "model_type": "rf",
        "rep": "ohe",
        "rep_path": None,
        "split": "smart",
        "k_folds": None,
        "grid_search": False,
        "custom_params": None,
        "custom_model": None,
        "optim": "adam",
        "lr": 10e-4,
        "seed": 42,
        "dest": None,
        "pbar": None,
        "min_cluster_size": 30,
        "min_samples": 50,
        "metric": "euclidean",
        "cluster_selection_epsilon": 0.1,
        "dr_method": "umap",
    }

    def __init__(self, **kwargs):
        """
        Initialize a new model.

        Args:
            model: Contains the trained model.
            library (proteusAI.Library): Library object to train a model. Default None.
            model_type (str): Type of model to create. Default random forrest ('rf').
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
        self.unlabelled_data = []
        self.test_true = []
        self.test_predictions = []
        self.val_true = []
        self.val_predictions = []
        self.test_r2 = []
        self.val_r2 = []
        self.val_pearson = []
        self.val_ken_tau = []
        self.y_unlabelled_pred = []
        self.y_unlabelled_sigma = []
        self.y_unlabelled = []
        self.calibration = None
        self.rep_path = None
        self.dest = None
        self.y_best = None
        self.out_df = None
        self.search_df = None

        # check for device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set attributes using the provided kwargs
        self._set_attributes(**kwargs)

    ### args
    def _set_attributes(self, **kwargs):
        defaults = self.defaults

        # Update defaults with provided keyword arguments
        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)

    def _update_attributes(self, **kwargs):
        defaults = self.defaults

        # Update defaults with provided keyword arguments
        defaults.update(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self, n_neighbors=70, pbar=None):
        """
        Train the model.

        Args:
            pbar: Shiny progress bar
        """
        # Update attributes if new values are provided
        # self._update_attributes(**kwargs)

        # split data
        self.train_data, self.test_data, self.val_data, self.unlabelled_data = (
            self.split_data()
        )

        # load model
        self._model = self.model(**self.defaults)

        # train
        out = None
        if self.model_type in self._sklearn_models:
            out = self.train_sklearn(rep_path=self.rep_path, pbar=self.pbar)
        elif self.model_type in self._clustering_algs:
            out = self.cluster(
                rep_path=self.rep_path, n_neighbors=n_neighbors, pbar=self.pbar
            )
        elif self.model_type in self._pt_models:
            out = self.train_gp(rep_path=self.rep_path, pbar=self.pbar)
        else:
            raise ValueError(
                f"The training method for '{self.model_type}' models has not been implemented yet"
            )

        return out

    ### Helpers ###
    def split_data(self):
        """
        Split data into train, test, and validation sets.

            1. 'random': Randomly splits data points
            2. 'site': Splits data by sites in the protein,
                such that the same site cannot be found
                in the training, testing and validation set
            3. 'custom': Splits the data according to a custom pattern
            4. 'smart': Stratified split for both classification and regression

        Returns:
            tuple: returns three lists of train, test, and validation proteins.
        """

        proteins = self.library.proteins

        if self.y_type == "class":
            unlabelled_data = [
                prot for prot in proteins if self.library.class_dict[prot.y] == "nan"
            ]
            labelled_data = [
                prot for prot in proteins if self.library.class_dict[prot.y] != "nan"
            ]
        else:
            labelled_data = proteins
            unlabelled_data = []

        if self.seed:
            random.seed(self.seed)

        train_data, test_data, val_data = [], [], []

        if isinstance(self.split, tuple):
            train_ratio, test_ratio, val_ratio = tuple(
                value / sum(self.split) for value in self.split
            )
            train_size = int(train_ratio * len(labelled_data))
            test_size = int(test_ratio * len(labelled_data))

            random.Random(self.seed).shuffle(labelled_data)

            # Split the data
            train_data = labelled_data[:train_size]
            test_data = labelled_data[train_size : train_size + test_size]
            val_data = labelled_data[train_size + test_size :]

        # custom datasplit
        elif isinstance(self.split, dict):
            train_data = self.split["train"]
            test_data = self.split["test"]
            val_data = self.split["val"]

        # use stratified split that works for both regression and classification
        elif self.split == "smart":
            if self.y_type == "class":
                # get the unique classes
                classes = list(set([prot.y for prot in labelled_data]))

                # split the data into classes
                class_data = {
                    c: [prot for prot in labelled_data if prot.y == c] for c in classes
                }

                # split the data into train, test, val
                for c in classes:
                    class_size = len(class_data[c])
                    train_size = int(0.8 * class_size)
                    test_size = int(0.1 * class_size)

                    random.shuffle(class_data[c])

                    train_data += class_data[c][:train_size]
                    test_data += class_data[c][train_size : train_size + test_size]
                    val_data += class_data[c][train_size + test_size :]
            else:
                y_values = np.array([prot.y for prot in labelled_data])

                # Dynamically adjust the number of bins based on dataset size
                n_bins = min(
                    10, max(2, len(labelled_data) // 5)
                )  # At least 2 bins, at most 10
                bins = np.linspace(np.min(y_values), np.max(y_values), n_bins)
                y_binned = np.digitize(y_values, bins) - 1  # Ensure bins start at 0

                # Merge rare bins by combining the smallest bins first
                class_counts = np.bincount(y_binned)
                while np.any(class_counts < 2):
                    if n_bins <= 2:
                        break  # Exit when bins reach the minimum allowed size

                    # Identify the two smallest bins to merge
                    smallest_bins = np.argsort(class_counts)
                    bin_to_merge_1 = smallest_bins[0]
                    bin_to_merge_2 = smallest_bins[1]

                    # Merge the smaller bin into the larger bin
                    y_binned[y_binned == bin_to_merge_2] = bin_to_merge_1

                    # Recompute bin counts
                    unique_bins, y_binned = np.unique(y_binned, return_inverse=True)
                    n_bins = len(unique_bins)
                    class_counts = np.bincount(y_binned)

                # Split data into train/test, then test/val
                try:
                    train_data_idx, temp_data_idx = train_test_split(
                        range(len(labelled_data)),
                        test_size=0.2,
                        stratify=y_binned,
                        random_state=self.seed,
                    )
                    temp_y_binned = y_binned[temp_data_idx]

                    test_data_idx, val_data_idx = train_test_split(
                        temp_data_idx,
                        test_size=0.5,
                        stratify=temp_y_binned,
                        random_state=self.seed,
                    )
                except ValueError:
                    # Fallback to random split if stratification fails
                    train_data_idx, temp_data_idx = train_test_split(
                        range(len(labelled_data)), test_size=0.2, random_state=self.seed
                    )
                    test_data_idx, val_data_idx = train_test_split(
                        temp_data_idx, test_size=0.5, random_state=self.seed
                    )

                # Map indices back to the data
                train_data = [labelled_data[i] for i in train_data_idx]
                test_data = [labelled_data[i] for i in test_data_idx]
                val_data = [labelled_data[i] for i in val_data_idx]

        else:
            raise ValueError(f"The {self.split} split has not been implemented yet...")

        return train_data, test_data, val_data, unlabelled_data

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
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        reps = self.library.load_representations(rep=self.rep, proteins=proteins)

        return reps

    def model(self, **kwargs):
        """
        Load or create model according to user specifications and parameters.

        For a complete list and detailed explanation of model parameters, refer to the scikit-learn documentation.
        """

        model_type = self.model_type
        model = None

        if model_type in self._sklearn_models:
            # model_params = kwargs.copy() # optional to add custom parameters

            if self.y_type == "class":
                if model_type == "rf":
                    model = RandomForestClassifier()
                elif model_type == "svm":
                    model = SVC()
                elif model_type == "knn":
                    model = KNeighborsClassifier()
                elif model_type == "ridge":  # Added support for Ridge Classification
                    model = RidgeClassifier()
            elif self.y_type == "num":
                if model_type == "rf":
                    model = RandomForestRegressor()
                elif model_type == "svm":
                    model = SVR()
                elif model_type == "knn":
                    model = KNeighborsRegressor()
                elif model_type == "ridge":  # Added support for Ridge Regression
                    model = Ridge()

            return model

        elif model_type == "hdbscan":
            model_params = kwargs.copy()
            model = hdbscan.HDBSCAN(
                min_samples=model_params["min_samples"],
                min_cluster_size=model_params["min_cluster_size"],
            )
            return model

        elif model_type in self._pt_models:
            if self.y_type == "class":
                if model_type == "gp":
                    raise ValueError(
                        f"Model type '{model_type}' has not been implemented yet"
                    )
            elif self.y_type == "num":
                if model_type == "gp":
                    model = "GP_MODEL"

            return model
        else:
            raise ValueError(f"Model type '{model_type}' has not been implemented yet")

    def train_sklearn(self, rep_path, pbar=None):
        """
        Train sklearn models and save the model.

        Args:
            rep_path (str): representation path
            pbar: Progress bar for shiny app.
        """
        assert self._model is not None

        if pbar:
            pbar.set(message="Loading representations", detail="...")

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

        if self.library.pred_data:
            self.y_train = [protein.y_pred for protein in self.train_data]
            self.y_test = [protein.y_pred for protein in self.test_data]
            self.y_val = [protein.y_pred for protein in self.val_data]
        else:
            self.y_train = [protein.y for protein in self.train_data]
            self.y_test = [protein.y for protein in self.test_data]
            self.y_val = [protein.y for protein in self.val_data]

        self.test_names = [protein.name for protein in self.test_data]

        # combine train and val if no hyperparameter tuning
        if not self.grid_search:
            x_train = np.concatenate((x_train, x_val), axis=0)
            self.y_train = np.concatenate((self.y_train, self.y_val))
            self.val_names = [protein.name for protein in self.val_data]

        if self.k_folds is None:
            if pbar:
                pbar.set(message="Training {self.model_type}", detail="...")

            # train model
            self._model.fit(x_train, self.y_train)

            # make predictions
            self.test_r2 = self._model.score(x_test, self.y_test)
            self.y_test_pred = self._model.predict(x_test)
            self.y_train_pred = self._model.predict(x_train)

            self.y_train_sigma = [None] * len(self.y_train)
            self.y_test_sigma = [None] * len(self.y_test)

            # conformal prediction and statistics
            self.calibration = self.calibrate(
                self.y_test, self.y_test_pred, confidence=0.90
            )

            self.test_pearson = pearsonr(self.y_test, self.y_test_pred)
            self.test_ken_tau = kendalltau(self.y_test, self.y_test_pred)

            # Add predictions to test proteins
            for i in range(len(test)):
                self.test_data[i].y_pred = self.y_test_pred[i]
                self.test_data[i].y_sigma = self.y_test_sigma[i]

            # Save dataframes
            if self.dest is not None:
                csv_dest = f"{self.dest}"
            else:
                csv_dest = os.path.join(
                    f"{self.library.rep_path}",
                    f"../models/{self.model_type}/{self.rep}",
                )

            if not os.path.exists(csv_dest):
                os.makedirs(csv_dest, exist_ok=True)

            train_df = self.save_to_csv(
                self.train_data,
                self.y_train,
                self.y_train_pred,
                self.y_train_sigma,
                f"{csv_dest}/train_data.csv",
            )

            test_df = self.save_to_csv(
                self.test_data,
                self.y_test,
                self.y_test_pred,
                self.y_test_sigma,
                f"{csv_dest}/test_data.csv",
            )

            # add split information to df
            train_df["split"] = "train"
            test_df["split"] = "test"

            if self.grid_search:
                self.val_r2 = self._model.score(x_val, self.y_val)
                self.y_val_pred = self._model.predict(x_val)
                self.y_val_sigma = [None] * len(self.y_val)
                self.calibration_ratio, within_calibration = self._within_calibration(
                    self.y_val_pred, self.y_val
                )
                self.val_pearson = pearsonr(self.y_val, self.y_val_pred)
                self.val_ken_tau = kendalltau(self.y_val, self.y_val_pred)
                val_df = self.save_to_csv(
                    self.val_data,
                    self.y_val,
                    self.y_val_pred,
                    self.y_val_sigma,
                    f"{csv_dest}/val_data.csv",
                )
                val_df["split"] = "val"

                # Concatenate the DataFrames
                self.out_df = pd.concat(
                    [train_df, test_df, val_df], axis=0
                ).reset_index(drop=True)

                self.y_best = max(
                    (max(self.y_train), max(self.y_test), max(self.y_val))
                )
            else:
                # Concatenate the DataFrames
                self.out_df = pd.concat([train_df, test_df], axis=0).reset_index(
                    drop=True
                )
                self.y_best = max((max(self.y_train), max(self.y_test)))

        # handle ensembles
        else:
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            fold_results = []
            ensemble = []
            calibrations = []
            for i, data in enumerate(kf.split(x_train)):
                train_index, val_index = data

                if pbar:
                    pbar.set(
                        message=f"Training {self.model_type} {i+1}/{self.k_folds}",
                        detail="...",
                    )

                x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_val_fold = (
                    np.array(self.y_train)[train_index],
                    np.array(self.y_train)[val_index],
                )

                self._model = self.model()
                self._model.fit(x_train_fold, y_train_fold)
                val_r2 = self._model.score(x_val_fold, y_val_fold)
                fold_results.append(val_r2)
                ensemble.append(self._model)

                # conformal prediction
                calibration = self.calibrate(
                    y_val_fold, self._model.predict(x_val_fold), confidence=0.90
                )
                calibrations.append(calibration)

            # avg_test_r2 = np.mean(fold_results)
            # avg_calibration_ratio = np.mean([c for c in calibrations])

            # Store model ensemble as model
            self._model = ensemble

            # Prediction on validation set
            predictions = self.predict(self.train_data)
            self.train_data = predictions["pred_proteins"]
            self.y_train_pred = predictions["y_pred"]
            self.y_train_sigma = predictions["y_sigma"]
            self.y_train = predictions["y_true"]

            # Prediction on test set
            predictions = self.predict(self.test_data)
            self.test_data = predictions["pred_proteins"]
            self.y_test_pred = predictions["y_pred"]
            self.y_test_sigma = predictions["y_sigma"]
            self.y_test = predictions["y_true"]

            # Compute R-squared on test dataset
            self.test_r2 = self.score(self.test_data)

            # conformal prediction and statistics
            self.calibration = self.calibrate(
                self.y_test, self.y_test_pred, confidence=0.90
            )
            self.calibration_ratio, within_calibration = self._within_calibration(
                self.y_test_pred, self.y_test
            )

            # Prediction unlabelled data if exists
            if len(self.unlabelled_data) > 0:
                predictions = self.predict(self.unlabelled_data)
                self.y_unlabelled_pred = predictions["y_pred"]
                self.y_unlabelled_sigma = predictions["y_sigma"]
                self.y_unlabelled = predictions["y_true"]

            self.test_pearson = pearsonr(self.y_test, self.y_test_pred)
            self.test_ken_tau = kendalltau(self.y_test, self.y_test_pred)

            # Save the model
            if self.dest is not None:
                csv_dest = f"{self.dest}"
            else:
                csv_dest = os.path.join(
                    f"{self.library.rep_path}",
                    f"../models/{self.model_type}/{self.rep}",
                )

            if not os.path.exists(csv_dest):
                os.makedirs(csv_dest, exist_ok=True)

            # Save the sequences, y-values, and predicted y-values to CSV
            train_df = self.save_to_csv(
                self.train_data,
                self.y_train,
                self.y_train_pred,
                self.y_train_sigma,
                f"{csv_dest}/train_data.csv",
            )
            # Save the sequences, y-values, and predicted y-values to CSV
            test_df = self.save_to_csv(
                self.test_data,
                self.y_test,
                self.y_test_pred,
                self.y_test_sigma,
                f"{csv_dest}/test_data.csv",
            )

            # save unlabelled data if exists
            if len(self.unlabelled_data) > 0:
                unlabelled_df = self.save_to_csv(
                    self.unlabelled_data,
                    self.y_unlabelled,
                    self.y_unlabelled_pred,
                    self.y_unlabelled_sigma,
                    f"{csv_dest}/unlabelled_data.csv",
                )

            if self.grid_search:
                predictions = self.predict(self.val_data)
                self.test_data = predictions["pred_proteins"]
                self.y_val_pred = predictions["y_pred"]
                self.y_val_sigma = predictions["y_sigma"]
                self.y_val = predictions["y_true"]
                # Compute R-squared on validataion dataset
                self.val_r2 = self.score(self.val_data)
                self.val_pearson = pearsonr(self.y_val, self.y_val_pred)
                self.val_ken_tau = kendalltau(self.y_val, self.y_val_pred)
                val_df = self.save_to_csv(
                    self.val_data,
                    self.y_val,
                    self.y_val_pred,
                    self.y_val_sigma,
                    f"{csv_dest}/val_data.csv",
                )
                train_df["split"] = "train"
                test_df["split"] = "test"
                val_df["split"] = "val"
                if len(self.unlabelled_data) > 0:
                    unlabelled_df["split"] = "unlabelled"
                    comb_df = [train_df, test_df, unlabelled_df]
                else:
                    comb_df = [train_df, test_df, val_df]
                self.y_best = max(
                    (max(self.y_train), max(self.y_test), max(self.y_val))
                )
            else:
                # add split information to df
                train_df["split"] = "train"
                test_df["split"] = "test"
                if len(self.unlabelled_data) > 0:
                    unlabelled_df["split"] = "unlabelled"
                    comb_df = [train_df, test_df, unlabelled_df]
                else:
                    comb_df = [train_df, test_df]
                # TODO: that depends on minimization or maximization goal
                self.y_best = max((max(self.y_train), max(self.y_test)))

            # Concatenate the DataFrames
            self.out_df = pd.concat(comb_df, axis=0).reset_index(drop=True)

            self.library.y_pred = [prot.y_pred for prot in self.library.proteins]

        out = {
            "df": self.out_df,
            "rep_path": self.library.rep_path,
            "struc_path": self.library.struc_path,
            "y_type": self.library.y_type,
            "y_col": "y_true",
            "y_pred_col": "y_predicted",
            "y_sigma_col": "y_sigma",
            "seqs_col": "sequence",
            "names_col": "name",
            "reps": self.library.reps,
            "class_dict": self.library.class_dict,
            "dr_df": None,
        }

        print(
            f"Training completed:\ntest_r2:\t{self.test_r2}\ntest_pearson:\t{self.test_pearson}"
        )

        return out

    def train_gp(
        self,
        rep_path,
        epochs=150,
        initial_lr=0.1,
        final_lr=1e-6,
        decay_rate=0.1,
        pbar=None,
    ):
        """
        Train a Gaussian Process model and save the model.

        Args:
            rep_path (str): representation path
            pbar: Progress bar for shiny app.
        """

        assert self._model is not None

        if pbar:
            pbar.set(message="Loading representations", detail="...")

        # This is for representations that are not stored in memory
        train = self.load_representations(self.train_data, rep_path=rep_path)
        test = self.load_representations(self.test_data, rep_path=rep_path)
        val = self.load_representations(self.val_data, rep_path=rep_path)

        # handle representations that are not esm
        if len(train[0].shape) == 2:
            train = [x.view(-1) for x in train]
            test = [x.view(-1) for x in test]
            val = [x.view(-1) for x in val]

        x_train = torch.stack(train).to(device=self.device)
        x_test = torch.stack(test).to(device=self.device)
        x_val = torch.stack(val).to(device=self.device)

        if self.library.pred_data:
            self.y_train = (
                torch.stack(
                    [torch.Tensor([protein.y_pred]) for protein in self.train_data]
                )
                .view(-1)
                .to(device=self.device)
            )
            self.y_test = (
                torch.stack(
                    [torch.Tensor([protein.y_pred]) for protein in self.test_data]
                )
                .view(-1)
                .to(device=self.device)
            )
            y_val = (
                torch.stack(
                    [torch.Tensor([protein.y_pred]) for protein in self.val_data]
                )
                .view(-1)
                .to(device=self.device)
            )
        else:
            self.y_train = (
                torch.stack([torch.Tensor([protein.y]) for protein in self.train_data])
                .view(-1)
                .to(device=self.device)
            )
            self.y_test = (
                torch.stack([torch.Tensor([protein.y]) for protein in self.test_data])
                .view(-1)
                .to(device=self.device)
            )
            y_val = (
                torch.stack([torch.Tensor([protein.y]) for protein in self.val_data])
                .view(-1)
                .to(device=self.device)
            )

        self.test_names = [protein.name for protein in self.test_data]

        if self.grid_search:
            x_train = torch.cat((x_train, x_val), dim=0)
            self.y_train = torch.cat((self.y_train, y_val))
            self.val_names = [protein.name for protein in self.val_data]

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self.device
        )
        self._model = GP(x_train, self.y_train, self.likelihood).to(device=self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._model)

        self._model.train()
        self.likelihood.train()
        prev_loss = float("inf")

        if pbar:
            pbar.set(message=f"Training {self.model_type}", detail="...")

        for _ in range(epochs):
            optimizer.zero_grad()
            output = self._model(x_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Check for convergence
            if abs(prev_loss - loss.item()) < 0.0001:
                #    print(f'Convergence reached. Stopping training...')
                break

            prev_loss = loss.item()

        print(f"Training completed. Final loss: {loss.item()}")

        # prediction on train set
        y_train_pred, y_train_sigma = predict_gp(self._model, self.likelihood, x_train)
        self.y_train_pred, self.y_train_sigma = (
            y_train_pred.cpu().numpy(),
            y_train_sigma.cpu().numpy(),
        )

        # prediction on test set
        y_test_pred, y_test_sigma = predict_gp(self._model, self.likelihood, x_test)
        self.test_r2 = computeR2(self.y_test, y_test_pred)
        self.y_test_pred, self.y_test_sigma = (
            y_test_pred.cpu().numpy(),
            y_test_sigma.cpu().numpy(),
        )
        self.y_train = self.y_train.cpu().numpy()
        self.y_test = self.y_test.cpu().numpy()
        self.test_pearson = pearsonr(self.y_test, self.y_test_pred)
        self.test_ken_tau = kendalltau(self.y_test, self.y_test_pred)
        self.y_best = max((max(self.y_train), max(self.y_test)))

        # conformal prediction and statistics
        self.calibration = self.calibrate(
            self.y_test, self.y_test_pred, confidence=0.90
        )

        # Add predictions to proteins
        for i in range(len(train)):
            self.train_data[i].y_pred = self.y_train_pred[i].item()
            self.train_data[i].y_sigma = self.y_train_sigma[i].item()

        # Add predictions to test proteins
        for i in range(len(test)):
            self.test_data[i].y_pred = self.y_test_pred[i].item()
            self.test_data[i].y_sigma = self.y_test_sigma[i].item()

        # save dataframes
        if self.dest is not None:
            csv_dest = f"{self.dest}"
        else:
            csv_dest = os.path.join(
                f"{self.library.rep_path}",
                f"../models/{self.model_type}/{self.rep}",
            )

        if not os.path.exists(csv_dest):
            os.makedirs(csv_dest, exist_ok=True)

        train_df = self.save_to_csv(
            self.train_data,
            self.y_train,
            self.y_train_pred,
            self.y_train_sigma,
            f"{csv_dest}/train_data.csv",
        )
        test_df = self.save_to_csv(
            self.test_data,
            self.y_test,
            self.y_test_pred,
            self.y_test_sigma,
            f"{csv_dest}/test_data.csv",
        )

        # add split information to df
        train_df["split"] = "train"
        test_df["split"] = "test"

        if self.grid_search:
            # prediction on validation set
            y_val_pred, y_val_sigma = predict_gp(self._model, self.likelihood, x_val)
            self.val_r2 = computeR2(y_val, y_val_pred)
            self.y_train = self.y_train.cpu().numpy()

            self.y_val_pred, self.y_val_sigma = (
                y_val_pred.cpu().numpy(),
                y_val_sigma.cpu().numpy(),
            )

            self.y_val = y_val.cpu().numpy()
            self.y_test = self.y_test.cpu().numpy()

            self.calibration_ratio, within_calibration = self._within_calibration(
                self.y_val_pred, self.y_val
            )
            self.val_pearson = pearsonr(self.y_val, self.y_val_pred)
            self.val_ken_tau = kendalltau(self.y_val, self.y_val_pred)

            self.y_best = max((max(self.y_train), max(self.y_test), max(self.y_val)))
            self.library.y_pred = [prot.y_pred for prot in self.library.proteins]

            # Add predictions to val proteins
            for i in range(len(val)):
                self.val_data[i].y_pred = self.y_val_pred[i].item()
                self.val_data[i].y_pred = self.y_val_sigma[i].item()

            val_df = self.save_to_csv(
                self.val_data,
                self.y_val,
                self.y_val_pred,
                self.y_val_sigma,
                f"{csv_dest}/val_data.csv",
            )
            val_df["split"] = "val"
            # Concatenate the DataFrames
            self.out_df = pd.concat([train_df, test_df, val_df], axis=0).reset_index(
                drop=True
            )
        else:
            # Concatenate the DataFrames
            self.out_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

        out = {
            "df": self.out_df,
            "rep_path": self.library.rep_path,
            "struc_path": self.library.struc_path,
            "y_type": self.library.y_type,
            "y_col": "y_true",
            "y_pred_col": "y_predicted",
            "y_sigma_col": "y_sigma",
            "seqs_col": "sequence",
            "names_col": "name",
            "reps": self.library.reps,
            "class_dict": self.library.class_dict,
            "dr_df": None,
        }

        return out

    # calibration for conformal predictions
    def calibrate(self, y_cal, y_cal_pred, confidence=0.90):
        """
        Calibrate the model using conformal predictions.

        Args:
            y_cal (list): List of true y-values.
            y_cal_pred (list): List of predicted y-values.
            confidence (float): Confidence level for the prediction interval.

        Returns:
            width: The width of the prediction interval.
        """
        residuals = y_cal - y_cal_pred
        conformity_scores = np.sort(np.abs(residuals))[::-1]
        quantile_index = int((1 - confidence) * (len(conformity_scores) + 1)) - 1
        width = conformity_scores[quantile_index]
        return width

    def _within_calibration(self, y_pred=None, y_true=None):
        """
        Compute ratio and determin points within calibration

        Args:
            y_pred (list): List of predicted y-values.
            y_true (list): List of true y-values.
        Returns:
            tuple: Tuple of ratio and points within calibration
        """
        if y_pred is None:
            y_pred = self.y_test_pred
        if y_true is None:
            y_true = self.y_test

        residuals = np.abs(y_pred - y_true)
        within_cal = np.where(residuals <= self.calibration, 1, 0)
        ratio = np.sum(within_cal) / len(y_true)

        return ratio, within_cal

    # Save the sequences, y-values, and predicted y-values to CSV
    def save_to_csv(
        self,
        proteins,
        y_values,
        y_pred_values,
        y_sigma_values,
        filename,
        acq_scores=None,
    ):
        # Prepare data for CSV and DataFrame
        data = []
        names = [prot.name for prot in proteins]

        # Determine if acquisition scores are provided
        if acq_scores is not None:
            header = [
                "name",
                "sequence",
                "y_value",
                "y_predicted",
                "y_sigma",
                "acq_score",
            ]  # CSV header with acq_scores
        else:
            header = [
                "name",
                "sequence",
                "y_value",
                "y_predicted",
                "y_sigma",
            ]  # CSV header without acq_scores

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i, (name, protein, y, y_pred, y_sigma) in enumerate(
                zip(names, proteins, y_values, y_pred_values, y_sigma_values)
            ):
                if acq_scores is not None:
                    row = [name, protein.seq, y, y_pred, y_sigma, acq_scores[i]]
                else:
                    row = [name, protein.seq, y, y_pred, y_sigma]
                writer.writerow(row)
                data.append(row)

        # Create a DataFrame from the collected data
        if acq_scores is not None:
            df = pd.DataFrame(
                data,
                columns=[
                    "name",
                    "sequence",
                    "y_true",
                    "y_predicted",
                    "y_sigma",
                    "acq_score",
                ],
            )
        else:
            df = pd.DataFrame(
                data, columns=["name", "sequence", "y_true", "y_predicted", "y_sigma"]
            )

        return df

    def predict(self, proteins, rep_path=None, acq_fn="greedy", batch_size=10000):
        """
        Makes predictions using the trained model, either for a list of proteins, sequence(s), or a Library object.

        Args:
            proteins (list): List of proteins, sequence(s), or a Library object.
            rep_path (str): Path to representations for proteins in the list.
                If None, the library project path and representation type for training
                will be assumed

        Returns:
            list: Predictions generated by the model.
        """
        if self._model is None:
            raise ValueError("Model is 'None'")

        if acq_fn == "ei":
            acq = BO.EI
        elif acq_fn == "greedy":
            acq = BO.greedy
        elif acq_fn == "ucb":
            acq = BO.UCB
        elif acq_fn == "random":
            acq = BO.random_acquisition

        all_y_pred = []
        all_sigma_pred = []
        all_acq_scores = []

        # pred_dest
        # save dataframes
        if self.dest is not None:
            pred_dest = f"{self.dest}"
        else:
            pred_dest = os.path.join(
                f"{self.library.rep_path}",
                f"../models/{self.model_type}/{self.rep}/predictions",
            )

        pred_rep_path = os.path.join(pred_dest, "rep")

        if not os.path.exists(pred_dest):
            os.makedirs(pred_dest, exist_ok=True)

        if not os.path.exists(pred_rep_path):
            os.makedirs(pred_rep_path, exist_ok=True)

        # if proteins is a sequence string
        if isinstance(proteins, str):
            proteins = [proteins]

        # if proteins is a single Protein object
        if isinstance(proteins, Protein):
            proteins = [proteins]

        # if proteins is a list of strings
        if isinstance(proteins[0], str):
            proteins = [
                Protein(
                    hashlib.sha256(seq.encode()).hexdigest()[:16],
                    seq,
                    y=None,
                    y_pred=None,
                    y_sigma=None,
                    acq_score=None,
                    user=self.library.user,
                    rep_path=pred_rep_path,
                )
                for seq in proteins
            ]

        # if proteins is a list of Protein objects
        elif isinstance(proteins[0], Protein):
            proteins = [prot for prot in proteins]

        # if proteins is a Library object
        elif isinstance(proteins, Library):
            proteins = proteins.proteins

        for i in range(0, len(proteins), batch_size):
            batch_proteins = proteins[i : i + batch_size]
            batch_reps = self.load_representations(batch_proteins, rep_path)

            if len(batch_reps[0].shape) == 2:
                batch_reps = [x.view(-1) for x in batch_reps]

            # GP
            if self.model_type == "gp":
                self.likelihood.eval()
                x = torch.stack(batch_reps).to(device=self.device)
                y_pred, sigma_pred = predict_gp(self._model, self.likelihood, x)
                y_pred = y_pred.cpu().numpy()
                sigma_pred = sigma_pred.cpu().numpy()
                acq_score = acq(y_pred, sigma_pred, self.y_best)

            # Handle ensembles
            elif isinstance(self._model, list):
                ys = []
                for model in self._model:
                    x = torch.stack(batch_reps).cpu().numpy()
                    y_pred = model.predict(x)
                    ys.append(y_pred)

                y_stack = np.stack(ys)
                y_pred = np.mean(y_stack, axis=0)
                sigma_pred = np.std(y_stack, axis=0)
                acq_score = acq(y_pred, sigma_pred, self.y_best)

            # Handle single model
            else:
                x = torch.stack(batch_reps).cpu().numpy()
                y_pred = self._model.predict(x)
                sigma_pred = np.zeros_like(y_pred)
                acq_score = acq(y_pred, sigma_pred, self.y_best)

            all_y_pred.extend(y_pred)
            all_sigma_pred.extend(sigma_pred)
            all_acq_scores.extend(acq_score)

        all_y_pred = np.array(all_y_pred)
        all_sigma_pred = np.array(all_sigma_pred)
        all_acq_scores = np.array(all_acq_scores)

        # Sort acquisition scores and get sorted indices
        sorted_indices = np.argsort(all_acq_scores)[::-1]

        # Sort all lists/arrays by the sorted indices
        pred_proteins = [proteins[i] for i in sorted_indices]
        y_val = [prot.y for prot in pred_proteins]
        y_val_pred = all_y_pred[sorted_indices]
        y_val_sigma = all_sigma_pred[sorted_indices]
        sorted_acq_score = all_acq_scores[sorted_indices]

        for i, prot in enumerate(pred_proteins):
            prot.y_pred = y_val_pred[i]
            prot.y_sigma = y_val_sigma[i]

        # Prediction dataframe
        pred_df = pd.DataFrame(
            {
                "name": [prot.name for prot in pred_proteins],
                "sequence": [prot.seq for prot in pred_proteins],
                "y_true": y_val,
                "y_predicted": y_val_pred,
                "y_sigma": y_val_sigma,
                "acq_score": sorted_acq_score,
            }
        )
        # Save dataframes

        out = {
            "df": pred_df,
            "rep_path": pred_rep_path,
            "struc_path": self.library.struc_path,
            "y_type": self.library.y_type,
            "y_col": "y_true",
            "y_pred_col": "y_predicted",
            "y_sigma_col": "y_sigma",
            "seqs_col": "sequence",
            "acq_col": "acq_score",
            "names_col": "name",
            "reps": self.library.reps,
            "class_dict": self.library.class_dict,
            "dr_df": None,
            "pred_proteins": pred_proteins,
            "y_pred": y_val_pred,
            "y_sigma": y_val_sigma,
            "y_true": y_val,
            "acq_score": sorted_acq_score,
        }

        return out

    def score(self, proteins: list, rep_path=None):
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
            raise ValueError("Model is 'None'")

        reps = self.load_representations(proteins, rep_path)

        if len(reps[0].shape) == 2:
            reps = [x.view(-1) for x in reps]

        x = torch.stack(reps).cpu().numpy()
        y = [protein.y for protein in proteins]

        # ensemble
        ensemble_scores = []
        if isinstance(self._model, list):
            for model in self._model:
                score = model.score(x, y)
                ensemble_scores.append(score)
            ensemble_scores = np.stack(ensemble_scores)
            scores = np.mean(ensemble_scores, axis=0)
        else:
            scores = self._model.score(x, y)

        return scores

    def true_vs_predicted(
        self,
        y_true: Union[list, None] = None,
        y_pred: Union[list, None] = None,
        title: Union[str, None] = None,
        x_label: Union[str, None] = None,
        y_label: Union[str, None] = None,
        plot_grid: bool = True,
        file: Union[str, None] = None,
        show_plot: bool = False,
    ):
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

        if y_true is None:
            y_true = self.y_test
        if y_pred is None:
            y_pred = self.y_test_pred

        if self.dest:
            dest = os.path.join(self.dest, "plots")
        else:
            dest = os.path.join(
                self.library.rep_path, f"../models/{self.model_type}/{self.rep}/plots"
            )
        if not os.path.exists(dest):
            os.makedirs(dest)

        fig, ax = vis.plot_predictions_vs_groundtruth(
            y_true,
            y_pred,
            title,
            x_label,
            y_label,
            plot_grid,
            file,
            show_plot,
            self.calibration,
        )

        return fig, ax

    def predicted_vs_true(
        self,
        y_true: Union[list, None] = None,
        y_pred: Union[list, None] = None,
        title: Union[str, None] = None,
        x_label: Union[str, None] = None,
        y_label: Union[str, None] = None,
        plot_grid: bool = True,
        file: Union[str, None] = None,
        show_plot: bool = False,
    ):
        """
        Calls true_vs_predicted method to plot true vs predicted values. (I often get confused with the names)
        """
        out = self.true_vs_predicted(
            y_true=y_true,
            y_pred=y_pred,
            title=title,
            x_label=x_label,
            y_label=y_label,
            plot_grid=plot_grid,
            file=file,
            show_plot=show_plot,
        )

        return out

    def cluster(self, rep_path, n_neighbors=70, pbar=None):
        """
        Clustering algorithm using UMAP.

        Args:
            rep_path (str): representation path.
            n_neighbors (int): number of neighbors. Default 70.
            pbar: Progress bar for shiny app.
        """
        assert self._model is not None

        if pbar:
            pbar.set(message="Loading representations", detail="...")

        reps = self.load_representations(self.library.proteins, rep_path=rep_path)

        # handle representations that are not esm
        if len(reps[0].shape) == 2:
            reps = [x.view(-1) for x in reps]

        x_reps = torch.stack(reps).cpu().numpy()

        # do UMAP
        if self.dr_method == "umap":
            clusterable_embedding = umap.UMAP(
                n_neighbors=70, min_dist=0.0, n_components=2, random_state=self.seed
            ).fit_transform(x_reps)
            dr_df = pd.DataFrame(clusterable_embedding, columns=["z1", "z2"])
            dr_df["names"] = [protein.name for protein in self.library.proteins]
        elif self.dr_method == "pca":
            pca = PCA(n_components=2)
            clusterable_embedding = pca.fit_transform(x_reps)
            dr_df = pd.DataFrame(clusterable_embedding, columns=["z1", "z2"])
            dr_df["names"] = [protein.name for protein in self.library.proteins]
        elif self.dr_method == "tsne":
            tsne = TSNE(n_components=2, verbose=6, random_state=self.seed)
            clusterable_embedding = tsne.fit_transform(x_reps)
            dr_df = pd.DataFrame(clusterable_embedding, columns=["z1", "z2"])
            dr_df["names"] = [protein.name for protein in self.library.proteins]

        # perform clustering
        if self._model_type == "hdbscan":
            labels = self._model.fit_predict(clusterable_embedding)

        elif self._model_type in self._sklearn_models:
            self._model.fit(clusterable_embedding)
            labels = self._model.labels_

        # store prediction results in protein
        y_trues = []
        for i, prot in enumerate(self.library.proteins):
            self.library.proteins[i].y_pred = labels[i]
            y_true = prot.y
            y_trues.append(y_true)

        self.library.y_pred = labels

        if self.dest is not None:
            csv_dest = f"{self.dest}"
        else:
            csv_dest = os.path.join(
                f"{self.library.rep_path}", f"../models/{self.model_type}/{self.rep}"
            )

        if not os.path.exists(csv_dest):
            os.makedirs(csv_dest, exist_ok=True)

        # create out dataframe
        out_df = self.save_to_csv(
            self.library.proteins,
            y_trues,
            labels,
            [None] * len(self.library.proteins),
            f"{csv_dest}/clustering.csv",
        )

        self.out_df = out_df

        out = {
            "df": self.out_df,
            "rep_path": self.library.rep_path,
            "struc_path": self.library.struc_path,
            "y_type": self.library.y_type,
            "y_col": "y_true",
            "y_pred_col": "y_predicted",
            "y_sigma_col": "y_sigma",
            "seqs_col": "sequence",
            "names_col": "name",
            "reps": self.library.reps,
            "class_dict": self.library.class_dict,
            "dr_df": dr_df,
        }

        return out

    def search(
        self,
        N=10,
        labels=[],
        optim_problem="max",
        method="ga",
        max_eval=10000,
        explore=0.1,
        batch_size=100,
        pbar=None,
        acq_fn="ei",
        overwrite=True,
    ):
        """Search for new mutants or select variants from a set of sequences"""

        if self.y_type == "class":
            out = self._class_search(
                N=N, labels=labels, method=method, max_eval=max_eval, pbar=pbar
            )
        elif self.y_type == "num":
            out = self._num_search(
                method=method,
                optim_problem=optim_problem,
                max_eval=max_eval,
                explore=explore,
                batch_size=batch_size,
                pbar=pbar,
                acq_fn=acq_fn,
                overwrite=overwrite,
            )

        return out

    def _class_search(
        self,
        N=10,
        optim_problem="max",
        labels=[],
        method="ga",
        max_eval=10000,
        pbar=None,
    ):
        """
        Sample diverse sequences and return a mask with 1 for selected indices and 0 for non-selected.

        Args:
            N (int): Number of sequences to be returned.
            optim_problem (float): Minimization or maximization of y-values. Default 'max', alternatively 'min'.
            labels (list): list of labels to sample from. Default [] will sample from all labels.
            method (str): Method used for sampling. Default 'ga' - Genetic Algorithm.
            max_eval (int): Maximum number of evaluations. Default 1000.
            pbar: Progress bar for ProteusAI app.
        """

        class_dict = self.library.class_dict
        full_proteins = self.library.proteins  # Full list of proteins

        if len(labels) < 1 or labels == ["all"]:
            if self.model_type in self._clustering_algs:
                labels = list(set([prot.y_pred for prot in full_proteins]))
            else:
                labels = list(class_dict.keys())
                labels = [class_dict[label] for label in labels]
            proteins = full_proteins
            full_indices = list(range(len(full_proteins)))  # Indices for all proteins
        else:
            # Filter proteins by label and keep track of their original indices
            proteins, full_indices = zip(
                *[
                    (prot, idx)
                    for idx, prot in enumerate(full_proteins)
                    if int(prot.y_pred) in labels
                ]
            )
            proteins = list(proteins)
            full_indices = list(full_indices)

        vectors = self.load_representations(proteins, rep_path=self.library.rep_path)

        if pbar:
            pbar.set(message=f"Searching {N} diverse sequences", detail="...")

        selected_indices, diversity = BO.simulated_annealing(vectors, N, pbar=pbar)

        # Map selected_indices back to full_protein list using full_indices
        full_selected_indices = [full_indices[i] for i in selected_indices]

        # Create a mask for the full protein list
        mask = np.zeros(len(full_proteins), dtype=int)

        # Set the selected indices in the full list to 1
        mask[full_selected_indices] = 1

        # Select the proteins based on the full_selected_indices
        selected_proteins = [full_proteins[i] for i in full_selected_indices]
        ys = [prot.y for prot in selected_proteins]
        y_pred = [prot.y_pred for prot in selected_proteins]
        y_sigma = [prot.y_sigma for prot in selected_proteins]

        # Save the search results
        if self.dest is not None:
            csv_dest = self.dest
        else:
            csv_dest = os.path.join(
                f"{self.library.rep_path}", f"../models/{self.model_type}/{self.rep}"
            )

        self.search_df = self.save_to_csv(
            selected_proteins, ys, y_pred, y_sigma, f"{csv_dest}/search_results.csv"
        )

        out = {
            "df": self.search_df,
            "rep_path": self.library.rep_path,
            "struc_path": self.library.struc_path,
            "y_type": self.library.y_type,
            "y_col": "y_true",
            "y_pred_col": "y_predicted",
            "y_sigma_col": "y_sigma",
            "seqs_col": "sequence",
            "names_col": "name",
            "reps": self.library.reps,
            "class_dict": self.library.class_dict,
            "mask": mask,
            "dr_df": None,
        }

        return out

    def _num_search(
        self,
        optim_problem="max",
        method="ga",
        max_eval=10000,
        explore=0.1,
        batch_size=100,
        acq_fn="ei",
        overwrite=True,
        pbar=None,
    ):
        """
        Search for improved mutants.

        Args:
            optim_problem (str): Minimization or maximization of y-values. Default 'max', alternatively 'min'.
            method (str): Method used for sampling. Default 'ga' - Genetic Algorithm.
            max_eval (int): Maximum number of evaluations. Default 1000.
            explore (float): Exploration ratio, float between 0 and 1 to control the exploratory tendency of the sampling algorithm.
            batch_size (int): Batch size for loading representations. Default 100.
            acq_fn (str): Acquisition function. Default 'ei' - Expected Improvement.
            overwrite (bool): Overwrite existing search results. Default True.
            pbar: Progress bar for ProteusAI app.
        """
        if pbar:
            pbar.set(message=f"Evaluation {max_eval} sequences", detail="...")

        # Sort proteins based on the optimization problem
        if optim_problem == "max":
            proteins = sorted(
                self.library.proteins, key=lambda prot: prot.y, reverse=True
            )
        elif optim_problem == "min":
            proteins = sorted(
                self.library.proteins, key=lambda prot: prot.y, reverse=False
            )
        else:
            raise ValueError(f"'{optim_problem}' is an invalid optimization problem")

        # Extract y values and compute the mean
        ys = [prot.y for prot in proteins]
        mean_y = np.mean(ys)

        # Get the sequences of the top N proteins that have y > mean_y or y < mean_y based on the optimization problem
        if optim_problem == "max":
            improved_seqs = [prot.seq for prot in proteins if prot.y > mean_y]
        elif optim_problem == "min":
            improved_seqs = [prot.seq for prot in proteins if prot.y < mean_y]

        # Introduce random mutations from the mutations dictionary
        seq_lens = set([len(seq) for seq in improved_seqs])
        if len(seq_lens) > 1:
            # allow all amin acids for each position
            aa_list = [
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
            ]
            mutations = {i: aa_list for i in range(1, max(seq_lens))}
        else:
            mutations = BO.find_mutations(improved_seqs)

        # Save destination for search_results
        if self.dest is not None:
            csv_dest = self.dest
        else:
            csv_dest = os.path.join(
                f"{self.library.rep_path}",
                f"../models/{self.model_type}/{self.rep}/predictions",
            )
            os.makedirs(csv_dest, exist_ok=True)

        csv_file = os.path.join(
            csv_dest, f"{self.model_type}_{self.rep}_predictions.csv"
        )

        # results file name
        fname = f"{csv_dest}/{self.model_type}_{self.rep}.csv"

        if os.path.exists(os.path.join(csv_dest, fname)) and not overwrite:
            self.search_df = pd.read_csv(os.path.join(csv_dest, fname))
        else:
            self.search_df = None

        mutant_df = self._mutate(
            proteins, mutations, explore=explore, max_eval=max_eval
        )

        out = {
            "df": mutant_df,
            "rep_path": self.library.rep_path,
            "struc_path": self.library.struc_path,
            "y_type": self.library.y_type,
            "seqs_col": "sequence",
            "y_col": "y_true",
            "y_pred_col": "y_predicted",
            "y_sigma_col": "y_sigma",
            "acq_col": "acq_score",
            "names_col": "name",
            "reps": self.library.reps,
            "class_dict": self.library.class_dict,
            "dr_df": None,
        }

        library = Library(user=self.library.user, source=out)

        if self.rep not in self._in_memory_representations:
            library.compute(method=self.rep, pbar=pbar, batch_size=batch_size)

        predictions = self.predict(library.proteins, rep_path=library.rep_path)
        val_data = predictions["pred_proteins"]
        y_pred = predictions["y_pred"]
        y_sigma = predictions["y_sigma"]
        y_val = predictions["y_true"]
        acq_score = predictions["acq_score"]

        self.search_df = self.save_to_csv(
            val_data, y_val, y_pred, y_sigma, csv_file, acq_scores=acq_score
        )

        return self.search_df

    def _mutate(self, proteins, mutations, explore=0.1, max_eval=100):
        """
        Propose new mutations

        Args:
            proteins (list): list of proteins
            mutations (dict): dictionary of positions and mutations. Index start at 1
                example: {15:['A', 'L', 'I']}
            exploration (float): Exploration ratio, float between 0 and 1 to control
                the exploratory tendency of the sampling algorithm.
            max_eval (int): maximum number of evaluations before termination.

        Returns:
            pandas dataframe
        """

        if self.search_df is not None and not self.search_df.empty:
            mutated_seqs = self.search_df.sequence.to_list()
            mutated_names = self.search_df.name.to_list()
            y_trues = [None] * len(mutated_seqs)
            y_preds = self.search_df.y_predicted.to_list()
            y_sigmas = self.search_df.y_sigma.to_list()
            acq_scores = self.search_df.acq_score.to_list()

        else:
            mutated_seqs = []
            mutated_names = []
            y_trues = []
            y_preds = []
            y_sigmas = []
            acq_scores = []

        for _ in range(max_eval):
            prot = random.choice(proteins)
            name = prot.name
            seq_list = list(prot.seq)

            if random.random() < explore:
                # Explore: random position and random mutation
                pos = random.randint(0, len(seq_list) - 1)
                mut = random.choice("ACDEFGHIKLMNPQRSTVWY")
                mutated_name = name + f"+{seq_list[pos]}{pos+1}{mut}"
            else:
                # Exploit: use known mutation from the provided mutations dictionary
                try:
                    pos, mut_list = random.choice(list(mutations.items()))
                    pos = pos - 1
                    if pos < len(seq_list):
                        mut = random.choice(mut_list)
                    mutated_name = (
                        name + f"+{seq_list[pos]}{pos+1}{mut}"
                    )  # list is indexed at 0 but mutation descriptions at 1
                except Exception:
                    pos = random.randint(0, len(seq_list) - 1)
                    mut = random.choice("ACDEFGHIKLMNPQRSTVWY")
                    mutated_name = name + f"+{seq_list[pos]}{pos+1}{mut}"

            if (
                seq_list[pos] != mut and mutated_name not in mutated_names
            ):  # Exclude mutations to the same residue
                seq_list[pos] = mut
                mutated_seq = "".join(seq_list)
                mutated_seqs.append(mutated_seq)
                mutated_names.append(mutated_name)
                y_trues.append(None)
                y_preds.append(None)
                y_sigmas.append(None)
                acq_scores.append(None)

        out_df = pd.DataFrame(
            {
                "name": mutated_names,
                "sequence": mutated_seqs,
                "y_true": y_trues,
                "y_predicted": y_preds,
                "y_sigma": y_sigmas,
                "acq_score": acq_scores,
            }
        )

        return out_df

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

    ### Future warnings ###
    @property
    def x(self):
        import warnings

        warnings.warn(
            "'x' is deprecated and will be removed in a future version. Please use 'rep' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.rep  # Return the value of `rep` for compatibility

    @x.setter
    def x(self, value):
        import warnings

        warnings.warn(
            "'x' is deprecated and will be removed in a future version. Please use 'rep' instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.rep = value  # Automatically set `rep` to the provided value
