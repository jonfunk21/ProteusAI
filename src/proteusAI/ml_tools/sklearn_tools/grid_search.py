# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.svm import SVR
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import json

def knnr_grid_search(Xs_train: numpy.ndarray, Xs_test: numpy.ndarray, ys_train: list, ys_test: list, param_grid: dict=None, verbose: int=1):
    """
    Performs a KNN regressor grid search using 5 fold cross validation.

    Parameters:
    -----------
        Xs_train, Xs_test (numpy.ndarray): train and test values for training
        ys_train, ys_test (list): y values for train and test
        param_grid (dict): parameter grid for model
        verbose (int): Type of information printing during run

    Returns:
    --------
        Returns the best performing model of grid search, the test R squared value, correlation coefficient,
        p-value of the fit and a dataframe containing the fit information.
    """
    # Instantiate the model
    knnr = KNeighborsRegressor()

    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 15, 20],
            'p': [1, 2, 3]
        }
    # Create a GridSearchCV object and fit the model
    # grid_search = GridSearchCV(estimator=knnr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search = GridSearchCV(
        estimator=knnr,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=verbose,
        n_jobs=-1  # use all available cores
    )

    grid_search.fit(Xs_train, ys_train)
    test_r2 = grid_search.score(Xs_test, ys_test)
    predictions = grid_search.best_estimator_.predict(Xs_test)
    corr_coef, p_value = pearsonr(predictions, ys_test)

    # Print the best hyperparameters and the best score
    if verbose is not None:
        print("Best hyperparameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        print("Test R^2 score: ", test_r2)
        print("Correlation coefficient: {:.2f}".format(corr_coef))
        print("p-value: {:.4f}".format(p_value))

    return grid_search.best_estimator_, test_r2, corr_coef, p_value, pd.DataFrame.from_dict(grid_search.cv_results_)


def rfr_grid_search(Xs_train: numpy.ndarray, Xs_test: numpy.ndarray, ys_train: list, ys_test: list, param_grid: dict=None, verbose: int=1):
    """
    Performs a Random Forrest regressor grid search using 5 fold cross validation.

    Parameters:
    -----------
        Xs_train, Xs_test (numpy.ndarray): train and test values for training
        ys_train, ys_test (list): y values for train and test
        param_grid (dict): parameter grid for model
        verbose (int): Type of information printing during run

    Returns:
    --------
        Returns the best performing model of grid search, the test R squared value, correlation coefficient,
        p-value of the fit and a dataframe containing the fit information.
    """
    # Instantiate the model
    rfr = RandomForestRegressor(random_state=42)

    if param_grid is None:
        param_grid = {
            'n_estimators': [20, 50, 100, 200],
            'criterion': ['squared_error', 'absolute_error'],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 4],
        }

    # Create a GridSearchCV object and fit the model
    grid_search = GridSearchCV(
        estimator=rfr,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=verbose,
        n_jobs=-1  # use all available cores
    )
    grid_search.fit(Xs_train, ys_train)

    # Evaluate the performance of the model on the test set
    test_r2 = grid_search.score(Xs_test, ys_test)
    predictions = grid_search.best_estimator_.predict(Xs_test)
    corr_coef, p_value = pearsonr(predictions, ys_test)

    if verbose is not None:
        # Print the best hyperparameters and the best score
        print("Best hyperparameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        print("Test R^2 score: ", test_r2)
        print("Correlation coefficient: {:.2f}".format(corr_coef))
        print("p-value: {:.4f}".format(p_value))

    return grid_search.best_estimator_, test_r2, corr_coef, p_value, pd.DataFrame.from_dict(grid_search.cv_results_)


def svr_grid_search(Xs_train: numpy.ndarray, Xs_test: numpy.ndarray, ys_train: list, ys_test: list, param_grid: dict=None, verbose: int=1):
    """
    Performs a Support Vector regressor grid search using 5 fold cross validation.

    Parameters:
    -----------
        Xs_train, Xs_test (numpy.ndarray): train and test values for training
        ys_train, ys_test (list): y values for train and test
        param_grid (dict): parameter grid for model
        verbose (int): Type of information printing during run

    Returns:
    --------
        Returns the best performing model of grid search, the test R squared value, correlation coefficient,
        p-value of the fit and a dataframe containing the fit information.
    """
    # Instantiate the model
    svr = SVR()

    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 2, 5, 10, 100, 200, 400],
            'gamma': ['scale'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3],
        }

    # Create a GridSearchCV object and fit the model
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=verbose,
        n_jobs=-1  # use all available cores
    )
    grid_search.fit(Xs_train, ys_train)

    # Evaluate the performance of the model on the test set
    test_r2 = grid_search.score(Xs_test, ys_test)
    predictions = grid_search.best_estimator_.predict(Xs_test)
    corr_coef, p_value = pearsonr(predictions, ys_test)

    if verbose is not None:
        # Print the best hyperparameters and the best score
        print("Best hyperparameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        print("Test R^2 score: ", test_r2)
        print("Correlation coefficient: {:.2f}".format(corr_coef))
        print("p-value: {:.4f}".format(p_value))

    return grid_search.best_estimator_, test_r2, corr_coef, p_value, pd.DataFrame.from_dict(grid_search.cv_results_)


def plot_predictions_vs_groundtruth(y_true, y_pred, fname=None):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    
    # Extract model name from fname and use it in the plot title
    if fname is not None:
        name = fname.split('/')[-1].split('.')[0].replace('_', ' ')
        plt.title(f'Predicted vs. True Activity Levels for {name}')
    else:
        plt.title('Predicted vs. True Activity Levels')
        
    plt.xlabel('True Activity Levels')
    plt.ylabel('Predicted Activity Levels')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='grey', linestyle='dotted', linewidth=2)  # diagonal line
    plt.grid(True)

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

    return y_pred

def save_best_params_to_json(model, fname):
    best_params = model.best_params_
    with open(fname, 'w') as f:
        json.dump(best_params, f)

def plot_attention(attention, layer, head, sequence):
    """
    Plot the attention weights for a specific layer and head.

    :param attention: List of attention weights from the model
    :param layer: Index of the layer to visualize
    :param head: Index of the head to visualize
    :param sequence: Input sequence as a list of tokens
    """
    # Get the attention weights for the specified layer and head
    attn_weights = attention[layer][head].detach().cpu().numpy()

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 10))
    sns.heatmap(attn_weights, xticklabels=sequence, yticklabels=sequence, cmap="viridis")

    # Set plot title and labels
    plt.title(f'Attention weights - Layer {layer + 1}, Head {head + 1}')
    plt.xlabel('Input tokens')
    plt.ylabel('Output tokens')

    # Show the plot
    plt.show()