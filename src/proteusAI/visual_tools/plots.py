import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import umap

representation_dict = {"One-hot":"ohe", "BLOSUM50":"blosum50", "BLOSUM62":"blosum62", "ESM-2":"esm2", "ESM-1v":"esm1v"}

def plot_predictions_vs_groundtruth(y_true: list, y_pred: list, title: Union[str, None] = None, 
                                    x_label: Union[str, None] = None, y_label: Union[str, None] = None, 
                                    plot_grid: bool = True, file: Union[str, None] = None, show_plot: bool = True
                                    ):
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=ax)
    
    # Set plot title and labels
    if title is None:
        title = 'Predicted vs. True y-values'
    if y_label is None:
        y_label = 'predicted y'
    if x_label is None:
        x_label = 'y'

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='grey', linestyle='dotted', linewidth=2)  # diagonal line
    ax.grid(plot_grid)

    # Save the plot to a file
    if file is not None:
        fig.savefig(file)

    # Show the plot
    if show_plot:
        plt.show()

    # Return the figure and axes
    return fig, ax

def plot_tsne(x: List[np.ndarray], y: Union[List[float], None] = None, 
              y_upper: Union[float, None] = None, y_lower: Union[float, None] = None, 
              names: Union[List[str], None] = None, y_type: str = 'num', 
              random_state: int = 42, rep_type: Union[str, None] = None):
    """
    Create a t-SNE plot and optionally color by y values, with special coloring for points outside given thresholds.
    Handles cases where y is None or a list of Nones by not applying hue.

    Args:
        x (List[np.ndarray]): List of sequence representations as numpy arrays.
        y (List[float]): List of y values, can be None or contain None.
        y_upper (float): Upper threshold for special coloring.
        y_lower (float): Lower threshold for special coloring.
        names (List[str]): List of names for each point.
        y_type (str): 'class' for categorical labels or 'num' for numerical labels.
        random_state (int): Random state.
        rep_type (str): Representation type used for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.array([t.numpy() if hasattr(t, 'numpy') else t for t in x])

    if len(x.shape) == 3:  # Flatten if necessary
        x = x.reshape(x.shape[0], -1)

    tsne = TSNE(n_components=2, verbose=1, random_state=random_state)
    z = tsne.fit_transform(x)

    df = pd.DataFrame(z, columns=['z1', 'z2'])
    df['y'] = y if y is not None and any(y) else None  # Use y if it's informative
    if len(names) == len(y):
        df['names'] = names
    else:
        df['names'] = [None] * len(y)

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    hue = 'y' if df['y'].isnull().sum() != len(df['y']) else None  # Use hue only if y is informative
    scatter = sns.scatterplot(x='z1', y='z2', hue=hue, palette=cmap if hue else None, data=df)

    # Apply special coloring only if y values are valid and thresholds are provided
    if hue and (y_upper is not None or y_lower is not None):
        outlier_mask = (df['y'] > y_upper) if y_upper is not None else np.zeros(len(df), dtype=bool)
        outlier_mask |= (df['y'] < y_lower) if y_lower is not None else np.zeros(len(df), dtype=bool)
        scatter.scatter(df['z1'][outlier_mask], df['z2'][outlier_mask], color='lightgrey')

    scatter.set_title(f"t-SNE projection of {rep_type if rep_type else 'data'}")
    return fig, ax, df


def plot_umap(x: List[np.ndarray], y: Union[List[float], None] = None, 
              y_upper: Union[float, None] = None, y_lower: Union[float, None] = None, 
              names: Union[List[str], None] = None, y_type: str = 'num', 
              random_state: int = 42, rep_type: Union[str, None] = None):
    """
    Create a UMAP plot and optionally color by y values, with special coloring for points outside given thresholds.
    Handles cases where y is None or a list of Nones by not applying hue.

    Args:
        x (List[np.ndarray]): List of sequence representations as numpy arrays.
        y (List[float]): List of y values, can be None or contain None.
        y_upper (float): Upper threshold for special coloring.
        y_lower (float): Lower threshold for special coloring.
        names (List[str]): List of names for each point.
        y_type (str): 'class' for categorical labels or 'num' for numerical labels.
        random_state (int): Random state.
        rep_type (str): Representation type used for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.array([t.numpy() if hasattr(t, 'numpy') else t for t in x])

    if len(x.shape) == 3:  # Flatten if necessary
        x = x.reshape(x.shape[0], -1)
    
    #assert isinstance(x, list), "x should be a list of numpy arrays"

    umap_model = umap.UMAP(n_components=2, random_state=random_state)
    z = umap_model.fit_transform(x)

    df = pd.DataFrame(z, columns=['z1', 'z2'])
    df['y'] = y if y is not None and any(y) else None  # Use y if it's informative
    if len(names) == len(y):
        df['names'] = names
    else:
        df['names'] = [None] * len(y)

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    hue = 'y' if df['y'].isnull().sum() != len(df['y']) else None  # Use hue only if y is informative
    scatter = sns.scatterplot(x='z1', y='z2', hue=hue, palette=cmap if hue else None, data=df)

    # Apply special coloring only if y values are valid and thresholds are provided
    if hue and (y_upper is not None or y_lower is not None):
        outlier_mask = (df['y'] > y_upper) if y_upper is not None else np.zeros(len(df), dtype=bool)
        outlier_mask |= (df['y'] < y_lower) if y_lower is not None else np.zeros(len(df), dtype=bool)
        scatter.scatter(df['z1'][outlier_mask], df['z2'][outlier_mask], color='lightgrey')

    scatter.set_title(f"UMAP projection of {rep_type if rep_type else 'data'}")
    return fig, ax, df

def plot_pca(x: List[np.ndarray], y: Union[List[float], None] = None, 
            y_upper: Union[float, None] = None, y_lower: Union[float, None] = None, 
            names: Union[List[str], None] = None, y_type: str = 'num', 
            random_state: int = 42, rep_type: Union[str, None] = None):
    """
    Create a PCA plot and optionally color by y values, with special coloring for points outside given thresholds.
    Handles cases where y is None or a list of Nones by not applying hue.

    Args:
        x (List[np.ndarray]): List of sequence representations as numpy arrays.
        y (List[float]): List of y values, can be None or contain None.
        y_upper (float): Upper threshold for special coloring.
        y_lower (float): Lower threshold for special coloring.
        names (List[str]): List of names for each point.
        y_type (str): 'class' for categorical labels or 'num' for numerical labels.
        random_state (int): Random state.
        rep_type (str): Representation type used for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.array([t.numpy() if hasattr(t, 'numpy') else t for t in x])

    if len(x.shape) == 3:  # Flatten if necessary
        x = x.reshape(x.shape[0], -1)

    pca = PCA(n_components=2, random_state=random_state)
    z = pca.fit_transform(x)

    df = pd.DataFrame(z, columns=['z1', 'z2'])
    df['y'] = y if y is not None and any(y) else None  # Use y if it's informative
    if len(names) == len(y):
        df['names'] = names
    else:
        df['names'] = [None] * len(y)

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    hue = 'y' if df['y'].isnull().sum() != len(df['y']) else None  # Use hue only if y is informative
    scatter = sns.scatterplot(x='z1', y='z2', hue=hue, palette=cmap if hue else None, data=df)

    # Apply special coloring only if y values are valid and thresholds are provided
    if hue and (y_upper is not None or y_lower is not None):
        outlier_mask = (df['y'] > y_upper) if y_upper is not None else np.zeros(len(df), dtype=bool)
        outlier_mask |= (df['y'] < y_lower) if y_lower is not None else np.zeros(len(df), dtype=bool)
        scatter.scatter(df['z1'][outlier_mask], df['z2'][outlier_mask], color='lightgrey')

    scatter.set_title(f"PCA projection of {rep_type if rep_type else 'data'}")
    return fig, ax, df