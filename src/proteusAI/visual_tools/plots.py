import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
from plotnine import ggplot, aes, geom_point, geom_abline, labs, theme_minimal, theme, element_line
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

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

def plot_predictions_vs_groundtruth_ggplot(data: pd.DataFrame,
                                           title: Union[str, None] = None, x_label: Union[str, None] = None, 
                                           y_label: Union[str, None] = None, plot_grid: bool = True, 
                                           file: Union[str, None] = None):
    """
    Plotting function used for the app.
    """

    # Set default labels if none provided
    if title is None:
        title = 'Predicted vs. True y-values'
    if y_label is None:
        y_label = 'Predicted y'
    if x_label is None:
        x_label = 'True y'

    # Create the ggplot object with updated aesthetics
    p = (ggplot(data, aes('y_true', 'y_pred')) +
         geom_point(alpha=0.5) +  # Blue points
         geom_abline(slope=1, intercept=0, color='grey', linetype='dotted', size=1.5) +
         labs(title=title, x=x_label, y=y_label) +
         theme_minimal())  # White background

    # Add grid if required
    if plot_grid:
        p += theme(panel_grid_major=element_line(color='grey', size=0.5))

    # Save the plot to a file
    if file is not None:
        p.save(file)

    # Return the plot object
    print(data)
    return p

def plot_tsne(x: List[np.ndarray], y: Union[List[float], None] = None, 
              y_upper: Union[float, None] = None, y_lower: Union[float, None] = None, 
              names: Union[List[str], None] = None, y_type: str = 'num', 
              random_state: int = 42, rep_type: Union[str, None] = None):
    """
    Create a t-SNE plot and color by y values, with special coloring for points outside given thresholds.

    Args:
        x (List[np.ndarray]): List of sequence representations as numpy arrays (R^1).
        y (List[float]): List of y values numpy array (R^1).
        y_upper (float): Upper threshold for special coloring.
        y_lower (float): Lower threshold for special coloring.
        names (List[str]): List of names for each point.
        y_type (str): 'class' for categorical labels or 'num' for numerical labels.
        random_state (int): Random state.
        rep_type (str): Representation type used for plotting.
    """
    
    fig, ax = plt.subplots(figsize=(10, 5))

    assert isinstance(x, list)

    if y is None:
        y = [None] * len(x)

    x = np.array([t.numpy() for t in x])

    # Handle OHE and Blosum encodings
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], -1)

    tsne = TSNE(n_components=2, verbose=1, random_state=random_state)
    z = tsne.fit_transform(x)

    df = pd.DataFrame(z, columns=['z1', 'z2'])
    df['y'] = y
    if names:
        df['names'] = names

    # Inverted representations mapping (if needed)
    inverted_reps = {v: k for k, v in representation_dict.items()}

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    scatter = sns.scatterplot(x='z1', y='z2', hue='y', palette=cmap, data=df)

    if y_upper is not None or y_lower is not None:
        # Color points outside the thresholds in light grey
        outlier_mask = np.full(len(df), False, dtype=bool)
        if y_upper is not None:
            outlier_mask |= (df['y'] > y_upper)
        if y_lower is not None:
            outlier_mask |= (df['y'] < y_lower)
        scatter.scatter(df['z1'][outlier_mask], df['z2'][outlier_mask], color='lightgrey')

    scatter.set(title=f"t-SNE projection of {inverted_reps.get(rep_type, rep_type)} representations")

    return fig, ax, df
