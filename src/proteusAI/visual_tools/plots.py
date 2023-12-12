import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
from plotnine import ggplot, aes, geom_point, geom_abline, labs, theme_minimal, theme, element_line
import pandas as pd

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