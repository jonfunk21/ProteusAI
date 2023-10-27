import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions_vs_groundtruth(y_true: list, y_pred: list, title: str = None, 
                                    x_label: str = None, y_label: str = None , plot_grid: bool = True, 
                                    file: str = None, show_plot: bool = True
                                    ):
    """
    Plot predicted values versus ground truth.

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

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    
    # Extract model name from fname and use it in the plot title
    if file is not None:
        name = file.split('/')[-1].split('.')[0].replace('_', ' ')

    # title of the plot
    if title is None:
        title = 'Predicted vs. True y-values'
    if y_label is None:
        y_label = 'predicted y'
    if x_label is None:
        x_label = 'y'

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='grey', linestyle='dotted', linewidth=2)  # diagonal line
    plt.grid(plot_grid)

    if file is not None:
        plt.savefig(file)

    if show_plot:
        plt.show()

    return y_pred