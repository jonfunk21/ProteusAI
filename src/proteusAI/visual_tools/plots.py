import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions_vs_groundtruth(y_true, y_pred, file=None):
    """
    Plot predicted values versus ground truth.

    Args:
        y_true (list): True y values.
        y_pred (list): Predicted y values
    """

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    
    # Extract model name from fname and use it in the plot title
    if file is not None:
        name = file.split('/')[-1].split('.')[0].replace('_', ' ')
        plt.title(f'Predicted vs. True Activity Levels for {name}')
    else:
        plt.title('Predicted vs. True Activity Levels')
        
    plt.xlabel('y')
    plt.ylabel('predicted y')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='grey', linestyle='dotted', linewidth=2)  # diagonal line
    plt.grid(True)

    if file is not None:
        plt.savefig(file)
    else:
        plt.show()

    return y_pred