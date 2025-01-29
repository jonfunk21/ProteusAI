import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import umap
import plotly.graph_objects as go


matplotlib.use("Agg")
representation_dict = {
    "One-hot": "ohe",
    "BLOSUM50": "blosum50",
    "BLOSUM62": "blosum62",
    "ESM-2": "esm2",
    "ESM-1v": "esm1v",
}

def plot_predictions_vs_groundtruth_interactive(
    y_true: list,
    y_pred: list,
    y_names: list,
    title: Union[str, None] = None,
    x_label: Union[str, None] = None,
    y_label: Union[str, None] = None,
    plot_grid: bool = True,
    file: Union[str, None] = None,
    show_plot: bool = True,
    width: Union[float, None] = None,
):

    if y_label is None:
        y_label = "Predicted Value"
    if x_label is None:
        x_label = "True Value"

    # Calculate diagonal line bounds
    min_val = min(min(y_true) * 1.05, min(y_pred) * 1.05)
    max_val = max(max(y_true) * 1.05, max(y_pred) * 1.05)

    # Confidence region
    confidence_traces = []
    if width is not None:
        x_range = np.linspace(min_val, max_val, 100)
        upper_conf = x_range + width
        lower_conf = x_range - width

        confidence_region = go.Scatter(
            x=np.concatenate([x_range, x_range[::-1]]),
            y=np.concatenate([upper_conf, lower_conf[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 98, 155, 0.2)",  # Hex #00629b converted to RGBA
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Region",
            hoverinfo="skip",  # Disable hover on the confidence region
            showlegend=False,
        )
        confidence_traces.append(confidence_region)

    # Create scatter plot with hover text
    scatter_trace = go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(size=10, opacity=0.8, color="#d64527"),
        name="Data Points",
        text=y_names,  # Hover text
        hoverinfo="text",  # Display only the text on hover
        showlegend=False,
    )

    # Add error bars
    error_bar_traces = []
    if width is not None:
        for true, pred in zip(y_true, y_pred):
            # Add vertical line (error bar)
            error_bar_traces.append(
                go.Scatter(
                    x=[true, true],
                    y=[pred - width, pred + width],
                    mode="lines",
                    line=dict(color="#C0C0C0", width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Add T-shape at top and bottom
            error_bar_traces.append(
                go.Scatter(
                    x=[
                        true - 0.005 * (max_val - min_val),
                        true + 0.005 * (max_val - min_val),
                    ],
                    y=[pred - width, pred - width],
                    mode="lines",
                    line=dict(color="#C0C0C0", width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            error_bar_traces.append(
                go.Scatter(
                    x=[
                        true - 0.005 * (max_val - min_val),
                        true + 0.005 * (max_val - min_val),
                    ],
                    y=[pred + width, pred + width],
                    mode="lines",
                    line=dict(color="#C0C0C0", width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Add diagonal line
    line_trace = go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#00629b", dash="dot", width=2),
        name="Diagonal Line",
        showlegend=False,
        hoverinfo="skip",
    )

    # Combine traces, ensuring confidence region and error bars are added first
    traces = confidence_traces + error_bar_traces + [line_trace, scatter_trace]

    # Create the layout
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_label,
            range=[min_val, max_val],
            showgrid=False,
            zeroline=True,  # Show x-axis
            zerolinewidth=1,
        ),
        yaxis=dict(
            title=y_label,
            range=[min_val, max_val],
            showgrid=False,
            zeroline=True,  # Show y-axis
            zerolinewidth=1,
        ),
        template="plotly_white",
        margin=dict(
            l=40, r=40, t=40, b=40  # Adjust margins for space around the plot
        ),
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    return fig

def plot_predictions_vs_groundtruth(
    y_true: list,
    y_pred: list,
    title: Union[str, None] = None,
    x_label: Union[str, None] = None,
    y_label: Union[str, None] = None,
    plot_grid: bool = True,
    file: Union[str, None] = None,
    show_plot: bool = True,
    width: Union[float, None] = None,
):
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=ax)

    # Set plot title and labels
    if title is None:
        title = "Predicted vs. True y-values"
    if y_label is None:
        y_label = "predicted y"
    if x_label is None:
        x_label = "y"

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add the diagonal line
    min_val = min(min(y_true) * 1.05, min(y_pred) * 1.05)
    max_val = max(max(y_true) * 1.05, max(y_pred) * 1.05)
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="grey",
        linestyle="dotted",
        linewidth=2,
    )

    # Add vertical error bars and confidence region if width is specified
    if width is not None:
        # Add error bars with T-shape
        for true, pred in zip(y_true, y_pred):
            ax.plot(
                [true, true], [pred - width, pred + width], color="darkgrey", alpha=0.8
            )
            ax.plot(
                [
                    true - 0.005 * (max_val - min_val),
                    true + 0.005 * (max_val - min_val),
                ],
                [pred - width, pred - width],
                color="darkgrey",
                alpha=0.8,
            )
            ax.plot(
                [
                    true - 0.005 * (max_val - min_val),
                    true + 0.005 * (max_val - min_val),
                ],
                [pred + width, pred + width],
                color="darkgrey",
                alpha=0.8,
            )

        # Add confidence region
        x_range = np.linspace(min_val, max_val, 100)
        upper_conf = x_range + width
        lower_conf = x_range - width
        ax.fill_between(
            x_range,
            lower_conf,
            upper_conf,
            color="blue",
            alpha=0.08,
            label="Confidence Region",
        )

    # Adjust layout to remove extra whitespace
    fig.tight_layout()

    # Add grid if specified
    ax.set_xlim(min_val, max_val)
    ax.grid(plot_grid)

    # Save the plot to a file
    if file is not None:
        fig.savefig(file)

    # Show the plot
    if show_plot:
        plt.show()

    # Return the figure and axes
    return fig, ax


def plot_tsne(
    x: List[np.ndarray],
    y: Union[List[Union[float, str]], None] = None,
    y_upper: Union[float, None] = None,
    y_lower: Union[float, None] = None,
    names: Union[List[str], None] = None,
    y_type: str = "num",
    random_state: int = 42,
    rep_type: Union[str, None] = None,
    highlight_mask: Union[List[Union[int, float]], None] = None,
    highlight_label: str = "Highlighted",
    df: Union[pd.DataFrame, None] = None,
):
    """
    Create a t-SNE plot and optionally color by y values, with special coloring for points outside given thresholds.
    Handles cases where y is None or a list of Nones by not applying hue.
    Optionally highlights points based on the highlight_mask.

    Args:
        x (List[np.ndarray]): List of sequence representations as numpy arrays.
        y (List[Union[float, str]]): List of y values, can be None or contain None.
        y_upper (float): Upper threshold for special coloring.
        y_lower (float): Lower threshold for special coloring.
        names (List[str]): List of names for each point.
        y_type (str): 'class' for categorical labels or 'num' for numerical labels.
        random_state (int): Random state.
        rep_type (str): Representation type used for plotting.
        highlight_mask (List[Union[int, float]]): List of mask values, non-zero points will be highlighted.
        highlight_label (str): Text for the legend entry of highlighted points.
        df (pd.DataFrame): DataFrame containing the data to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.array([t.numpy() if hasattr(t, "numpy") else t for t in x])

    if len(x.shape) == 3:  # Flatten if necessary
        x = x.reshape(x.shape[0], -1)

    if df is None:
        tsne = TSNE(n_components=2, verbose=1, random_state=random_state)
        z = tsne.fit_transform(x)

        df = pd.DataFrame(z, columns=["z1", "z2"])
        df["y"] = y if y is not None and any(y) else None  # Use y if it's informative
        if names and len(names) == len(y):
            df["names"] = names
        else:
            df["names"] = [None] * len(y)

    # Handle the palette based on whether y is numerical or categorical
    if isinstance(y[0], (int, float)):  # If y is numerical
        cmap = sns.cubehelix_palette(rot=-0.2, as_cmap=True)
    else:  # If y is categorical
        cmap = sns.color_palette("Set2", as_cmap=False)

    hue = (
        "y" if df["y"].isnull().sum() != len(df["y"]) else None
    )  # Use hue only if y is informative
    scatter = sns.scatterplot(
        x="z1", y="z2", hue=hue, palette=cmap if hue else None, data=df
    )

    # Apply special coloring only if y values are valid and thresholds are provided
    if hue and (y_upper is not None or y_lower is not None):
        outlier_mask = (
            (df["y"] > y_upper)
            if y_upper is not None
            else np.zeros(len(df), dtype=bool)
        )
        outlier_mask |= (
            (df["y"] < y_lower)
            if y_lower is not None
            else np.zeros(len(df), dtype=bool)
        )
        scatter.scatter(
            df["z1"][outlier_mask], df["z2"][outlier_mask], color="lightgrey"
        )

    # Highlight points based on the highlight_mask
    if highlight_mask is not None:
        highlight_mask = np.array(highlight_mask)
        highlight_points = highlight_mask != 0  # Non-zero entries in the highlight_mask
        scatter.scatter(
            df["z1"][highlight_points],
            df["z2"][highlight_points],
            color="red",
            marker="x",
            s=60,
            alpha=0.7,
            label=highlight_label,
        )

    scatter.set_title(f"t-SNE projection of {rep_type if rep_type else 'data'}")

    # Add the legend, making sure to include highlighted points
    handles, labels = scatter.get_legend_handles_labels()
    if highlight_label in labels:
        ax.legend(handles, labels, title="Legend")
    else:
        ax.legend(title="Legend")

    return plot_interactive_scatterplot(df), ax, df


def plot_umap(
    x: List[np.ndarray],
    y: Union[List[Union[float, str]], None] = None,
    y_upper: Union[float, None] = None,
    y_lower: Union[float, None] = None,
    names: Union[List[str], None] = None,
    y_type: str = "num",
    random_state: int = 42,
    rep_type: Union[str, None] = None,
    highlight_mask: Union[List[Union[int, float]], None] = None,
    highlight_label: str = "Highlighted",
    df: Union[pd.DataFrame, None] = None,
    html: bool = False,
):
    """
    Create a UMAP plot and optionally color by y values, with special coloring for points outside given thresholds.
    Handles cases where y is None or a list of Nones by not applying hue.
    Optionally highlights points based on the highlight_mask.

    Args:
        x (List[np.ndarray]): List of sequence representations as numpy arrays.
        y (List[Union[float, str]]): List of y values, can be None or contain None.
        y_upper (float): Upper threshold for special coloring.
        y_lower (float): Lower threshold for special coloring.
        names (List[str]): List of names for each point.
        y_type (str): 'class' for categorical labels or 'num' for numerical labels.
        random_state (int): Random state.
        rep_type (str): Representation type used for plotting.
        highlight_mask (List[Union[int, float]]): List of mask values, non-zero points will be highlighted.
        highlight_label (str): Text for the legend entry of highlighted points.
        df (pd.DataFrame): DataFrame containing the data to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.array([t.numpy() if hasattr(t, "numpy") else t for t in x])

    if len(x.shape) == 3:  # Flatten if necessary
        x = x.reshape(x.shape[0], -1)

    if df is None:
        umap_model = umap.UMAP(
            n_neighbors=70, min_dist=0.0, n_components=2, random_state=random_state
        )

        z = umap_model.fit_transform(x)
        df = pd.DataFrame(z, columns=["z1", "z2"])
        df["y"] = y if y is not None and any(y) else None  # Use y if it's informative
        if names and len(names) is not None and len(names) == len(y):
            df["names"] = names
        else:
            df["names"] = [None] * len(y)

    # Handle the palette based on whether y is numerical or categorical
    if isinstance(y[0], (int, float)):  # If y is numerical
        cmap = sns.cubehelix_palette(rot=-0.2, as_cmap=True)
    else:  # If y is categorical
        cmap = sns.color_palette("Set2", as_cmap=False)

    hue = (
        "y" if df["y"].isnull().sum() != len(df["y"]) else None
    )  # Use hue only if y is informative
    scatter = sns.scatterplot(
        x="z1", y="z2", hue=hue, palette=cmap if hue else None, data=df
    )

    # Apply special coloring only if y values are valid and thresholds are provided
    if hue and (y_upper is not None or y_lower is not None):
        outlier_mask = (
            (df["y"] > y_upper)
            if y_upper is not None
            else np.zeros(len(df), dtype=bool)
        )
        outlier_mask |= (
            (df["y"] < y_lower)
            if y_lower is not None
            else np.zeros(len(df), dtype=bool)
        )
        scatter.scatter(
            df["z1"][outlier_mask], df["z2"][outlier_mask], color="lightgrey"
        )

    # Highlight points based on the highlight_mask
    if highlight_mask is not None:
        highlight_mask = np.array(highlight_mask)
        highlight_points = highlight_mask != 0  # Non-zero entries in the highlight_mask
        scatter.scatter(
            df["z1"][highlight_points],
            df["z2"][highlight_points],
            color="red",
            marker="x",
            s=60,
            alpha=0.7,
            label=highlight_label,
        )

    scatter.set_title(f"UMAP projection of {rep_type if rep_type else 'data'}")

    # Add the legend, making sure to include highlighted points
    handles, labels = scatter.get_legend_handles_labels()
    if highlight_label in labels:
        ax.legend(handles, labels, title="Legend")
    else:
        ax.legend(title="Legend")

    return plot_interactive_scatterplot(df), ax, df


def plot_interactive_scatterplot(df):
    """
    A function for plotting an interactive scatterplot of a dataframe containing data with x & y coordinates and
    a 'y' column containing numerical data with which to label and color-code the points.

    Args:
        df (pd.DataFrame): DataFrame with columns ['z1', 'z2', 'y', 'names'].

    Returns:
        fig: Plotly Figure object for the interactive plot.
    """
    if not {"z1", "z2", "y", "names"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'z1', 'z2', 'y', and 'names' columns.")

    # Normalize 'y' column for color scaling
    y_min, y_max = df["y"].min(), df["y"].max()
    df["y_normalized"] = (df["y"] - y_min) / (y_max - y_min)

    custom_colorscale = [
        [0.0, "#d64527"],  # Minimum value (deep red)
        [0.5, "white"],  # Zero value (white)
        [1.0, "#00629b"],  # Maximum value (deep blue)
    ]

    # Create the scatter plot using graph_objects
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["z1"],
            y=df["z2"],
            mode="markers",
            marker=dict(
                size=10,  # Increase the size of the points
                color=df["y_normalized"],  # Color based on normalized 'y'
                colorscale=custom_colorscale,  # Apply custom color scale
                line=dict(color="black", width=1),  # Black outline with 1px width
                colorbar=dict(
                    title="Labels", ticks="outside"
                ),  # Add colorbar for reference
            ),
            text=df["names"],  # Hover information
            hovertemplate=("<b>%{text}<br>Label: %{marker.color:.2f}<extra></extra>"),
        )
    )

    # Update layout for cleaner visualization
    fig.update_layout(
        xaxis=dict(
            title="Dimension 1",
            showgrid=False,
            showticklabels=False,
            showline=True,
            linewidth=2,
            linecolor="black",
            zeroline=False,  # Disable the x=0 line
        ),
        yaxis=dict(
            title="Dimension 2",
            showgrid=False,
            showticklabels=False,
            showline=True,
            linewidth=2,
            linecolor="black",
            zeroline=False,  # Disable the x=0 line
        ),
        template="plotly_white",
        legend_title="Labels",
    )

    return fig


def plot_pca(
    x: List[np.ndarray],
    y: Union[List[Union[float, str]], None] = None,
    y_upper: Union[float, None] = None,
    y_lower: Union[float, None] = None,
    names: Union[List[str], None] = None,
    y_type: str = "num",
    random_state: int = 42,
    rep_type: Union[str, None] = None,
    highlight_mask: Union[List[Union[int, float]], None] = None,
    highlight_label: str = "Highlighted",
    df: Union[pd.DataFrame, None] = None,
):
    """
    Create a PCA plot and optionally color by y values, with special coloring for points outside given thresholds.
    Handles cases where y is None or a list of Nones by not applying hue.
    Optionally highlights points based on the highlight_mask.

    Args:
        x (List[np.ndarray]): List of sequence representations as numpy arrays.
        y (List[Union[float, str]]): List of y values, can be None or contain None.
        y_upper (float): Upper threshold for special coloring.
        y_lower (float): Lower threshold for special coloring.
        names (List[str]): List of names for each point.
        y_type (str): 'class' for categorical labels or 'num' for numerical labels.
        random_state (int): Random state.
        rep_type (str): Representation type used for plotting.
        highlight_mask (List[Union[int, float]]): List of mask values, non-zero points will be highlighted.
        highlight_label (str): Text for the legend entry of highlighted points.
        df (pd.DataFrame): DataFrame containing the data to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.array([t.numpy() if hasattr(t, "numpy") else t for t in x])

    if len(x.shape) == 3:  # Flatten if necessary
        x = x.reshape(x.shape[0], -1)

    if df is None:
        pca = PCA(n_components=2, random_state=random_state)
        z = pca.fit_transform(x)
        df = pd.DataFrame(z, columns=["z1", "z2"])
        df["y"] = y if y is not None and any(y) else None  # Use y if it's informative
        if names and len(names) == len(y):
            df["names"] = names
        else:
            df["names"] = [None] * len(y)

    # Handle the palette based on whether y is numerical or categorical
    if isinstance(y[0], (int, float)):  # If y is numerical
        cmap = sns.cubehelix_palette(rot=-0.2, as_cmap=True)
    else:  # If y is categorical
        cmap = sns.color_palette("Set2", as_cmap=False)

    hue = (
        "y" if df["y"].isnull().sum() != len(df["y"]) else None
    )  # Use hue only if y is informative
    scatter = sns.scatterplot(
        x="z1", y="z2", hue=hue, palette=cmap if hue else None, data=df
    )

    # Apply special coloring only if y values are valid and thresholds are provided
    if hue and (y_upper is not None or y_lower is not None):
        outlier_mask = (
            (df["y"] > y_upper)
            if y_upper is not None
            else np.zeros(len(df), dtype=bool)
        )
        outlier_mask |= (
            (df["y"] < y_lower)
            if y_lower is not None
            else np.zeros(len(df), dtype=bool)
        )
        scatter.scatter(
            df["z1"][outlier_mask], df["z2"][outlier_mask], color="lightgrey"
        )

    # Highlight points based on the highlight_mask
    if highlight_mask is not None:
        highlight_mask = np.array(highlight_mask)
        highlight_points = highlight_mask != 0  # Non-zero entries in the highlight_mask
        scatter.scatter(
            df["z1"][highlight_points],
            df["z2"][highlight_points],
            color="red",
            marker="x",
            s=60,
            alpha=0.7,
            label=highlight_label,
        )

    scatter.set_title(f"PCA projection of {rep_type if rep_type else 'data'}")

    # Add the legend, making sure to include highlighted points
    handles, labels = scatter.get_legend_handles_labels()
    if highlight_label in labels:
        ax.legend(handles, labels, title="Legend")
    else:
        ax.legend(title="Legend")

    return plot_interactive_scatterplot(df), ax, df
