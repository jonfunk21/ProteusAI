import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

# Initialize the argparse parser
parser = argparse.ArgumentParser(description="Plot the benchmark results.")

# Add arguments
parser.add_argument('--rep', type=str, default='blosum62', help='Representation type.')
parser.add_argument('--max-sample', type=int, default=100, help='Maximum sample size.')
parser.add_argument('--model', type=str, default='gp', help='Model name.')
parser.add_argument('--acquisition-fn', type=str, default='ei', help='Acquisition function name.')

# Parse the arguments
args = parser.parse_args()

# Assign parsed arguments to variables
REP = args.rep
MAX_SAMPLE = args.max_sample
MODEL = args.model
ACQ_FN = args.acquisition_fn

# Dictionaries for pretty names
rep_dict = {"ohe":"One-hot", "blosum50":"BLOSUM50", "blosum62":"BLOSUM62", "esm2":"ESM-2", "esm1v":"ESM-1v", "vae":"VAE"}
model_dict = {"rf":"Random Forest", "knn":"KNN", "svm":"SVM", "esm2":"ESM-2", "esm1v":"ESM-1v", "gp":"Gaussian Process", "ridge":"Ridge Regression"}
acq_dict = {"ei":"Expected Improvement", "ucb":"Upper Confidence Bound", "greedy":"Greedy", "random":"Random"}

# Load data from JSON file
with open(f'usrs/benchmark/first_discovered_data_{MODEL}_{REP}_{ACQ_FN}.json') as f:
    data = json.load(f)

# Define the top N variants and sample sizes
datasets = list(data.keys())
sample_sizes = [int(i) for i in list(data[datasets[0]].keys())]
top_n_variants = ['5', '10', '20', '50', 'improved']

# Initialize a list to store the processed data
plot_data = []

# Process the data
for dataset, results in data.items():
    for sample_size, rounds in results.items():
        if rounds is not None:
            for i, round_count in enumerate(rounds):
                if round_count is not None:
                    plot_data.append({
                        'Dataset': dataset,
                        'Sample Size': int(sample_size),
                        'Top N Variants': top_n_variants[i],
                        'Rounds': round_count
                    })

# Create a DataFrame from the processed data
df = pd.DataFrame(plot_data)
df['Top N Variants'] = pd.Categorical(df['Top N Variants'], categories=top_n_variants, ordered=True)

# Set plot style
sns.set(style="whitegrid")

# Create a grid of box plots for each sample size with shared y-axis
g = sns.FacetGrid(df, col="Sample Size", col_wrap=2, height=5.5, aspect=1.8, sharey=True)  # Adjusted height and aspect

# Function to create box plot and strip plot
def plot_box_and_strip(data, x, y, **kwargs):
    ax = plt.gca()
    boxplot = sns.boxplot(data=data, x=x, y=y, color='lightgray', width=0.5, ax=ax)
    sns.stripplot(data=data, x=x, y=y, **kwargs)
    
    # Add median values as text
    medians = data.groupby(x)[y].median()
    for i, median in enumerate(medians):
        ax.text(i, median + 1, f'{median:.2f}', ha='center', va='bottom', color='black')  # Adjusting the position to be just above the median line
    
    # Set y-axis limit to MAX_SAMPLE
    ax.set_ylim(0, MAX_SAMPLE)

# Apply the function to each subplot
g.map_dataframe(plot_box_and_strip, x="Top N Variants", y="Rounds", hue="Dataset", dodge=True, jitter=True, palette="muted")

# Draw vertical bars between the sampling sizes
for ax in g.axes.flatten():
    for x in range(len(top_n_variants) - 1):
        ax.axvline(x + 0.5, color='grey', linestyle='-', linewidth=0.8)

# Adjust the title and layout
g.set_titles(col_template="Sample Size: {col_name}")
g.set_axis_labels("Top N Variants", "Rounds")

rep = rep_dict[REP]
model = model_dict[MODEL]
acq_fn = acq_dict[ACQ_FN]
subtitle = f'Representation: {rep}, Model: {model}, Acquisition Function: {acq_fn}'

# Move legend to the right side
g.add_legend(title="Dataset", bbox_to_anchor=(0.87, 0.47), loc='center left', borderaxespad=0)

# Set the main title and subtitle
g.fig.suptitle(f'Rounds to Discover Top N Variants Across Different Sample Sizes', y=0.94, x=0.44)
plt.text(0.44, 0.9, subtitle, ha='center', va='center', fontsize=10, transform=g.fig.transFigure)

# Adjust the layout to optimize spacing
plt.tight_layout(pad=2.0)  # Adjusted padding
g.fig.subplots_adjust(top=0.85, right=0.85, hspace=0.3, wspace=0.3)  # Adjusted spacing

# Save the plot
plt.savefig(f'usrs/benchmark/first_discovered_{MODEL}_{REP}_{ACQ_FN}.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()
