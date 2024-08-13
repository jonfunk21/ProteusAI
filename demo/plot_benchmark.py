import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Representation_type
TYPE = 'blosum62'
MAX_SAMPLE = 50

# Load data from JSON file
with open(f'usrs/benchmark/first_discovered_data_{TYPE}.json') as f:
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
g = sns.FacetGrid(df, col="Sample Size", col_wrap=2, height=4, aspect=1.5, sharey=True)

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
g.fig.suptitle('Rounds to Discover Top N Variants Across Different Sample Sizes', y=1.02)

# Adjust the layout
plt.tight_layout()
g.fig.subplots_adjust(top=0.9, right=0.85)

# Move legend to the right side
g.add_legend(title="Dataset", bbox_to_anchor=(0.75, 0.5), loc='center left', borderaxespad=0)

# Save the plot
plt.savefig(f'box_and_strip_plot_with_shared_y_axis_{TYPE}.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()
