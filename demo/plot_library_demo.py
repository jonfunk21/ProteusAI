import proteusAI as pai
import matplotlib.pyplot as plt

# Initialize the library
library = pai.Library(
    source="demo/demo_data/Nitric_Oxide_Dioxygenase.csv",
    seqs_col="Sequence",
    y_col="Data",
    y_type="num",
    names_col="Description",  # Column containing the sequence descriptions
)
print(library.seqs)
# Compute embeddings using the specified method
library.compute(method="esm2_8M")

# Generate a UMAP plot
fig, ax, df = library.plot_umap(rep="esm2_8M")

# Customize and display the plot
plt.show()
