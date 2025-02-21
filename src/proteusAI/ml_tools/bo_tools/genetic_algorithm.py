import numpy as np
import random


#####################################
### Simulated Annealing Discovery ###
#####################################


def precompute_distances(vectors):
    """Precompute the pairwise Euclidean distance matrix."""
    num_vectors = len(vectors)
    distance_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def diversity_score_incremental(
    current_score, selected_indices, idx_in, idx_out, distance_matrix
):
    """Update the diversity score incrementally when a vector is swapped."""
    new_score = current_score

    for idx in selected_indices:
        if idx != idx_out:
            new_score -= distance_matrix[idx_out, idx]
            new_score += distance_matrix[idx_in, idx]

    return new_score


def simulated_annealing(
    vectors,
    N,
    initial_temperature=1000.0,
    cooling_rate=0.003,
    max_iterations=10000,
    pbar=None,
):
    """
    Simulated Annealing to select N vectors that maximize diversity.

    Args:
        vectors (list): List of numpy arrays.
        N (int): Number of sequences that should be sampled.
        initial_temperature (float): Initial temperature of the simulated annealing algorithm. Default 1000.0.
        cooling_rate (float): Cooling rate of the simulated annealing algorithm. Default 0.003.
        max_iterations (int): Maximum number of iterations of the simulated annealing algorithm. Default 10000.

    Returns:
        list: Indices of diverse vectors.
    """

    if pbar:
        pbar.set(message="Computing distance matrix", detail="...")

    # Precompute all pairwise distances
    distance_matrix = precompute_distances(vectors)

    # Randomly initialize the selection of N vectors
    selected_indices = random.sample(range(len(vectors)), N)
    current_score = sum(
        distance_matrix[i, j]
        for i in selected_indices
        for j in selected_indices
        if i < j
    )

    temperature = initial_temperature
    best_score = current_score
    best_selection = selected_indices[:]

    for iteration in range(max_iterations):

        if pbar:
            pbar.set(iteration, message="Minimizing energy", detail="...")

        # Randomly select a vector to swap
        idx_out = random.choice(selected_indices)
        idx_in = random.choice(
            [i for i in range(len(vectors)) if i not in selected_indices]
        )

        # Incrementally update the diversity score
        new_score = diversity_score_incremental(
            current_score, selected_indices, idx_in, idx_out, distance_matrix
        )

        # Decide whether to accept the new solution
        delta = new_score - current_score
        if delta > 0 or np.exp(delta / temperature) > random.random():
            selected_indices.remove(idx_out)
            selected_indices.append(idx_in)
            current_score = new_score

            # Update the best solution found so far
            if new_score > best_score:
                best_score = new_score
                best_selection = selected_indices[:]

        # Cool down the temperature
        temperature *= 1 - cooling_rate

        # Early stopping if the temperature is low enough
        # if temperature < 1e-8:
        #    break

    return best_selection, best_score


#######################################
### Genetic Algorithm for Mutations ###
#######################################


def find_mutations(sequences):
    """
    Takes a list of protein sequences and returns a dictionary with mutation positions
    as keys and lists of amino acids at those positions as values.

    Parameters:
    sequences (list): A list of protein sequences (strings).

    Returns:
    dict: A dictionary where keys are positions (1-indexed) and values are lists of amino acids at those positions.
            If sequences have different lengths, returns all possible mutations at all positions.
    """
    # Check if the list is empty
    if not sequences:
        return {}

    # Check if all sequences have the same length
    lengths = {len(seq) for seq in sequences}
    if len(lengths) > 1:
        print(
            "WARNING: Sequences have different lengths. Exploitation won't work and explore will be set to 1."
        )
        # Get the maximum length
        max_length = max(lengths)
        mutations = {}
        all_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

        # For each position up to the longest sequence, allow all possible mutations
        for i in range(max_length):
            mutations[i + 1] = all_amino_acids  # +1 to make position 1-indexed
        return mutations

    # Convert sequences to numpy array for faster operations
    seq_array = np.array([list(seq) for seq in sequences])

    # Original logic for same-length sequences
    mutations = {}
    reference_seq = sequences[0]

    # For each position, get unique amino acids
    for i in range(len(reference_seq)):
        unique_aas = np.unique(seq_array[:, i])

        # If there is more than one unique amino acid at this position, it's a mutation
        if len(unique_aas) > 1:
            mutations[i + 1] = unique_aas.tolist()  # +1 to make position 1-indexed

    return mutations
