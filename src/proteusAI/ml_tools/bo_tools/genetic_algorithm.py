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


def diversity_score_incremental(current_score, selected_indices, idx_in, idx_out, distance_matrix):
    """Update the diversity score incrementally when a vector is swapped."""
    new_score = current_score
    
    for idx in selected_indices:
        if idx != idx_out:
            new_score -= distance_matrix[idx_out, idx]
            new_score += distance_matrix[idx_in, idx]
    
    return new_score


def simulated_annealing(vectors, N, initial_temperature=1000.0, cooling_rate=0.003, max_iterations=10000, pbar=None):
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
        pbar.set(message="Computing distance matrix", detail=f"...")

    # Precompute all pairwise distances
    distance_matrix = precompute_distances(vectors)
    
    # Randomly initialize the selection of N vectors
    selected_indices = random.sample(range(len(vectors)), N)
    current_score = sum(distance_matrix[i, j] for i in selected_indices for j in selected_indices if i < j)

    temperature = initial_temperature
    best_score = current_score
    best_selection = selected_indices[:]

    for iteration in range(max_iterations):

        if pbar:
            pbar.set(iteration, message="Minimizing energy", detail=f"...")

        # Randomly select a vector to swap
        idx_out = random.choice(selected_indices)
        idx_in = random.choice([i for i in range(len(vectors)) if i not in selected_indices])

        # Incrementally update the diversity score
        new_score = diversity_score_incremental(current_score, selected_indices, idx_in, idx_out, distance_matrix)
        
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
        temperature *= (1 - cooling_rate)
        
        # Early stopping if the temperature is low enough
        #if temperature < 1e-8:
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
    """
    # Check if the list is empty
    if not sequences:
        return {}

    # Initialize a dictionary to store mutations
    mutations = {}

    # Get the reference sequence (assuming all sequences have the same length)
    reference_seq = sequences[0]

    # Iterate through each position in the sequence
    for i in range(len(reference_seq)):
        # Initialize a set to store the different amino acids at this position
        amino_acids = set()

        # Check the amino acid at this position in each sequence
        for seq in sequences:
            amino_acids.add(seq[i])

        # If there is more than one unique amino acid at this position, it's a mutation
        if len(amino_acids) > 1:
            mutations[i + 1] = list(amino_acids)  # +1 to make position 1-indexed

    return mutations