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