# This source code is part of the proteusAI package and is distributed
# under the MIT License.

__name__ = "proteusAI"
__author__ = "Jonathan Funk and Laura Sofia Machado"

import numpy as np
from scipy.stats import norm

def greedy(mean, std=None, current_best=None, xi=None):
    """
    Greedy acquisition function.

    Args:
        mean (np.array): This is the mean function from the GP over the considered set of points.
        std (np.array, optional): This is the standard deviation function from the GP over the considered set of points. Default is None.
        current_best (float, optional): This is the current maximum of the unknown function: mu^+. Default is None.
        xi (float, optional): Small value added to avoid corner cases. Default is None.

    Returns:
        np.array: The mean values for all the points, as greedy acquisition selects the best based on mean.
    """
    return mean


def EI(mean, std, current_best, xi=0.1):
    """
    Expected Improvement acquisition function.

    It implements the following function:

            | (mu - mu^+ - xi) Phi(Z) + sigma phi(Z) if sigma > 0
    EI(x) = |
            | 0                                       if sigma = 0

            where Phi is the CDF and phi the PDF of the normal distribution
            and
            Z = (mu - mu^+ - xi) / sigma

    Args:
        mean (np.array): This is the mean function from the GP over the considered set of points.
        std (np.array): This is the standard deviation function from the GP over the considered set of points.
        current_best (float): This is the current maximum of the unknown function: mu^+.
        xi (float): Small value added to avoid corner cases.
    
    Returns:
        np.array: The value of this acquisition function for all the points.
    """
    
    Z = (mean - current_best - xi) / (std + 1e-9)
    EI = (mean - current_best - xi) * norm.cdf(Z) + std * norm.pdf(Z)
    EI[std == 0] = 0
    
    return EI


def UCB(mean, std, current_best=None, kappa=1.5):
    """
    Upper-Confidence Bound acquisition function.

    Args:
        mean (np.array): This is the mean function from the GP over the considered set of points.
        std (np.array): This is the standard deviation function from the GP over the considered set of points.
        current_best (float, optional): This is the current maximum of the unknown function: mu^+. Default is None.
        kappa (float): Exploration-exploitation trade-off parameter. The higher the value, the more exploration. Default is 0.
    
    Returns:
        np.array: The value of this acquisition function for all the points.
    """
    return mean + kappa * std


def random_acquisition(mean, std=None, current_best=None, xi=None):
    """
    Random acquisition function. Assigns random acquisition values to all points in the unobserved set.

    Args:
        mean (np.array): This is the mean function from the GP over the considered set of points.
        std (np.array, optional): This is the standard deviation function from the GP over the considered set of points. Default is None.
        current_best (float, optional): This is the current maximum of the unknown function: mu^+. Default is None.
        xi (float, optional): Small value added to avoid corner cases. Default is None.

    Returns:
        np.array: Random acquisition values for all points in the unobserved set.
    """
    n_unobserved = len(mean)
    np.random.seed(None) 
    random_acq_values = np.random.random(n_unobserved)  
    return random_acq_values