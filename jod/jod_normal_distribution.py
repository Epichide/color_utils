
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import numpy as np

"""_summary_
Functions to convert between JOD distance and probability using normal distribution. 
sigma is set to 1.4826 to match the standard deviation of the JOD scale.
"""

def jod_distance_to_probability(jod_distance):
    """Convert JOD distance to probability using the normal distribution CDF."""
    mu = 0
    sigma = 1
    xscale = 1 / norm.ppf(0.75, mu, sigma)
    print(f"Using xscale: {xscale} for conversion.")
    probability = norm.cdf(jod_distance/xscale, mu, sigma)
    return probability

def jod_probability_to_distance(probability,method="JOD"):
    """Convert probability to JOD distance using the normal distribution inverse CDF."""
    if method.lower()=="jod":
        mu = 0
        sigma = 1
        xscale = 1 / norm.ppf(0.75, mu, sigma)
        print(f"Using xscale: {xscale} for conversion.")
        jod_distance = norm.ppf(probability, mu, sigma) * xscale
        return jod_distance
    elif method.lower=="thurstone":
        # Thurstone scaling factor , eliminate the tail effect (==>approximately JOD CDF)
        jod_distance = (12/np.pi * np.arcsin( np.sqrt(probability) ) - 3)
        return jod_distance


def Cmatrix_2_Pmatrix(C_matrix):
    """Convert count matrix to probability matrix."""
    n_condition=C_matrix.shape[0]
    C_sum=C_matrix + C_matrix.T
    unanimous_mask=(C_sum<1)
    P_matrix=C_matrix/(C_matrix + C_matrix.T)
    P_matrix[unanimous_mask]=0.5  # 对于全为0的情况，设为0.5
    return P_matrix

def Pmatrix_2_Dmatrix(P_matrix):
    """Convert probability matrix to JOD distance matrix."""
    D_matrix=jod_probability_to_distance(P_matrix,method="thurstone")
    return D_matrix

def Dmatrix_2_iniQuality(D_matrix):
    D_mean=np.mean(D_matrix,axis=1) # mean of row
    return D_mean

def MLE_quality_function(C_matrix, ini_Quality):
    """Maximum Likelihood Estimation function for quality."""
    n_condition = P_matrix.shape[0]
    quality = ini_Quality.copy()
    for i in range(n_condition):
        for j in range(n_condition):
            if i != j:
                quality[i] += P_matrix[i, j] - P_matrix[j, i]
    return quality

def MLE_quality_estimation(P_matrix,ini_Quality, MLE_func,MLE_grad_func):
    pass


def sample_by_jod_distance(jod_distance, size=1):
    """Generate random samples based on JOD distance."""
    probs=jod_distance_to_probability(jod_distance)
    return samples

if __name__ == "__main__":
    p=jod_distance_to_probability(1)
    print(f"JOD distance 1 corresponds to probability: {p}")
    d=jod_probability_to_distance(0.75)
    print(f"Probability 0.75 corresponds to JOD distance: {d}")