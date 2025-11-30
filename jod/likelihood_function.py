import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
from jod_normal_distribution import jod_distance_to_probability, jod_probability_to_distance

"""_summary_
Functions to compute likelihood functions for binomial distributions in the context of JOD distances.
"""

def generate_binomial_samples(n, p, size):
    """Generate samples from a binomial distribution."""
    return np.random.binomial(n, p, size)

def likelihood_binomial_function(n,k,p):
    """Calculate the likelihood of observing k successes in n trials for a binomial distribution."""
    coeff = np.math.comb(n, k)
    likelihood = coeff * (p ** k) * ((1 - p) ** (n - k))
    return likelihood

def MLE_binomial(n, k):
    """Calculate the Maximum Likelihood Estimate (MLE) for the probability of success p."""
    return k / n

if __name__ == "__main__":
    # 不同样本数量下，观察到不同通过率的似然函数变化
    sample_numbers=[5,4,10,8,30,28]
    colors=["#efaa63","orange","#ce92e8","magenta","#8d9dee","blue"]
    linestyles=["-","--","-","--","-","--"]
    jods=np.linspace(-5,7,100)
    probs=jod_distance_to_probability(jods)
    oberserve_pass_likelihoods=[] # 观察到全为通过的似然函数
    pass_rate=0.75
    plt.plot(jods, probs, label='JOD to Probability', color='green')
    for n in sample_numbers:
        likelihoods=[]
        k=int(np.ceil(n*pass_rate))  # 通过样本数量
        print(f"Sample n={n}, observed k={k}")
        for p in probs:
            likelihood=likelihood_binomial_function(n,k,p)
            likelihoods.append(likelihood)
        oberserve_pass_likelihoods.append(likelihoods)
    # 绘图
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(sample_numbers):
        k=int(np.ceil(n*pass_rate))  # 通过样本数量
        plt.plot(jods, oberserve_pass_likelihoods[i], 
                 label=f'sample n={n} (passrate={pass_rate}:{k}/{n})', 
                 color=colors[i], linestyle=linestyles[i])
    plt.xlabel('JOD Distance')
    plt.ylabel('Likelihood')
    plt.title(f'Likelihood Function for Binomial Distribution (passrate={pass_rate})')
    plt.legend()
    plt.xticks(np.arange(-5, 8, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.show()
    

        
        
        