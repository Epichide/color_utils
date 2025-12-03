import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

def pw_scale(D, options=None):
    """
    Scaling method for pairwise comparisons, also for non-balanced
    (incomplete) designs.
    
    Parameters:
    -----------
    D : ndarray
        NxN matrix with positive integers. D[i,j] = k means that the
        condition i was better than j in k number of trials.
    options : dict, optional
        Dictionary with options. Recognized options:
        - 'prior': type of the distance prior. Options are:
            * 'none': do not use prior
            * 'gaussian': the normalised sum of probabilities of 
              observing a difference for all compared pairs of conditions.
            Default is 'gaussian'
        - 'regularization': Since the quality scores in pairwise comparisons
            are relative and the absolute value cannot be obtained, it
            is necessary to make an assumption how the absolute values
            are fixed in the optimization. Options are:
            * 'mean0': add a regularization term that makes the mean
              JOD value equal to 0 (default)
            * 'fix0': fix the score of the first condition to 0. That
              score is not optimized.
            The default is 'mean0'. 'mean0' results in a reduced
            overall estimation error as compared to 'fix0'. 'fix0' is 
            useful when one of the conditions is considered a
            baseline or a reference.
    
    Returns:
    --------
    Q : ndarray
        The JOD value for each method. The difference of 1 corresponds to
        75% of answers selecting one condition over another.
    R : ndarray
        Matrix of residuals. The residuals are due to projecting the
        differences into 1D space. These are the differences between the
        distances corresponding to the probabilities observed in the
        experiment and the resulting distances after scaling.
    
    Notes:
    ------
    The condition with index 0 (the first row in D) has the score fixed at value
    0 when using 'fix0' regularization. Always put "reference" condition at index 0.
    The measurement error is the smallest for the conditions that are the closest 
    to the reference condition.
    """
    
    # All elements must be non-negative integers
    D = np.array(D, dtype=float)
    if np.any(np.isinf(D)) or np.any(np.floor(D) != D) or np.any(D < 0):
        raise ValueError('Matrix of comparisons contains invalid inputs')
    
    # Default options
    if options is None:
        options = {}
    
    opt = {
        'prior': 'gaussian',
        'regularization': 'mean0',
        'use_gradients': True
    }
    
    # Update options
    for key, value in options.items():
        if key not in opt:
            raise ValueError(f'Unknown option {key}')
        
        if key == 'prior' and value not in ['none', 'bounded', 'gaussian']:
            raise ValueError('The "prior" option must be "none", "bounded", or "gaussian"')
        elif key == 'regularization' and value not in ['mean0', 'fix0']:
            raise ValueError('The "regularization" option must be "mean0" or "fix0"')
        
        opt[key] = value
    
    if D.shape[0] != D.shape[1]:
        raise ValueError('The comparison matrix must be square')
    
    # The number of compared conditions
    N = D.shape[0]
    
    # Change votes into probabilities - also for incomplete design
    M = D / (D + D.T)
    M[np.isnan(M)] = 0.5
    
    # Initial guess using ISO 20462 formula
    Q_init = np.mean(-(12/np.pi * np.arcsin(np.sqrt(M)) - 3), axis=1)
    
    # Find unanimous (UA) and non-unanimous relations (NUA) and build a graph
    NUA = (D > 0) * (D.T > 0)
    G = (D + D.T) > 0
    UA = G ^ NUA
    
    # Find connected components
    def connected_comp(G, node_gr, node, group):
        if node_gr[node] != 0:
            return node_gr
        node_gr[node] = group
        for nn in np.where(G[node, :])[0]:
            node_gr = connected_comp(G, node_gr, nn, group)
        return node_gr
    
    node_gr = np.zeros(N, dtype=int)
    group = 0
    for rr in range(N):
        if node_gr[rr] == 0:
            group += 1
        node_gr = connected_comp(G, node_gr, rr, group)
    
    # Add links between disconnected components
    Ng = np.max(node_gr)  # how many disconnected components
    if Ng > 1:
        warnings.warn(f'There are {Ng} disconnected components in the comparison graph. Some quality scores cannot be accurately computed')
        
        # Find the highest quality condition in each disconnected component
        Cb = np.zeros(Ng, dtype=int)
        for kk in range(1, Ng + 1):
            mask = (node_gr == kk)
            max_q = np.max(Q_init[mask])
            Cb[kk-1] = np.where((Q_init == max_q) & mask)[0][0]
        
        # Link all highest quality conditions and make them equivalent
        for kk in range(Ng):
            for jj in range(kk + 1, Ng):
                D[Cb[kk], Cb[jj]] = 1
                D[Cb[jj], Cb[kk]] = 1
                NUA[Cb[kk], Cb[jj]] = 1
                NUA[Cb[jj], Cb[kk]] = 1
    
    D_sum = D + D.T
    Dt = D.T
    nnz_d = (D_sum) > 0
    comp_made = np.sum(nnz_d)
    
    # Comparison matrix where we shift unanimous answers to the closest
    # non-unanimous solution
    D_wUA = D.copy()
    # Shift unanimous answers equal to 0 to 1
    D_wUA[(UA == 1) & (D == 0)] = 1
    # Subtract 1 from the rest of unanimous answers
    D_wUA[(UA == 1) & (D != 0)] = D_wUA[(UA == 1) & (D != 0)] - 1
    
    if opt['regularization'] == 'mean0':
        Q_0 = np.zeros(N)
    else:
        Q_0 = np.zeros(N - 1)
    
    # Get row and column indices for non-zero comparisons
    row_idx, col_idx = np.where(nnz_d)
    
    # Define the objective function and gradient
    def objective_function(q_trunc):
        if opt['regularization'] == 'mean0':
            q = q_trunc
        else:
            q = np.concatenate([[0], q_trunc])  # Add the condition with index 0, which is fixed to 0
        
        sigma_cdf = 1.4826  # for this sigma normal cumulative distribution is 0.75 @ 1
        
        # Compute distances
        Dd = q[:, np.newaxis] - q[np.newaxis, :]
        Dd = Dd[nnz_d]
        
        # Compute gradient of distances with respect to q
        dDd_dq = np.zeros((comp_made, N))
        for i in range(comp_made):
            dDd_dq[i, row_idx[i]] = 1
            dDd_dq[i, col_idx[i]] -= 1
        
        Pd = norm.cdf(Dd, 0, sigma_cdf)  # probabilities
        
        # Compute likelihoods
        prob = Pd
        Dn = D[nnz_d]
        Dtn = Dt[nnz_d]
        dprob_dq = norm.pdf(Dd, 0, sigma_cdf)[:, np.newaxis] * dDd_dq
        
        p = prob**Dn * (1 - prob)**Dtn
        
        # More numerically stable gradient computation
        dp_dq = np.zeros((comp_made, N))
        for i in range(comp_made):
            if Dn[i] > 0:
                term1 = Dn[i] * prob[i]**(max(Dn[i]-1, 0)) * (1-prob[i])**Dtn[i]
            else:
                term1 = 0
            
            if Dtn[i] > 0:
                term2 = Dtn[i] * (1-prob[i])**(max(Dtn[i]-1, 0)) * prob[i]**Dn[i]
            else:
                term2 = 0
            
            dp_dq[i, :] = (term1 - term2) * dprob_dq[i, :]
        
        # Regularization term
        L_reg = 0
        dLreg_dq = np.zeros_like(q)
        if opt['regularization'] == 'mean0':
            mean_q = np.mean(q)
            L_reg = 0.01 * mean_q**2
            dLreg_dq = 0.01 * 2 * mean_q * np.ones_like(q) / N
        
        # Compute prior
        if opt['prior'] == 'gaussian':
            n = D_sum[nnz_d]
            k = D_wUA[nnz_d]
            
            aux = prob**k * (1-prob)**(n-k)
            sumaux = np.sum(aux)
            prior = np.sum(aux / sumaux)
            
            # Derivatives
            daux_dq_A = np.zeros_like(aux)
            for i in range(len(aux)):
                if k[i] > 0:
                    term1 = k[i] * prob[i]**(max(k[i]-1, 0)) * (1-prob[i])**(n[i]-k[i])
                else:
                    term1 = 0
                
                if n[i]-k[i] > 0:
                    term2 = prob[i]**k[i] * (-n[i] + k[i]) * (1-prob[i])**(max(n[i]-k[i]-1, 0))
                else:
                    term2 = 0
                
                daux_dq_A[i] = term1 + term2
            
            part1 = (daux_dq_A[:, np.newaxis] * (1/sumaux)) * dprob_dq
            part2 = ((1/sumaux**2) * aux)[:, np.newaxis] * np.sum(daux_dq_A[:, np.newaxis] * dprob_dq, axis=0)
            dprior_dq = part1 - part2
            
        elif opt['prior'] == 'none':
            prior = np.ones(comp_made)
            dprior_dq = np.zeros((comp_made, N))
        else:
            raise ValueError(f'Unknown prior option {opt["prior"]}')
        
        # Compute gradient
        dpart_dq = np.zeros((comp_made, N))
        mask_p = p > 1e-400
        dpart_dq[mask_p] += (1 / np.maximum(p[mask_p], 1e-400))[:, np.newaxis] * dp_dq[mask_p]
        
        if opt['prior'] == 'gaussian':
            mask_prior = prior > -0.1 + 1e-400
            dpart_dq[mask_prior] += (1 / np.maximum(prior[mask_prior] + 0.1, 1e-400))[:, np.newaxis] * dprior_dq[mask_prior]
        
        grad = dLreg_dq - np.sum(dpart_dq, axis=0)
        
        if opt['regularization'] != 'mean0':
            grad = grad[1:]
        
        # Compute objective function value
        P = -np.sum(np.log(np.maximum(p, 1e-400)) + np.log(np.maximum(prior + 0.1, 1e-400))) + L_reg
        
        return P, grad
    
    # Optimize using BFGS
    result = minimize(objective_function, Q_0, method='BFGS', jac=True, options={'disp': False})
    
    if opt['regularization'] != 'mean0':
        Q = np.concatenate([[0], result.x])
    else:
        Q = result.x
    
    # Calculate the matrix of residuals
    JOD_dist_fit = Q[:, np.newaxis] - Q[np.newaxis, :]
    
    # Handle division by zero for JOD_dist_data
    with np.errstate(divide='ignore', invalid='ignore'):
        JOD_dist_data = norm.ppf(D / D_sum, 0, 1.4826)
        JOD_dist_data[np.isinf(JOD_dist_data)] = np.nan
        JOD_dist_data[np.isnan(JOD_dist_data)] = 0
    
    R = np.full(D.shape, np.nan)
    valid = nnz_d & NUA
    R[valid] = JOD_dist_fit[valid] - JOD_dist_data[valid]
    
    return Q, R


# Simple example showing how to execute scaling method
if __name__ == "__main__":
    # Example comparison matrices
    D1 = np.array([
        [0, 25, 9],
        [75, 0, 25],
        [91, 75, 0]
    ])
    
    D2 = np.array([
        [0, 25, 2, 3, 1],
        [75, 0, 2, 3, 1],
        [1, 2, 0, 3, 1],
        [1, 2, 3, 0, 1],
        [1, 2, 3, 1, 0]
    ])
    
    D3 = np.array([
        [0, 6, 3, 0, 0],
        [24, 0, 6, 3, 0],
        [27, 24, 0, 3, 1],
        [30, 27, 27, 0, 8],
        [30, 30, 29, 22, 0]
    ])
    
    D4 = np.array([
        [0, 0],
        [5, 0]
    ])
    
    # Test with different options
    # print("Testing with default options (gaussian prior, mean0 regularization):")
    # Q1, R1 = pw_scale(D1)
    # print(f"Q = {Q1}")
    # print(f"R = \n{R1}")
    # print()
    
    # print("Testing with fix0 regularization:")
    # Q2, R2 = pw_scale(D2, {'regularization': 'fix0'})
    # print(f"Q = {Q2}")
    # print()
    
    # print("Testing with no prior:")
    # Q3, R3 = pw_scale(D3, {'prior': 'none'})
    # print(f"Q = {Q3}")
    
    print("Testing with no prior:")
    Q4, R4 = pw_scale(D4, {'prior': 'none'})
    print(f"Q = {Q4}")
