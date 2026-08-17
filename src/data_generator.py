import numpy as np
from scipy.stats import norm, t
from scipy.special import expit

def generate_data(n_rows, n_cols, r, p, alpha, noise_type='logistic', model_type='general', skewness=0.0, seed=42):
    np.random.seed(seed)
    
    if model_type == 'rasch':
        theta = np.random.uniform(-1, 1, n_rows)
        beta = np.random.uniform(-1, 1, n_cols)
        theta = theta - np.mean(theta)
        M_star = theta[:, None] - beta[None, :]
    else:
        M1 = np.random.uniform(-0.5, 0.5, (n_rows, r))
        M2 = np.random.uniform(-0.5, 0.5, (n_cols, r))
        M_star = M1 @ M2.T

    M_star = M_star * (alpha / np.max(np.abs(M_star)))
    
    if skewness == 0.0:
        P_matrix = np.full((n_rows, n_cols), p)
    else:
        row_probs = np.exp(-skewness * np.linspace(0, 1, n_rows))
        col_probs = np.exp(-skewness * np.linspace(0, 1, n_cols))
        P_matrix = np.outer(row_probs, col_probs)
        P_matrix = P_matrix * (p / np.mean(P_matrix))
        P_matrix = np.clip(P_matrix, 0.01, 0.99)
        
    Omega = np.random.rand(n_rows, n_cols) < P_matrix

    if noise_type == 'logistic':
        prob = expit(M_star)
    elif noise_type == 'heavy_tail':
        heavy_noise = t.rvs(df=2, size=(n_rows, n_cols)) * 0.5
        prob = expit(M_star + heavy_noise)
    else:
        prob = norm.cdf(M_star)

    Y_true = np.ones((n_rows, n_cols))
    Y_true[np.random.rand(n_rows, n_cols) > prob] = -1
    Y_obs = Y_true.copy()
    Y_obs[~Omega] = 0

    return M_star, Omega, Y_obs, P_matrix