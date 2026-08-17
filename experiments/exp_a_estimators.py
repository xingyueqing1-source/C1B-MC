import numpy as np
import warnings

warnings.filterwarnings('ignore')

def calc_obj_and_grad_logistic(U, V, Omega_mask, Y, alpha, lam):
    Z = U @ V.T
    Y_omega, Z_omega = Y[Omega_mask], Z[Omega_mask]

    # Logistic Negative Log-Likelihood
    q = np.clip(-Y_omega * Z_omega, -50, 50)
    nll_val = np.sum(np.log1p(np.exp(q)))

    grad_Z_omega = -Y_omega / (1.0 + np.exp(np.clip(Y_omega * Z_omega, -50, 50)))
    grad_Z_nll = np.zeros_like(Z)
    grad_Z_nll[Omega_mask] = grad_Z_omega

    # Log-barrier Penalty
    Z_sq, alpha_sq = Z**2, alpha**2
    barrier_val = -np.sum(np.log(np.maximum(1 - Z_sq / alpha_sq, 1e-12)))
    grad_Z_barrier = (2 * Z) / np.maximum(alpha_sq - Z_sq, 1e-12)

    total_obj = nll_val + lam * barrier_val
    grad_Z = grad_Z_nll + lam * grad_Z_barrier

    return total_obj, grad_Z @ V, grad_Z.T @ U

def backtracking_line_search(U, V, grad_U, grad_V, total_obj, Omega_mask, Y, alpha, lam):
    t, beta, c1 = 1.0, 0.5, 1e-4
    grad_norm_sq = np.sum(grad_U ** 2) + np.sum(grad_V ** 2)

    while t > 1e-8:
        U_new = U - t * grad_U
        V_new = V - t * grad_V
        Z_new = U_new @ V_new.T

        if np.max(np.abs(Z_new)) >= alpha - 1e-6:
            t *= beta
            continue

        new_obj, _, _ = calc_obj_and_grad_logistic(U_new, V_new, Omega_mask, Y, alpha, lam)
        if new_obj <= total_obj - c1 * t * grad_norm_sq:
            return U_new, V_new, new_obj, t
        t *= beta

    return U, V, total_obj, 0.0

def train_gd(U, V, Omega_mask, Y, alpha, lam, max_iters=100):
    obj_val, gU, gV = calc_obj_and_grad_logistic(U, V, Omega_mask, Y, alpha, lam)
    for _ in range(max_iters):
        U_new, V_new, obj_new, step = backtracking_line_search(U, V, gU, gV, obj_val, Omega_mask, Y, alpha, lam)
        if step == 0.0 or (obj_val - obj_new) / abs(obj_val) < 1e-5:
            break
        U, V, obj_val = U_new, V_new, obj_new
        _, gU, gV = calc_obj_and_grad_logistic(U, V, Omega_mask, Y, alpha, lam)
    return U, V

def solve_1bit_mc_with_cv(Omega_tr, Y_obs, r, alpha):
    m, n = Y_obs.shape
    U, V = np.random.randn(m, r), np.random.randn(n, r)
    scale = np.sqrt(0.95 * alpha / max(1e-8, np.max(np.abs(U @ V.T))))
    U *= scale
    V *= scale
    return train_gd(U, V, Omega_tr, Y_obs, alpha, lam=1.0)