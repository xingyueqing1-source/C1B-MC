"""
Appendix experiment: Convex (max-norm) vs Non-convex (factorization) comparison.
Compares coverage and efficiency between the non-convex MLE (used in main text)
and a convex relaxation approach for the 1-bit MC point estimator.
"""
import sys, os, json
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import (compute_local_uncertainty, asymptotic_inference_chen2023,
                           generalized_cmc_inference)


def solve_convex_maxnorm(Omega_tr, Y_obs, alpha, lam=0.1, max_iters=200):
    """Convex relaxation: max-norm constrained logistic regression.

    Parameterizes M = U V^T with nuclear norm regularization proxy.
    Uses L-BFGS-B on the full M matrix with bound constraint |M_ij| <= alpha.

    This is a simplified convex proxy. A full max-norm approach would require
    semidefinite programming.
    """
    m, n = Y_obs.shape
    nnz = int(np.sum(Omega_tr))

    def objective_and_grad(vec_M):
        M = vec_M.reshape(m, n)
        Y_omega, M_omega = Y_obs[Omega_tr], M[Omega_tr]
        q = np.clip(-Y_omega * M_omega, -50, 50)
        nll = np.sum(np.log1p(np.exp(q)))
        reg = lam * np.sum(M ** 2)  # Frobenius regularization proxy

        grad_nll = np.zeros(m * n)
        grad_omega = -Y_omega / (1.0 + np.exp(np.clip(Y_omega * M_omega, -50, 50)))
        grad_M = np.zeros((m, n))
        grad_M[Omega_tr] = grad_omega
        grad_M += 2 * lam * M
        return nll + reg, grad_M.flatten()

    x0 = np.zeros(m * n)
    bounds = [(-alpha, alpha)] * (m * n)
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iters, 'disp': False})
    M_hat = res.x.reshape(m, n)
    return M_hat


def run_convex_comparison():
    print(">>> Convex vs Non-convex 1-bit MC Comparison")

    N, r_true, p = 300, 5, 0.2
    alpha_inf, alpha_cp = 1.0, 0.10

    M_star, Omega, Y_obs, P_matrix = generate_data(N, N, r_true, p, alpha_inf,
                                                     noise_type='heavy_tail', seed=42)
    omega_i, omega_j = np.nonzero(Omega)
    tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=0.2, random_state=42)
    Omega_tr  = np.zeros_like(Omega, dtype=bool); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
    Omega_cal = np.zeros_like(Omega, dtype=bool); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
    Omega_test = ~Omega

    results = {}

    # --- Non-convex (factorization) ---
    print("  [1/2] Non-convex factorization...")
    M_hat_nc = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r=r_true, alpha=alpha_inf)
    S_nc = compute_local_uncertainty(M_hat_nc, Omega_tr)

    a_cov, a_len, _, _, _, _ = asymptotic_inference_chen2023(M_hat_nc, S_nc, M_star, Omega_test, alpha_cp, alpha_inf)
    c_cov, c_len, _, _, _, _, _, _ = generalized_cmc_inference(M_hat_nc, S_nc, M_star, Omega_cal, Omega_test,
                                                                 P_matrix, alpha_cp, alpha_inf,
                                                                 score_type='normalized_residual', weight_type='odds_ratio')
    results['nonconvex'] = {'asym_cov': a_cov, 'asym_len': a_len, 'cmc_cov': c_cov, 'cmc_len': c_len,
                             'frob_err': float(np.linalg.norm(M_hat_nc - M_star, 'fro') / np.linalg.norm(M_star, 'fro'))}

    # --- Convex (max-norm proxy) ---
    print("  [2/2] Convex max-norm proxy...")
    M_hat_cv = solve_convex_maxnorm(Omega_tr, Y_obs, alpha=alpha_inf, lam=0.1, max_iters=300)
    S_cv = compute_local_uncertainty(M_hat_cv, Omega_tr)

    a_cov2, a_len2, _, _, _, _ = asymptotic_inference_chen2023(M_hat_cv, S_cv, M_star, Omega_test, alpha_cp, alpha_inf)
    c_cov2, c_len2, _, _, _, _, _, _ = generalized_cmc_inference(M_hat_cv, S_cv, M_star, Omega_cal, Omega_test,
                                                                   P_matrix, alpha_cp, alpha_inf,
                                                                   score_type='normalized_residual', weight_type='odds_ratio')
    results['convex'] = {'asym_cov': a_cov2, 'asym_len': a_len2, 'cmc_cov': c_cov2, 'cmc_len': c_len2,
                          'frob_err': float(np.linalg.norm(M_hat_cv - M_star, 'fro') / np.linalg.norm(M_star, 'fro'))}

    os.makedirs('results', exist_ok=True)
    with open('results/exp_convex_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n=== Results ===")
    for method, r in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Frobenius error: {r['frob_err']:.4f}")
        print(f"  C1B-MC coverage: {r['cmc_cov']:.4f}, length: {r['cmc_len']:.4f}")
        print(f"  Asym coverage:   {r['asym_cov']:.4f}, length: {r['asym_len']:.4f}")
    print("\n[OK] Results saved to results/exp_convex_comparison.json")


if __name__ == "__main__":
    run_convex_comparison()
