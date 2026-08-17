"""Generate standardized score data for Figure 4 (score distribution plot)."""
import sys, os, json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference

def generate_and_save(N, r_true, p, alpha_inf, model_type, noise, out_path, r_fit=12):
    print(f"  Generating: N={N}, {model_type}, {noise}, r_fit={r_fit}...")
    M_star, Omega, Y_obs, P_matrix = generate_data(N, N, r_true, p, alpha_inf, noise_type=noise,
                                                     model_type=model_type, seed=42)
    omega_i, omega_j = np.nonzero(Omega)
    tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=0.2, random_state=42)
    Omega_tr  = np.zeros_like(Omega, dtype=bool); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
    Omega_cal = np.zeros_like(Omega, dtype=bool); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
    Omega_test = ~Omega

    M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r=r_fit, alpha=alpha_inf)
    S_matrix = compute_local_uncertainty(M_hat, Omega_tr)

    # Compute scores for test set
    test_i, test_j = np.nonzero(Omega_test)
    n_sample = min(5000, len(test_i))
    sample_idx = np.random.RandomState(42).choice(len(test_i), n_sample, replace=False)

    scores = (M_hat[test_i, test_j][sample_idx] - M_star[test_i, test_j][sample_idx]) / (S_matrix[test_i, test_j][sample_idx] + 1e-8)

    # Also run full inference for 100 pointwise entries
    n_pt = min(100, len(test_i))
    pt_idx = np.random.RandomState(42).choice(len(test_i), n_pt, replace=False)
    _, _, asym_L, asym_U, _, _ = asymptotic_inference_chen2023(M_hat, S_matrix, M_star, Omega_test, 0.10, alpha_inf)
    _, _, cmc_L, cmc_U, _, _, _, _ = generalized_cmc_inference(M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix,
                                                                  0.10, alpha_inf, score_type='normalized_residual',
                                                                  weight_type='odds_ratio')

    # Load existing JSON or create new
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[f'plot_data_r{r_fit}'] = {
        'scores': scores.tolist(),
        'pointwise': {
            'true': M_star[test_i, test_j][pt_idx].tolist(),
            'cmc_L': cmc_L[pt_idx].tolist(),
            'cmc_U': cmc_U[pt_idx].tolist(),
            'asym_L': asym_L[pt_idx].tolist(),
            'asym_U': asym_U[pt_idx].tolist(),
        }
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"    -> Saved to {out_path} (n_scores={len(scores)}, n_pointwise={n_pt})")


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(root, 'results', 'grid')

    # Generate for the two key configs used in Figure 4
    configs = [
        (500, 5, 0.2, 1.0, 'rasch',   'logistic',    'res_N500_rasch_p0.2_calib0.2_a1.0_logistic.json'),
        (500, 5, 0.2, 1.0, 'general', 'heavy_tail',  'res_N500_general_p0.2_calib0.2_a1.0_heavy_tail.json'),
    ]

    for N, r_true, p, alpha_inf, model_type, noise, fname in configs:
        out_path = os.path.join(results_dir, fname)
        generate_and_save(N, r_true, p, alpha_inf, model_type, noise, out_path, r_fit=12)

    print("\nDone. Plot data generated for all configs.")
