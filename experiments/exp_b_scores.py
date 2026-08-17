import sys, os, json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, generalized_cmc_inference


def run_exp_b():
    """实验B: 对比不同 nonconformity score 对 CMC 推断的影响"""
    print(">>> 开始执行实验B: Score Function 对比")

    configs = [
        {'N': 500, 'p': 0.1, 'noise': 'heavy_tail', 'label': 'Sparse + HeavyTail'},
        {'N': 500, 'p': 0.2, 'noise': 'logistic',   'label': 'Moderate + Logistic'},
    ]

    scores_to_test = ['absolute_residual', 'normalized_residual']
    alpha_inf, alpha_cp, r_true = 1.0, 0.10, 5
    all_results = []

    for cfg in configs:
        print(f"\n--- 场景: {cfg['label']} ---")
        M_star, Omega, Y_obs, P_matrix = generate_data(
            cfg['N'], cfg['N'], r_true, cfg['p'], alpha_inf, noise_type=cfg['noise'], seed=42
        )

        omega_i, omega_j = np.nonzero(Omega)
        tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=0.2, random_state=42)

        Omega_tr  = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
        Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
        Omega_test = ~Omega

        M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r_true, alpha_inf)
        S_matrix = compute_local_uncertainty(M_hat, Omega_tr)

        scene = {'label': cfg['label'], 'params': cfg, 'scores': {}}
        for s_type in scores_to_test:
            cov, length, _, _, _, _ = generalized_cmc_inference(
                M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix,
                alpha_cp=alpha_cp, alpha_inf=alpha_inf, score_type=s_type, weight_type='odds_ratio'
            )
            scene['scores'][s_type] = {'coverage': cov, 'length': length}
        all_results.append(scene)

    os.makedirs('results', exist_ok=True)
    with open('results/exp_b_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n[成功] 实验B完成，结果保存至 results/exp_b_results.json")


if __name__ == "__main__":
    run_exp_b()
