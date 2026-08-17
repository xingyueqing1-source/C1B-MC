import sys, os, json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, generalized_cmc_inference


def run_exp_c():
    """实验C: 对比不同权重函数在协变量偏移下的表现"""
    print(">>> 开始执行实验C: Weight Function 对比")

    weights_to_test = ['uniform', 'odds_ratio']
    alpha_inf, alpha_cp, r_true, N = 1.0, 0.10, 5, 500
    all_results = []

    # 场景1: 均匀观测概率 (baseline, 无偏移)
    print("\n--- 场景1: Homogeneous，无协变量偏移 ---")
    M_star, Omega, Y_obs, P_matrix = generate_data(N, N, r_true, 0.2, alpha_inf, noise_type='heavy_tail', seed=42)
    omega_i, omega_j = np.nonzero(Omega)
    tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=0.2, random_state=42)
    Omega_tr  = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
    Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
    Omega_test = ~Omega

    M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r_true, alpha_inf)
    S_matrix = compute_local_uncertainty(M_hat, Omega_tr)

    homo_results = {'label': 'Homogeneous (p=0.2)', 'weights': {}}
    for w_type in weights_to_test:
        cov, length, _, _, _, _ = generalized_cmc_inference(
            M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix,
            alpha_cp=alpha_cp, alpha_inf=alpha_inf, score_type='normalized_residual', weight_type=w_type
        )
        homo_results['weights'][w_type] = {'coverage': cov, 'length': length}
    all_results.append(homo_results)

    # 场景2: 极端异质性 (前一半 p=0.6, 后一半 p=0.05)
    print("\n--- 场景2: Heterogeneous，协变量偏移 ---")
    P_hetero = np.zeros((N, N))
    P_hetero[:N//2, :] = 0.6
    P_hetero[N//2:, :] = 0.05
    Omega_hetero = np.random.rand(N, N) < P_hetero

    M_star2, _, Y_obs2, _ = generate_data(N, N, r_true, 0.2, alpha_inf, noise_type='heavy_tail', seed=42)
    Y_obs2[~Omega_hetero] = 0

    omega_i2, omega_j2 = np.nonzero(Omega_hetero)
    tr_idx2, calib_idx2 = train_test_split(np.arange(len(omega_i2)), test_size=0.2, random_state=42)
    Omega_tr2  = np.zeros_like(Omega_hetero); Omega_tr2[omega_i2[tr_idx2], omega_j2[tr_idx2]] = True
    Omega_cal2 = np.zeros_like(Omega_hetero); Omega_cal2[omega_i2[calib_idx2], omega_j2[calib_idx2]] = True
    Omega_test2 = ~Omega_hetero

    M_hat2 = solve_1bit_mc_with_cv(Omega_tr2, Y_obs2, r_true, alpha_inf)
    S_matrix2 = compute_local_uncertainty(M_hat2, Omega_tr2)

    hetero_results = {'label': 'Heterogeneous (p∈[0.05,0.6])', 'weights': {}}
    for w_type in weights_to_test:
        cov, length, _, _, _, _ = generalized_cmc_inference(
            M_hat2, S_matrix2, M_star2, Omega_cal2, Omega_test2, P_hetero,
            alpha_cp=alpha_cp, alpha_inf=alpha_inf, score_type='normalized_residual', weight_type=w_type
        )
        hetero_results['weights'][w_type] = {'coverage': cov, 'length': length}
    all_results.append(hetero_results)

    os.makedirs('results', exist_ok=True)
    with open('results/exp_c_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n[成功] 实验C完成，结果保存至 results/exp_c_results.json")


if __name__ == "__main__":
    run_exp_c()
