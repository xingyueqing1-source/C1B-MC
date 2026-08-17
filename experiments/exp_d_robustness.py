import sys
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference

def run_exp_d():
    print(">>> 实验D: 边界测试")
    
    scenarios =[
        {'name': 'Scene1_Ideal_Logistic', 'N': 500, 'p': 0.8, 'noise': 'logistic'},
        {'name': 'Scene2_Misspecified_HeavyTail', 'N': 300, 'p': 0.5, 'noise': 'heavy_tail'}
    ]
    
    results =[]
    alpha_cp, alpha_inf, r = 0.10, 1.0, 5

    for sc in scenarios:
        print(f"\n--- 正在跑场景: {sc['name']} ---")
        M_star, Omega, Y_obs, P_matrix = generate_data(sc['N'], r, sc['p'], alpha_inf, noise_type=sc['noise'])
        
        omega_i, omega_j = np.nonzero(Omega)
        tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=0.2, random_state=42)
        
        Omega_tr = np.zeros_like(Omega)
        Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
        Omega_cal = np.zeros_like(Omega)
        Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
        Omega_test = ~Omega
        
        # 模型拟合与不确定度计算
        M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r, alpha_inf)
        S_matrix = compute_local_uncertainty(M_hat, Omega_tr)
        
        # 运行渐近推断
        asym_cov, asym_len, asym_L, asym_U, _, _ = asymptotic_inference_chen2023(
            M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf)
        
        # 运行 CMC 推断
        cmc_cov, cmc_len, cmc_L, cmc_U, hat_vals, true_vals = generalized_cmc_inference(
            M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha_inf)
        
        n_total_test = len(true_vals)
        num_plot = min(50, n_total_test)
        np.random.seed(42) # 固定种子保证每次画图点不变
        plot_idx = np.random.choice(n_total_test, num_plot, replace=False)
        
        results.append({
            'scenario': sc['name'],
            'metrics': {
                'asym_cov': asym_cov, 'asym_len': asym_len,
                'cmc_cov': cmc_cov, 'cmc_len': cmc_len
            },
            'plot_data': {
                'true_vals': true_vals[plot_idx].tolist(),
                'hat_vals': hat_vals[plot_idx].tolist(),
                'cmc_lower': cmc_L[plot_idx].tolist(),
                'cmc_upper': cmc_U[plot_idx].tolist(),
                'asym_lower': asym_L[plot_idx].tolist(),
                'asym_upper': asym_U[plot_idx].tolist()
            }
        })

    os.makedirs('results', exist_ok=True)
    with open('results/exp_d_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n[成功] 包含绘图点阵数据的 JSON 已保存至 results/exp_d_results.json")

if __name__ == "__main__":
    run_exp_d()