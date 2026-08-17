import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, json
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference

def run_sparsity_exp(config):
    model_type, n_rows, n_cols, calib_ratio, noise = config
    alpha_inf, alpha_cp = 1.0, 0.10 
    r_true = 2 if model_type == 'rasch' else 5
    r_fit = r_true 
    
    # X轴：观测率 p
    p_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8] 
    
    pid = os.getpid()
    print(f"[进程 {pid}] 启动: {n_rows}x{n_cols} 稀疏度相变 | {model_type.upper()} + {noise}", flush=True)
    
    results = {str(p): {'asym_cov':0, 'asym_len':0, 'cmc_cov':0, 'cmc_len':0} for p in p_list}
    
    for p in p_list:
        for trial in range(3): # 跑 3 次平均
            M_star, Omega, Y_obs, P_matrix = generate_data(n_rows, n_cols, r_true, p, alpha_inf, noise_type=noise, model_type=model_type, seed=42+trial)
            omega_i, omega_j = np.nonzero(Omega)
            
            # 数据极少时直接跳过以防报错
            if len(omega_i) < 50: continue 
            
            tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=calib_ratio, random_state=42+trial)
            
            Omega_tr = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
            Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
            Omega_test = ~Omega
            
            # baseline：严格 CV 点估计 + Fisher 渐近方差
            M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r_fit, alpha_inf)
            S_matrix = compute_local_uncertainty(M_hat, Omega_tr, Y_obs=Y_obs, r_fit=r_fit)
            
            a_cov, a_len, _, _, _, _ = asymptotic_inference_chen2023(M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf)
            c_cov, c_len, _, _, _, _ = generalized_cmc_inference(M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha_inf)
            
            results[str(p)]['asym_cov'] += a_cov / 3
            results[str(p)]['asym_len'] += a_len / 3
            results[str(p)]['cmc_cov'] += c_cov / 3
            results[str(p)]['cmc_len'] += c_len / 3
            
        print(f"  -> [进程 {pid}] 稀疏度 p={p} 完成", flush=True)

    res_dict = {
        'params': {'model_type': model_type, 'n_rows': n_rows, 'n_cols': n_cols, 'r_fit': r_fit, 'calib_ratio': calib_ratio, 'noise': noise, 'alpha_inf': alpha_inf},
        'p_list': p_list,
        'data': results
    }
    
    file_name = f"res_Sparsity_{model_type}_{noise}.json"
    os.makedirs('results/grid', exist_ok=True)
    with open(f"results/grid/{file_name}", 'w') as f:
        json.dump(res_dict, f, indent=4)
    return file_name

if __name__ == "__main__":
    configs = [
        ('rasch', 600, 400, 0.2, 'logistic'),     
        ('general', 600, 400, 0.2, 'heavy_tail')  
    ]
    print(f"\n启动矩阵观测率相变测试...")
    with Pool(processes=2) as pool:
        pool.map(run_sparsity_exp, configs)
    print("\n观测率相变数据生成完毕！")