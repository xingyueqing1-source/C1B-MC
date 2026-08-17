import os
# 强制底层单线程，防止 CPU 死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, json, itertools
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference

def run_single_experiment(config):
    # 终极版：接收 7 个核心参数
    N, model_type, r_true, p, calib_ratio, noise, alpha_inf = config
    alpha_cp = 0.10 
    rank_range = [2, 3, 4, 5, 6, 8, 12] # X轴：假设的秩
    
    pid = os.getpid()
    print(f"[进程 {pid}] | {model_type.upper()} | N={N} p={p} calib={calib_ratio} {noise} alpha={alpha_inf}", flush=True)
    
    results_avg = {str(r): {'asym_cov':0, 'asym_len':0, 'cmc_cov':0, 'cmc_len':0} for r in rank_range}
    plot_data_r12 = {} # 保存 r=12 的微观画图数据
    
    # 每个配置跑 3 次取平均，消除随机波动
    for trial in range(3):
        M_star, Omega, Y_obs, P_matrix = generate_data(
            N, r_true, p, alpha_inf, noise_type=noise, model_type=model_type, seed=42+trial)
        
        omega_i, omega_j = np.nonzero(Omega)
        tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=calib_ratio, random_state=42)
        
        Omega_tr = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
        Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
        Omega_test = ~Omega
        
        for r_fit in rank_range:
            M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r_fit, alpha_inf)
            S_matrix = compute_local_uncertainty(M_hat, Omega_tr)
            
            a_cov, a_len, a_L, a_U, _, _ = asymptotic_inference_chen2023(M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf)
            c_cov, c_len, c_L, c_U, h_vals, t_vals = generalized_cmc_inference(M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha_inf)
            
            results_avg[str(r_fit)]['asym_cov'] += a_cov / 3
            results_avg[str(r_fit)]['asym_len'] += a_len / 3
            results_avg[str(r_fit)]['cmc_cov'] += c_cov / 3
            results_avg[str(r_fit)]['cmc_len'] += c_len / 3
            
            # 仅在 trial==0 且 r_fit==12 时保存微观数据
            if trial == 0 and r_fit == 12:
                np.random.seed(42)
                plot_idx = np.random.choice(len(t_vals), min(50, len(t_vals)), replace=False)
                test_i, test_j = np.nonzero(Omega_test)
                scores = ((M_hat[test_i, test_j] - M_star[test_i, test_j]) / S_matrix[test_i, test_j]).tolist()
                plot_data_r12 = {
                    'scores': scores,
                    'pointwise': {
                        'true': t_vals[plot_idx].tolist(), 'hat': h_vals[plot_idx].tolist(),
                        'cmc_L': c_L[plot_idx].tolist(), 'cmc_U': c_U[plot_idx].tolist(),
                        'asym_L': a_L[plot_idx].tolist(), 'asym_U': a_U[plot_idx].tolist()
                    }
                }

    res_dict = {
        'params': {'N': N, 'model_type': model_type, 'r_true': r_true, 'p': p, 'calib_ratio': calib_ratio, 'noise': noise, 'alpha_inf': alpha_inf},
        'data': results_avg,
        'plot_data_r12': plot_data_r12
    }
    
    file_name = f"res_N{N}_{model_type}_p{p}_calib{calib_ratio}_a{alpha_inf}_{noise}.json"
    os.makedirs('results/grid', exist_ok=True)
    with open(f"results/grid/{file_name}", 'w') as f:
        json.dump(res_dict, f, indent=4)
    return file_name

if __name__ == "__main__":
    # 控制台
    r_true = 5 # 固定为 5
    N_list = [500] 
    model_type_list = ['general']
    p_list = [0.3]
    calib_list = [0.1, 0.2, 0.3, 0.4, 0.5] # 10% 校准 ~  50% 校准
    noise_list = ['heavy_tail']
    alpha_inf_list = [1.0]
    
    raw_configs = list(itertools.product(N_list, model_type_list, [r_true], p_list, calib_list, noise_list, alpha_inf_list))
    
    valid_configs = [cfg for cfg in raw_configs if cfg[3] * (1 - cfg[4]) >= 0.08]
    
    print(f"\n开始执行 {len(valid_configs)} 组实验...")
    with Pool(processes=4) as pool: # 服务器上改 8
        pool.map(run_single_experiment, valid_configs)
    print("\n本次网格运算全部完成！")