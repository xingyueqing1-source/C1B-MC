import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, json, itertools
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference

def run_single_experiment(config):
    N, r_true, p, calib_ratio, noise = config
    alpha_inf, alpha_cp = 1.0, 0.10 
    rank_range =[2, 3, 4, 5, 6, 8, 12, 15] 
    
    # 获取当前进程ID
    pid = os.getpid()
    
    results_avg = {str(r): {'asym_cov':0, 'asym_len':0, 'cmc_cov':0, 'cmc_len':0} for r in rank_range}
    
    for trial in range(3):
        print(f"[进程 {pid}] 启动 | N={N} p={p} calib={calib_ratio} {noise} | Trial {trial+1}/3", flush=True)
        
        M_star, Omega, Y_obs, P_matrix = generate_data(N, r_true, p, alpha_inf, noise_type=noise, seed=42+trial)
        omega_i, omega_j = np.nonzero(Omega)
        
        tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=calib_ratio, random_state=42)
        
        Omega_tr = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
        Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
        Omega_test = ~Omega
        
        for r_fit in rank_range:
            print(f"  -> [进程 {pid}] 拟合 r_fit={r_fit}...", flush=True)
            
            M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r_fit, alpha_inf)
            S_matrix = compute_local_uncertainty(M_hat, Omega_tr)
            
            a_cov, a_len, _, _, _, _ = asymptotic_inference_chen2023(M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf)
            c_cov, c_len, _, _, _, _ = generalized_cmc_inference(M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha_inf)
            
            results_avg[str(r_fit)]['asym_cov'] += a_cov / 3
            results_avg[str(r_fit)]['asym_len'] += a_len / 3
            results_avg[str(r_fit)]['cmc_cov'] += c_cov / 3
            results_avg[str(r_fit)]['cmc_len'] += c_len / 3

    # 保存结果
    res_dict = {
        'params': {'N': N, 'r_true': r_true, 'p': p, 'calib_ratio': calib_ratio, 'noise': noise},
        'data': results_avg
    }
    file_name = f"res_N{N}_p{p}_calib{calib_ratio}_{noise}.json"
    os.makedirs('results/grid', exist_ok=True)
    with open(f"results/grid/{file_name}", 'w') as f:
        json.dump(res_dict, f, indent=4)
        
    print(f"[进程 {pid}] 完成并保存: {file_name}", flush=True)
    return file_name


if __name__ == "__main__":
    # 网格搜索配置
    N_list = [500]
    p_list =[0.2]
    calib_ratio_list =[ 0.2]
    noise_list = ['logistic', 'heavy_tail']
    r_true = 5

    raw_configs = list(itertools.product(N_list, [r_true], p_list, calib_ratio_list, noise_list))
   
    valid_configs =[]
    for cfg in raw_configs:
        N, r, p, calib, noise = cfg
        # 训练集实际占比 = 总观测率 p * (1 - calib_ratio)
        train_density = p * (1 - calib)
        
        if train_density < 0.08:
            print(f"  去除无效配置: p={p}, calib={calib} (有效训练密度 {train_density:.2f} 过低)")
            continue
            
        valid_configs.append(cfg)
        
    print(f"\n过滤完毕：从 {len(raw_configs)} 组精简至 {len(valid_configs)} 组有效实验")
    print("开始并行计算...\n")
    
    with Pool(processes=4) as pool:
        pool.map(run_single_experiment, valid_configs)
        
    print("\n实验运行完毕！")