import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, json, itertools
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# 引入项目依赖
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import train_gd  # 🚨 这里我们只引入基础的 train_gd，不引入 CV 函数
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference
def solve_1bit_mc_fast(Omega_train, Y_obs, r, alpha):
    m, n = Y_obs.shape
    k = r + 1
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    scale = np.sqrt(0.95 * alpha / max(1e-8, np.max(np.abs(U @ V.T))))
    U *= scale; V *= scale
    U_res, V_res = train_gd(U, V, Omega_train, Y_obs, alpha, lam=0.001, max_iters=300)
    return U_res @ V_res.T

def run_overfitting_experiment(config):
    N, model_type, r_true, p, calib_ratio, noise = config
    
    # 把无穷范数约束从 1.0 放宽到 5.0，允许模型输出极端概率引发方差爆炸
    alpha_inf = 5.0  
    alpha_cp = 0.10 
    rank_range = [2, 3, 4, 5, 6, 8, 12] # 看 8 和 12 时的崩溃情况
    
    pid = os.getpid()
    print(f"[进程 {pid}] Phase 5 启动 | {model_type.upper()} | p={p} {noise}", flush=True)
    
    results_avg = {str(r): {'asym_cov':0, 'asym_len':0, 'cmc_cov':0, 'cmc_len':0} for r in rank_range}
    plot_data_r12 = {}
    
    for trial in range(3):
        M_star, Omega, Y_obs, P_matrix = generate_data(
            N, r_true, p, alpha_inf, noise_type=noise, model_type=model_type, seed=88+trial)
        
        omega_i, omega_j = np.nonzero(Omega)
        tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=calib_ratio, random_state=42)
        
        Omega_tr = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
        Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
        Omega_test = ~Omega

        for r_fit in rank_range:
            M_hat = solve_1bit_mc_fast(Omega_tr, Y_obs, r_fit, alpha_inf)
            
            # 由于重尾和弱惩罚，加上 Y_obs 和 r_fit 进行残差离差校正，作为Baseline
            S_matrix = compute_local_uncertainty(M_hat, Omega_tr, Y_obs=Y_obs, r_fit=r_fit)
            
            a_cov, a_len, a_L, a_U, _, _ = asymptotic_inference_chen2023(M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf)
            c_cov, c_len, c_L, c_U, h_vals, t_vals = generalized_cmc_inference(M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha_inf)
            
            results_avg[str(r_fit)]['asym_cov'] += a_cov / 3
            results_avg[str(r_fit)]['asym_len'] += a_len / 3
            results_avg[str(r_fit)]['cmc_cov'] += c_cov / 3
            results_avg[str(r_fit)]['cmc_len'] += c_len / 3
            
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
    file_name = f"res_Phase5_{model_type}_p{p}_calib{calib_ratio}_a{alpha_inf}_{noise}.json"
    os.makedirs('results/grid', exist_ok=True)
    with open(f"results/grid/{file_name}", 'w') as f:
        json.dump(res_dict, f, indent=4)
    print(f"[进程 {pid}] 测试完成: {file_name}", flush=True)
    return file_name

if __name__ == "__main__":
    print("\n>>> 开始执行 Phase 5: 弱正则化下的过拟合测试")
    N_list = [500] 
    model_type_list = ['general'] # 结构失配
    r_true = 5
    p_list = [0.3]                # 采用最佳稀疏度
    calib_list = [0.2]            # 8:2 切分
    noise_list = ['heavy_tail']   # 噪声失配
    
    valid_configs = list(itertools.product(N_list, model_type_list, [r_true], p_list, calib_list, noise_list))

    with Pool(processes=1) as pool:
        pool.map(run_overfitting_experiment, valid_configs)
        
    print("\n数据生成完毕")