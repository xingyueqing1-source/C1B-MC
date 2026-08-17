import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, json
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# 确保能找到 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
# 引入两个求解器：一个是通用的，带CV，一个是纯正的Rasch，无偏MLE
from src.estimators import solve_1bit_mc_with_cv, solve_rasch_mle
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference


def run_server_samplesize_exp(config):
    model_type, p, calib_ratio, noise = config
    alpha_inf, alpha_cp = 1.0, 0.10 
    
    # R_true 控制数据生成。当 model_type=='rasch' 时，底层生成器会自动忽略它。
    r_true = 5 
    r_fit = 5
    N_list = [100, 200, 300, 500, 800, 1200, 1500] 
    pid = os.getpid()
    print(f"[服务器进程 {pid}] 启动: {model_type.upper()} + {noise}", flush=True)
    
    results = {str(n): {'asym_cov':0, 'asym_len':0, 'cmc_cov':0, 'cmc_len':0} for n in N_list}
    
    for n in N_list:
        n_rows = n
        n_cols = int(n / 1.5)
        # 每个 N 跑 3 次独立重复实验求平均，让相变曲线绝对平滑
        for trial in range(3):
            M_star, Omega, Y_obs, P_matrix = generate_data(
                n, n, r_true, p, alpha_inf, noise_type=noise, model_type=model_type, seed=42+trial
            )
            omega_i, omega_j = np.nonzero(Omega)
            tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=calib_ratio, random_state=42+trial)
            
            Omega_tr = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
            Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
            Omega_test = ~Omega
            if model_type == 'rasch':
                # 在主场，使用纯正的无偏 Rasch MLE (L-BFGS-B + Bounds)，不加 L2 惩罚
                M_hat = solve_rasch_mle(Omega_tr, Y_obs, alpha_inf)
            else:
                # 在客场，使用通用的 Rank-r 非凸交替下降 + CV
                M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r_fit, alpha_inf)
            
            # 按照 23Chen 公式计算局部不确定度，不加 r_fit 缩放
            S_matrix = compute_local_uncertainty(M_hat, Omega_tr)
            
            a_cov, a_len, _, _, _, _ = asymptotic_inference_chen2023(
                M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf)
                
            c_cov, c_len, _, _, _, _ = generalized_cmc_inference(
                M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha_inf)
            
            results[str(n)]['asym_cov'] += a_cov / 3
            results[str(n)]['asym_len'] += a_len / 3
            results[str(n)]['cmc_cov'] += c_cov / 3
            results[str(n)]['cmc_len'] += c_len / 3
            
        print(f"  -> [进程 {pid}] N={n} 跑批完成", flush=True)

    res_dict = {
        'params': {'model_type': model_type, 'r_fit': r_fit, 'p': p, 'calib_ratio': calib_ratio, 'noise': noise, 'alpha_inf': alpha_inf},
        'N_list': N_list,
        'data': results
    }
    
    file_name = f"res_Server_SampleSize_{model_type}_{noise}.json"
    os.makedirs('results/grid', exist_ok=True)
    with open(f"results/grid/{file_name}", 'w') as f:
        json.dump(res_dict, f, indent=4)
    return file_name


if __name__ == "__main__":
    configs = [
        # 场景 A: 23Chen主场 ，验证无偏状态下大数定律的有效性
        ('rasch', 0.2, 0.2, 'logistic'),     
        # 场景 B: 验证结构失配+重尾下 23Chen 崩溃，CMC 稳健
        ('general', 0.2, 0.2, 'heavy_tail')  
    ]
    
    print(f"\n多进程并行...")
    with Pool(processes=2) as pool:
        pool.map(run_server_samplesize_exp, configs)
        
    print("\n服务器运算完成！")