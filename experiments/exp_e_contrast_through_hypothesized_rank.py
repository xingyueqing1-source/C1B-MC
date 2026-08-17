import sys, os, json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023, generalized_cmc_inference

def run_cmc_style_experiment():
    print(">>> 开始执行顶会风格实验: 遍历 Hypothesized Rank")
    
    # 实验设定
    scenarios =[
        {'name': 'Setting 1: Logistic Noise', 'noise': 'logistic'},
        {'name': 'Setting 2: Heavy-tailed Noise', 'noise': 'heavy_tail'}
    ]
    
    N, r_true, p, alpha_inf, alpha_cp = 500, 5, 0.2, 1.0, 0.10
    rank_range =[2, 3, 4, 5, 6, 7, 8, 10, 12] # 假设的秩 (X轴)
    
    all_results =[]

    for sc in scenarios:
        print(f"\n=== 正在运行: {sc['name']} ===")
        # 1. 生成唯一真实的底层数据
        M_star, Omega, Y_obs, P_matrix = generate_data(N, r_true, p, alpha_inf, noise_type=sc['noise'], seed=42)
        
        # 划分数据集
        omega_i, omega_j = np.nonzero(Omega)
        tr_idx, calib_idx = train_test_split(np.arange(len(omega_i)), test_size=0.2, random_state=42)
        Omega_tr = np.zeros_like(Omega); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
        Omega_cal = np.zeros_like(Omega); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
        Omega_test = ~Omega
        
        scene_data = {'scenario': sc['name'], 'ranks': rank_range, 'data': {}}
        
        for r_fit in rank_range:
            print(f"  -> 拟合 Hypothesized Rank: {r_fit}...")
            # 模型拟合
            M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r_fit, alpha_inf)
            S_matrix = compute_local_uncertainty(M_hat, Omega_tr)
            
            # 渐近推断
            asym_cov, asym_len, asym_L, asym_U, _, _ = asymptotic_inference_chen2023(
                M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf)
            
            # CMC 推断
            cmc_cov, cmc_len, cmc_L, cmc_U, hat_vals, true_vals = generalized_cmc_inference(
                M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha_inf)
            
            # 提取 Figure 2 需要的标准化残差
            test_i, test_j = np.nonzero(Omega_test)
            scores = ((M_hat[test_i, test_j] - M_star[test_i, test_j]) / S_matrix[test_i, test_j]).tolist()
            
            # 抽取 50 个点
            np.random.seed(42)
            plot_idx = np.random.choice(len(true_vals), min(50, len(true_vals)), replace=False)
            
            scene_data['data'][str(r_fit)] = {
                'asym_cov': asym_cov, 'asym_len': asym_len,
                'cmc_cov': cmc_cov, 'cmc_len': cmc_len,
                'scores': scores, # 所有的得分，用于画直方图
                'pointwise': {    # 50个点的数据
                    'true': true_vals[plot_idx].tolist(),
                    'asym_L': asym_L[plot_idx].tolist(),
                    'asym_U': asym_U[plot_idx].tolist(),
                    'cmc_L': cmc_L[plot_idx].tolist(),
                    'cmc_U': cmc_U[plot_idx].tolist()
                }
            }
        all_results.append(scene_data)

    os.makedirs('results', exist_ok=True)
    with open('results/exp_e_cmc_style.json', 'w') as f:
        json.dump(all_results, f)
    print("\n实验完成，数据已保存")

if __name__ == "__main__":
    run_cmc_style_experiment()