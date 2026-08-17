import torch, sys, os, json
import numpy as np
from scipy.stats import norm
from scipy.special import expit
from sklearn.model_selection import KFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_data
from src.inference import generalized_cmc_inference

def torch_estimator(Y_pm1, Omega_mask, r, alpha=1.0, lam=1.0, lr=0.06, max_iters=300):
    m, n = Omega_mask.shape
    U = torch.randn(m, r, device=device) * 0.1
    V = torch.randn(n, r, device=device) * 0.1
    U.requires_grad_(True); V.requires_grad_(True)
    optimizer = torch.optim.Adam([U, V], lr=lr)

    Y_t = torch.as_tensor(Y_pm1, device=device, dtype=torch.float32)
    mask_t = torch.as_tensor(Omega_mask, device=device, dtype=torch.bool)

    for _ in range(max_iters):
        optimizer.zero_grad()
        Z = U @ V.T
        nll = torch.sum(torch.log1p(torch.exp(-Y_t[mask_t] * Z[mask_t])))
        barrier = -lam * torch.sum(torch.log(torch.clamp(1.0 - (Z ** 2) / (alpha ** 2), min=1e-8)))
        loss = nll + barrier
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            U.clamp_(-alpha, alpha)
            V.clamp_(-alpha, alpha)
        if loss.item() < 1e-4: break
    return (U @ V.T).detach().cpu().numpy()

def split_conformal_inference(M_hat, S_matrix, M_star, Omega_cal, Omega_test, alpha_cp=0.10):
    cal_i, cal_j = np.nonzero(Omega_cal)
    test_i, test_j = np.nonzero(Omega_test)
    cal_scores = np.abs(M_hat[cal_i, cal_j] - M_star[cal_i, cal_j]) / (S_matrix[cal_i, cal_j] + 1e-8)
    n_cal = len(cal_scores)
    q_idx = min(int(np.ceil((1.0 - alpha_cp) * (n_cal + 1))) - 1, n_cal - 1)
    Q = np.sort(cal_scores)[q_idx]
    M_test_hat, M_test_true = M_hat[test_i, test_j], M_star[test_i, test_j]
    S_test = S_matrix[test_i, test_j]
    lower, upper = M_test_hat - Q * S_test, M_test_hat + Q * S_test
    covered = (M_test_true >= lower) & (M_test_true <= upper)
    return float(np.mean(covered)), float(np.mean(upper - lower))

def cvplus_conformal(Y_pm1, M_star, Omega, r_fit, alpha=1.0, lam=1.0, alpha_cp=0.10, k_folds=5, n_bootstrap=150, lr=0.06):
    obs_i, obs_j = np.nonzero(Omega)
    n_obs = len(obs_i)
    M_hat_full = torch_estimator(Y_pm1, Omega, r=r_fit, alpha=alpha, lam=lam, lr=lr)

    prob_hat = expit(M_hat_full)
    sigma2_tr = np.clip(prob_hat * (1.0 - prob_hat), 1e-4, 0.25) * Omega
    S_matrix = np.sqrt(1.0 / (sigma2_tr.sum(axis=1, keepdims=True) + 1e-6) + 1.0 / (sigma2_tr.sum(axis=0, keepdims=True) + 1e-6))

    ensemble = []
    for _ in range(n_bootstrap):
        b_idx = np.random.choice(n_obs, n_obs, replace=True)
        Omega_b = np.zeros_like(Omega, dtype=bool); Omega_b[obs_i[b_idx], obs_j[b_idx]] = True
        ensemble.append(torch_estimator(Y_pm1, Omega_b, r=r_fit, alpha=alpha, lam=lam, lr=lr, max_iters=150))
    H_boot = np.std(np.stack(ensemble, axis=0), axis=0) + 1e-4

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_scores = {'CQR': [], 'Fisher': [], 'Bootstrap': []}

    for tr_idx, val_idx in kf.split(np.arange(n_obs)):
        O_tr = np.zeros_like(Omega, dtype=bool); O_tr[obs_i[tr_idx], obs_j[tr_idx]] = True
        M_k = torch_estimator(Y_pm1, O_tr, r=r_fit, alpha=alpha, lam=lam, lr=lr, max_iters=200)
        vi, vj = obs_i[val_idx], obs_j[val_idx]
        
        prob_k = expit(M_k)
        s2_tr_k = np.clip(prob_k * (1.0 - prob_k), 1e-4, 0.25) * O_tr
        S_k = np.sqrt(1.0 / (s2_tr_k.sum(axis=1, keepdims=True) + 1e-6) + 1.0 / (s2_tr_k.sum(axis=0, keepdims=True) + 1e-6))

        all_scores['CQR'].extend(np.abs(M_k[vi, vj] - M_star[vi, vj]).tolist())
        all_scores['Fisher'].extend((np.abs(M_k[vi, vj] - M_star[vi, vj]) / (S_k[vi, vj] + 1e-8)).tolist())
        all_scores['Bootstrap'].extend((np.abs(M_k[vi, vj] - M_star[vi, vj]) / (H_boot[vi, vj] + 1e-8)).tolist())

    test_i, test_j = np.nonzero(~Omega)
    M_hat_test, M_star_test = M_hat_full[test_i, test_j], M_star[test_i, test_j]
    S_test, H_boot_test = S_matrix[test_i, test_j], H_boot[test_i, test_j]

    q_level = min((1.0 - alpha_cp) * (1.0 + 1.0 / n_obs), 1.0)
    results = {}
    for score_name, score_list in all_scores.items():
        Q = np.quantile(score_list, q_level)
        margin = Q if score_name == 'CQR' else (Q * S_test if score_name == 'Fisher' else Q * H_boot_test)
        lo, hi = M_hat_test - margin, M_hat_test + margin
        results[score_name] = {
            'coverage': float(np.mean((M_star_test >= lo) & (M_star_test <= hi))),
            'width': float(np.mean(hi - lo))
        }
    return results, M_hat_full, S_matrix

def run_single_trial(N, J, p_obs, r_true, r_fit, alpha, alpha_cp, scenario, seed):
    torch.manual_seed(seed); np.random.seed(seed)

    if scenario == 'RASCH':
        M_star, Omega, Y_obs, P_matrix = generate_data(N, J, r_true, p_obs, alpha, noise_type='logistic', model_type='rasch', seed=seed)
    else:
        M_star, Omega, Y_obs, P_matrix = generate_data(N, J, r_true, p_obs, alpha, noise_type='heavy_tail', model_type='general', seed=seed)

    Y_pm1 = np.where(Y_obs == 1, 1, -1).astype(float)
    Y_pm1[~Omega] = 0 

    obs_i, obs_j = np.nonzero(Omega); n_obs = len(obs_i)
    tr_idx, cal_test_idx = train_test_split(np.arange(n_obs), test_size=0.4, random_state=seed)
    cal_idx, test_obs_idx = train_test_split(cal_test_idx, test_size=0.5, random_state=seed)

    Omega_tr = np.zeros_like(Omega, dtype=bool); Omega_tr[obs_i[tr_idx], obs_j[tr_idx]] = True
    Omega_cal = np.zeros_like(Omega, dtype=bool); Omega_cal[obs_i[cal_idx], obs_j[cal_idx]] = True
    Omega_test = ~Omega 

    trial_results = []

    # 1. Asymptotic
    M_hat_tr = torch_estimator(Y_pm1, Omega_tr, r=r_fit, alpha=alpha, lam=1.0, lr=0.06, max_iters=300)
    prob_hat = expit(M_hat_tr)
    sigma2_tr = np.clip(prob_hat * (1.0 - prob_hat), 1e-4, 0.25) * Omega_tr
    S_matrix = np.sqrt(1.0 / (sigma2_tr.sum(axis=1, keepdims=True) + 1e-6) + 1.0 / (sigma2_tr.sum(axis=0, keepdims=True) + 1e-6))

    test_i, test_j = np.nonzero(Omega_test)
    z = norm.ppf(1 - alpha_cp / 2.0)
    M_tt, M_th, S_t = M_star[test_i, test_j], M_hat_tr[test_i, test_j], S_matrix[test_i, test_j]
    asym_lo, asym_hi = np.maximum(M_th - z * S_t, -alpha), np.minimum(M_th + z * S_t, alpha)
    trial_results.append({'Method': 'Asymptotic', 'Coverage': float(np.mean((M_tt >= asym_lo) & (M_tt <= asym_hi))), 'Width': float(np.mean(asym_hi - asym_lo))})

    # 2. Split-CP
    sp_cov, sp_wid = split_conformal_inference(M_hat_tr, S_matrix, M_star, Omega_cal, Omega_test, alpha_cp)
    trial_results.append({'Method': 'Split-CP', 'Coverage': sp_cov, 'Width': sp_wid})

    # 3. C1B-MC
    cmc_cov, cmc_wid, *_ = generalized_cmc_inference(
        M_hat_tr, S_matrix, M_star, Omega_cal, Omega_test, P_matrix, alpha_cp, alpha, score_type='normalized_residual', weight_type='odds_ratio')
    trial_results.append({'Method': 'C1B-MC', 'Coverage': cmc_cov, 'Width': cmc_wid})

    # 4. CV+
    cv_res, _, _ = cvplus_conformal(Y_pm1, M_star, Omega, r_fit=r_fit, alpha=alpha, lam=1.0, alpha_cp=alpha_cp, k_folds=5, n_bootstrap=150, lr=0.06)
    for score_name in ['CQR', 'Fisher', 'Bootstrap']:
        trial_results.append({'Method': f'CV+ {score_name}', 'Coverage': cv_res[score_name]['coverage'], 'Width': cv_res[score_name]['width']})

    for r in trial_results:
        r['scenario'] = scenario; r['N'] = N; r['J'] = J; r['p_obs'] = p_obs; r['trial'] = seed % 1000

    return trial_results

def main():
    N, J = 500, 500
    r_true, r_fit, alpha, alpha_cp = 5, 5, 1.0, 0.10
    n_trials = 20
    p_scan = [0.2, 0.4, 0.6, 0.8]
    scenarios = [
        ('RASCH',    'rasch',   'logistic',   'Rasch Model + Logistic Noise'),
        # ('HIGH_RANK', 'general', 'heavy_tail', 'General Model + Heavy-tailed Noise'),
    ]

    out_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    # 生成全新的数据文件
    json_path = os.path.join(out_dir, 'boxplot_benchmark_PERFECT.json')
    all_results = []

    for scenario_key, model_type, noise, label in scenarios:
        print(f"\n{'='*50}\n>>> SCENARIO: {label}\n{'='*50}")
        for p_obs in p_scan:
            print(f"  p={p_obs}: 正在运行 {n_trials} 次 Trial...")
            for t in range(n_trials):
                seed = 42 + t + int(p_obs * 100)
                trial_res = run_single_trial(N, J, p_obs, r_true, r_fit, alpha, alpha_cp, scenario_key, seed)
                all_results.extend(trial_res)

            # 每跑完一个 P，就存一次盘，防止意外中断
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    print(f"\n全部实验完成，数据已保存至 {json_path}")

if __name__ == "__main__":
    main()