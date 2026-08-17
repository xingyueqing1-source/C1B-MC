import numpy as np
from sklearn.model_selection import KFold
from scipy.special import expit
from scipy.optimize import minimize

# --- 内部辅助函数 ---
def calc_obj_and_grad_logistic(U, V, Omega_mask, Y, alpha, lam):
    Z = U @ V.T
    Y_omega, Z_omega = Y[Omega_mask], Z[Omega_mask]
    q = -Y_omega * Z_omega
    q_clipped = np.clip(q, -50, 50)
    nll_val = np.sum(np.log1p(np.exp(q_clipped)))
    grad_Z_omega = -Y_omega / (1.0 + np.exp(np.clip(Y_omega * Z_omega, -50, 50)))
    grad_Z_nll = np.zeros_like(Z)
    grad_Z_nll[Omega_mask] = grad_Z_omega
    Z_sq, alpha_sq, eps = Z ** 2, alpha ** 2, 1e-12
    barrier_matrix = -np.log(np.maximum(1 - Z_sq / alpha_sq, eps))
    barrier_val = np.sum(barrier_matrix)
    grad_Z_barrier = (2 * Z) / np.maximum(alpha_sq - Z_sq, eps)
    grad_Z_total = grad_Z_nll + lam * grad_Z_barrier
    return nll_val + lam * barrier_val, nll_val, grad_Z_total @ V, grad_Z_total.T @ U

def backtracking_line_search(U, V, grad_U, grad_V, total_obj, Omega_mask, Y, alpha, lam):
    t, beta, c1 = 1.0, 0.5, 1e-4
    grad_norm_sq = np.sum(grad_U ** 2) + np.sum(grad_V ** 2)
    while t > 1e-8:
        U_new, V_new = U - t * grad_U, V - t * grad_V
        if np.max(np.abs(U_new @ V_new.T)) >= alpha - 1e-6:
            t *= beta
            continue
        new_obj, _, _, _ = calc_obj_and_grad_logistic(U_new, V_new, Omega_mask, Y, alpha, lam)
        if new_obj <= total_obj - c1 * t * grad_norm_sq:
            return U_new, V_new, new_obj, t
        t *= beta
    return U, V, total_obj, 0.0

def train_gd(U_init, V_init, Omega_mask, Y, alpha, lam, max_iters=200):
    U, V = U_init.copy(), V_init.copy()
    obj_val, _, grad_U, grad_V = calc_obj_and_grad_logistic(U, V, Omega_mask, Y, alpha, lam)
    for _ in range(max_iters):
        U_new, V_new, obj_new, step = backtracking_line_search(U, V, grad_U, grad_V, obj_val, Omega_mask, Y, alpha, lam)
        if step == 0.0 or (obj_val - obj_new) / abs(obj_val) < 1e-5: break
        U, V, obj_val = U_new, V_new, obj_new
        _, _, grad_U, grad_V = calc_obj_and_grad_logistic(U, V, Omega_mask, Y, alpha, lam)
    return U, V

# --- 导出的公共接口 ---
def solve_1bit_mc_with_cv(Omega_train, Y_obs, r, alpha):
    """现行baseline: Log-barrier MLE"""
    m, n = Y_obs.shape
    k = r + 1
    lambdas =[5.0 / (2 ** i) for i in range(8)]
    omega_i, omega_j = np.nonzero(Omega_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    avg_val_errors =[]
    for lam in lambdas:
        val_errors =[]
        for tr_idx, v_idx in kf.split(omega_i):
            tr_mask = np.zeros_like(Omega_train)
            tr_mask[omega_i[tr_idx], omega_j[tr_idx]] = True
            v_mask = np.zeros_like(Omega_train)
            v_mask[omega_i[v_idx], omega_j[v_idx]] = True
            U_init, V_init = np.random.randn(m, k), np.random.randn(n, k)
            scale = np.sqrt(0.95 * alpha / max(1e-8, np.max(np.abs(U_init @ V_init.T))))
            U_init *= scale; V_init *= scale
            U_res, V_res = train_gd(U_init, V_init, tr_mask, Y_obs, alpha, lam, max_iters=100)
            _, val_nll, _, _ = calc_obj_and_grad_logistic(U_res, V_res, v_mask, Y_obs, alpha, lam)
            val_errors.append(val_nll)
        avg_val_errors.append(np.mean(val_errors))

    U, V = np.random.randn(m, k), np.random.randn(n, k)
    scale = np.sqrt(0.95 * alpha / max(1e-8, np.max(np.abs(U @ V.T))))
    U *= scale; V *= scale
    for lam in lambdas[:np.argmin(avg_val_errors) + 1]:
        U, V = train_gd(U, V, Omega_train, Y_obs, alpha, lam, max_iters=200)
    return U @ V.T

def solve_rasch_mle(Omega_train, Y_obs, alpha):
    """
    100% 纯正无偏的 Rasch 极大似然估计器，对齐 23Chen 理论
    优化目标: 负对数似然, 无任何正则化惩罚
    """
    N, J = Y_obs.shape
    Y_mask = Y_obs[Omega_train]
    
    def obj_and_grad(params):
        theta = params[:N]
        beta = params[N:N+J]
        
        M = theta[:, None] - beta[None, :]
        M_mask = M[Omega_train]
        
        q = -Y_mask * M_mask
        q_clipped = np.clip(q, -50, 50) # 仅防止 np.exp 溢出，不影响数学值
        
        # 纯粹的负对数似然，不加任何 L2/L1 惩罚
        loss = np.sum(np.log1p(np.exp(q_clipped))) 
        
        # 解析梯度
        grad_M_mask = -Y_mask * expit(q_clipped)
        grad_M = np.zeros((N, J))
        grad_M[Omega_train] = grad_M_mask
        
        grad_theta = np.sum(grad_M, axis=1)
        grad_beta = -np.sum(grad_M, axis=0)
        
        return loss, np.concatenate([grad_theta, grad_beta])

    # 根据 23Chen 论文，使用边界限制参数空间
    bnds = [(-alpha, alpha)] * (N + J)
    
    # 使用 L-BFGS-B 求解，设置高容差确保收敛到最优点
    res = minimize(obj_and_grad, np.zeros(N+J), jac=True, method='L-BFGS-B', 
                   bounds=bnds, options={'ftol': 1e-9, 'maxiter': 2000})
    
    theta_opt = res.x[:N]
    beta_opt = res.x[N:N+J]
    M_hat = theta_opt[:, None] - beta_opt[None, :]
    
    return M_hat

def solve_mmgn(Omega_train, Y_obs, r, alpha):
    """占位: 留给后续接入 MMGN 算法"""
    raise NotImplementedError("MMGN solver is pending implementation.")

def solve_convex_svt(Omega_train, Y_obs, r, alpha):
    """占位: 留给后续接入 Convex Relaxation 算法"""
    raise NotImplementedError("Convex SVT solver is pending implementation.")