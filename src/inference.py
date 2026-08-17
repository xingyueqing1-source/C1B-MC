"""
Inference module for Conformalized 1-Bit Matrix Completion.
Provides: local uncertainty estimation, asymptotic inference (baseline),
weighted conformal inference, p-value computation, and real-data adaptations.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import expit


# ============================================================
# Stage 2: Local Uncertainty Quantification
# ============================================================

def compute_local_uncertainty(M_hat, Omega_tr, Y_obs=None, r_fit=None):
    """Chen (2023) Theorem 4: asymptotic standard error per entry.

    For the Rasch / low-rank logistic model, the pointwise Fisher information
    yields sigma2_ij = sigma(M_hat_ij) * (1 - sigma(M_hat_ij)).  The local
    standard error is the sqrt of inverse row + column aggregated information.

    Y_obs and r_fit are reserved for future extensions (e.g. residual deviance
    correction for the general low-rank model) but are currently unused.
    """
    prob_hat = expit(M_hat)
    sigma2_ij = prob_hat * (1.0 - prob_hat)
    sigma2_ij = np.clip(sigma2_ij, 1e-4, 0.25)

    sigma2_tr = sigma2_ij * Omega_tr
    sigma2_i_plus = np.sum(sigma2_tr, axis=1, keepdims=True) + 1e-6
    sigma2_plus_j = np.sum(sigma2_tr, axis=0, keepdims=True) + 1e-6

    S_matrix = np.sqrt(1.0 / sigma2_i_plus + 1.0 / sigma2_plus_j)
    return S_matrix


# ============================================================
# Baseline: Asymptotic Inference (Chen 2023)
# ============================================================

def asymptotic_inference_chen2023(M_hat, S_matrix, M_star, Omega_test, alpha_cp, alpha_inf):
    """Asymptotic z-intervals using plug-in Fisher information."""
    z_score = norm.ppf(1 - alpha_cp / 2.0)
    test_i, test_j = np.nonzero(Omega_test)
    n_test = len(test_i)

    M_test_hat = M_hat[test_i, test_j]
    M_test_true = M_star[test_i, test_j]
    S_test = S_matrix[test_i, test_j]

    lower_bounds = M_test_hat - z_score * S_test
    upper_bounds = M_test_hat + z_score * S_test
    lengths = upper_bounds - lower_bounds

    covered = (M_test_true >= lower_bounds) & (M_test_true <= upper_bounds)
    return float(np.mean(covered)), float(np.mean(lengths)), lower_bounds, upper_bounds, M_test_hat, M_test_true


# ============================================================
# Nonconformity Scores (with and without M*)
# ============================================================

def get_nonconformity_scores(M_calib_hat, M_calib_true, S_calib, score_type):
    """Nonconformity scores for simulation settings (M* known)."""
    if score_type == 'normalized_residual':
        return np.abs(M_calib_hat - M_calib_true) / (S_calib + 1e-8)
    elif score_type == 'absolute_residual':
        return np.abs(M_calib_hat - M_calib_true)
    else:
        raise NotImplementedError(f"Unknown score_type: {score_type}")


def get_nonconformity_scores_real(M_calib_hat, Y_calib, S_calib, score_type='hinge'):
    """Nonconformity scores for real data (M* unknown).

    Use observed binary labels Y directly:
      - 'hinge':      V_ij = -Y_ij * M_hat_ij   (binary classification margin)
      - 'abs_hinge':  V_ij = |M_hat_ij|          reflects prediction confidence

    The hinge score is natural: when Y_ij and M_hat_ij have same sign,
    the product is positive (good agreement), and we want large scores
    to indicate poor agreement.
    """
    if score_type == 'hinge':
        return -Y_calib * M_calib_hat
    elif score_type == 'abs_hinge':
        return -np.abs(M_calib_hat)
    elif score_type == 'normalized_hinge':
        return (-Y_calib * M_calib_hat) / (S_calib + 1e-8)
    else:
        raise NotImplementedError(f"Unknown real data score_type: {score_type}")


# ============================================================
# Conformal Weights
# ============================================================

def get_conformal_weights(P_calib, p_test, weight_type):
    """Compute calibration and test weights for weighted CP."""
    if weight_type == 'odds_ratio':
        calib_weights = (1.0 - P_calib) / (P_calib + 1e-8)
        test_weight = (1.0 - p_test) / (p_test + 1e-8)
    elif weight_type == 'uniform':
        calib_weights = np.ones_like(P_calib)
        test_weight = 1.0
    else:
        raise NotImplementedError(f"Unknown weight_type: {weight_type}")
    return calib_weights, test_weight


# ============================================================
# C1B-MC: Generalized Weighted Conformal Inference
# ============================================================

def generalized_cmc_inference(M_hat, S_matrix, M_star, Omega_cal, Omega_test, P_matrix,
                              alpha_cp, alpha_inf,
                              score_type='normalized_residual', weight_type='odds_ratio'):
    """Weighted conformal inference for 1-bit MC (simulation, M* known).

    Returns coverage, avg length, and entry-wise lower/upper bounds.
    Note: intervals are NOT truncated to [-alpha_inf, alpha_inf].
    Truncation is a post-processing choice, not part of the conformal guarantee.
    """
    calib_i, calib_j = np.nonzero(Omega_cal)
    M_calib_hat = M_hat[calib_i, calib_j]
    M_calib_true = M_star[calib_i, calib_j]
    S_calib = S_matrix[calib_i, calib_j]
    P_calib = P_matrix[calib_i, calib_j]

    scores = get_nonconformity_scores(M_calib_hat, M_calib_true, S_calib, score_type)

    test_i, test_j = np.nonzero(Omega_test)
    n_test = len(test_i)
    lower_bounds = np.zeros(n_test)
    upper_bounds = np.zeros(n_test)
    Q_values = np.zeros(n_test)
    p_values = np.zeros(n_test)

    M_test_hat = M_hat[test_i, test_j]
    M_test_true = M_star[test_i, test_j]

    for k in range(n_test):
        r_idx, c_idx = test_i[k], test_j[k]
        p_xy = P_matrix[r_idx, c_idx]

        calib_w, test_w = get_conformal_weights(P_calib, p_xy, weight_type)
        total_weight = np.sum(calib_w) + test_w
        p_norm = calib_w / total_weight
        p_test_norm = test_w / total_weight

        sort_idx = np.argsort(scores)
        sorted_scores = scores[sort_idx]
        cdf = np.cumsum(p_norm[sort_idx])

        # Conformal quantile Q_ij
        valid_indices = np.where(cdf >= (1.0 - alpha_cp))[0]
        Q_xy = sorted_scores[valid_indices[0]] if len(valid_indices) > 0 else np.inf
        Q_values[k] = Q_xy

        # p-value: smallest alpha for which the interval does NOT cover 0
        # = 1 - (fraction of calibration scores <= |M_hat| / S_ij)
        s_xy = S_matrix[r_idx, c_idx]
        test_score = np.abs(M_test_hat[k]) / (s_xy + 1e-8)
        p_val = 1.0 - np.sum(p_norm[sort_idx][sorted_scores <= test_score])
        p_values[k] = max(p_val, 1.0 / (len(calib_i) + 1))  # minimum achievable p-value

        margin = Q_xy * s_xy if score_type == 'normalized_residual' else Q_xy

        # Intervals WITHOUT truncation (conformal validity is unconditional)
        lower_bounds[k] = M_hat[r_idx, c_idx] - margin
        upper_bounds[k] = M_hat[r_idx, c_idx] + margin

    covered = (M_test_true >= lower_bounds) & (M_test_true <= upper_bounds)
    lengths = upper_bounds - lower_bounds

    return (float(np.mean(covered)), float(np.mean(lengths)),
            lower_bounds, upper_bounds, M_test_hat, M_test_true,
            Q_values, p_values)


# ============================================================
# Real-Data C1B-MC (M* unknown)
# ============================================================

def real_data_cmc_inference(M_hat, S_matrix, Y_obs, Omega_cal, Omega_test, P_matrix,
                            alpha_cp, score_type='hinge', weight_type='odds_ratio'):
    """Weighted conformal inference for real data (M* unknown).

    Uses Y-based nonconformity scores and produces prediction sets
    for Y (classification) rather than intervals for M*.  Returns
    per-entry prediction sets in {-1}, {+1}, or {-1, +1} (abstention).
    """
    calib_i, calib_j = np.nonzero(Omega_cal)
    M_calib_hat = M_hat[calib_i, calib_j]
    Y_calib = Y_obs[calib_i, calib_j]
    S_calib = S_matrix[calib_i, calib_j]
    P_calib = P_matrix[calib_i, calib_j]

    scores = get_nonconformity_scores_real(M_calib_hat, Y_calib, S_calib, score_type)

    test_i, test_j = np.nonzero(Omega_test)
    n_test = len(test_i)

    prediction_sets = []  # list of {1}, {-1}, or {-1, 1}
    Q_values = np.zeros(n_test)
    p_values = np.zeros(n_test)

    M_test_hat = M_hat[test_i, test_j]
    Y_test_true = Y_obs[test_i, test_j]

    for k in range(n_test):
        r_idx, c_idx = test_i[k], test_j[k]
        p_xy = P_matrix[r_idx, c_idx]

        calib_w, test_w = get_conformal_weights(P_calib, p_xy, weight_type)
        total_weight = np.sum(calib_w) + test_w
        p_norm = calib_w / total_weight

        sort_idx = np.argsort(scores)
        sorted_scores = scores[sort_idx]
        cdf = np.cumsum(p_norm[sort_idx])

        valid_indices = np.where(cdf >= (1.0 - alpha_cp))[0]
        Q_xy = sorted_scores[valid_indices[0]] if len(valid_indices) > 0 else np.inf
        Q_values[k] = Q_xy

        # Classification prediction set:
        # Y in {sign(M_hat)} if -Y * M_hat > Q (the score for that sign is above Q)
        # Otherwise Y in {-1, +1}
        score_if_plus1 = -(+1) * M_test_hat[k]   # score if we guess Y=+1
        score_if_minus1 = -(-1) * M_test_hat[k]  # score if we guess Y=-1

        pred_set = []
        if score_if_plus1 <= Q_xy:
            pred_set.append(1)
        if score_if_minus1 <= Q_xy:
            pred_set.append(-1)
        prediction_sets.append(set(pred_set))

        # p-value for rejecting H0: Y_ij = -1 (i.e. for predicting +1)
        test_score_minus1 = score_if_minus1
        p_val = 1.0 - np.sum(p_norm[sort_idx][sorted_scores <= test_score_minus1])
        p_values[k] = max(p_val, 1.0 / (len(calib_i) + 1))

    # Compute coverage and average set size
    covered = 0
    set_sizes = []
    singleton_count = 0
    for k in range(n_test):
        set_size = len(prediction_sets[k])
        set_sizes.append(set_size)
        if set_size == 1:
            singleton_count += 1
        if Y_test_true[k] in prediction_sets[k]:
            covered += 1

    results = {
        'coverage': covered / n_test,
        'avg_set_size': np.mean(set_sizes),
        'singleton_rate': singleton_count / n_test,
        'Q_values': Q_values,
        'p_values': p_values,
        'prediction_sets': prediction_sets,
        'Y_test_true': Y_test_true,
        'M_test_hat': M_test_hat,
    }
    return results


# ============================================================
# Full Conformal: Jackknife+ approximation
# ============================================================

def full_conformal_jackknife(M_hat_list, S_matrix_list, M_star, Omega_cal, Omega_test,
                              alpha_cp, alpha_inf):
    """Jackknife+ style full conformal approximation.

    M_hat_list: list of (M_hat_k, S_matrix_k) fitted on leave-one-out subsets.
    For computational feasibility, we use K-fold instead of LOO.

    Returns coverage and average length metrics.
    """
    K = len(M_hat_list)
    calib_i, calib_j = np.nonzero(Omega_cal)
    test_i, test_j = np.nonzero(Omega_test)
    n_test = len(test_i)

    all_scores = []
    for fold_idx, (M_hat_k, S_mat_k) in enumerate(M_hat_list):
        M_calib_hat_k = M_hat_k[calib_i, calib_j]
        S_calib_k = S_mat_k[calib_i, calib_j]
        scores_k = np.abs(M_calib_hat_k - M_star[calib_i, calib_j]) / (S_calib_k + 1e-8)
        all_scores.append(scores_k)

    # Aggregate scores across folds (jackknife+ style)
    aggregated_scores = np.mean(np.stack(all_scores, axis=0), axis=0)

    lower_bounds = np.zeros(n_test)
    upper_bounds = np.zeros(n_test)

    for k in range(n_test):
        r_idx, c_idx = test_i[k], test_j[k]
        n_cal = len(calib_i)
        q_idx = int(np.ceil((1.0 - alpha_cp) * (n_cal + 1))) - 1
        q_idx = min(q_idx, n_cal - 1)
        Q = np.sort(aggregated_scores)[q_idx]

        s_xy = S_matrix_list[0][r_idx, c_idx]  # use the first fold's S for simplicity
        margin = Q * s_xy
        lower_bounds[k] = M_hat_list[0][r_idx, c_idx] - margin  # first fold point estimate
        upper_bounds[k] = M_hat_list[0][r_idx, c_idx] + margin

    M_test_true = M_star[test_i, test_j]
    covered = (M_test_true >= lower_bounds) & (M_test_true <= upper_bounds)
    lengths = upper_bounds - lower_bounds

    return float(np.mean(covered)), float(np.mean(lengths))


# ============================================================
# Coverage Bounds (Theoretical)
# ============================================================

def compute_coverage_lower_bound(alpha_cp, delta_exch, delta_est, n_cal):
    """Theorem: Marginal coverage lower bound.

    P(Y_{n+1} in C) >= 1 - alpha_cp - delta_exch - delta_est

    where:
      delta_exch = total variation distance due to weighted exchangeability
      delta_est  = error from using M̂ instead of M* in nonconformity scores
    """
    return max(0.0, 1.0 - alpha_cp - delta_exch - delta_est)


def estimate_delta_exchangeability(P_calib, P_test_sample):
    """Estimate delta_exch from the total variation between calib and test distributions.

    delta_exch <= sup_{calib, test} |P_calib(w) - P_test(w)| / 2

    For odds-ratio weights w = (1-P)/P, we compute the empirical TV.
    """
    w_calib = (1.0 - P_calib) / (P_calib + 1e-8)
    w_test = (1.0 - P_test_sample) / (P_test_sample + 1e-8)

    # Use quantile-based approximation of TV distance
    all_w = np.concatenate([w_calib, w_test])
    bins = np.percentile(all_w, np.linspace(0, 100, 50))
    hist_calib, _ = np.histogram(w_calib, bins=bins, density=True)
    hist_test, _ = np.histogram(w_test, bins=bins, density=True)

    bin_widths = np.diff(bins)
    tv = 0.5 * np.sum(np.abs(hist_calib - hist_test) * bin_widths)
    return tv


def estimate_delta_estimation(M_hat, M_star, Omega_cal, S_matrix):
    """Estimate delta_est from the discrepancy in nonconformity score distributions.

    Uses Wasserstein-1 distance between scores computed with M̂ vs M*.
    """
    calib_i, calib_j = np.nonzero(Omega_cal)
    S_cal = S_matrix[calib_i, calib_j]

    scores_true = np.abs(M_star[calib_i, calib_j] - M_star[calib_i, calib_j]) / (S_cal + 1e-8)  # ≈ 0
    scores_hat = np.abs(M_hat[calib_i, calib_j] - M_star[calib_i, calib_j]) / (S_cal + 1e-8)

    # W1 distance approximated by mean absolute difference of sorted scores
    delta = np.mean(np.abs(np.sort(scores_hat) - np.sort(scores_true)))
    return delta
