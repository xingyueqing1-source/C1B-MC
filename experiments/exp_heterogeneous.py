"""
Heterogeneous missingness experiment — weighted C1B-MC vs unweighted vs Asymptotic.
Demonstrates that only weighted (odds-ratio) conformal maintains valid coverage
under non-uniform observation probabilities.

Design: 500x500, general rank-5, heavy-tail.
         P_ij varies from ~0.02 (sparse rows/cols) to ~0.95 (dense rows/cols).
         20 independent trials. 5000 test entries sampled per trial per method.
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_data
from src.estimators import solve_1bit_mc_with_cv
from src.inference import compute_local_uncertainty, asymptotic_inference_chen2023
from sklearn.model_selection import train_test_split

# --- style ---
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

COLOR_ASYM = '#E64B35'
COLOR_UNW  = '#F39B7F'
COLOR_W    = '#3C5488'
COLOR_LO   = '#B0C4DE'
COLOR_HI   = '#3C5488'

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def make_heterogeneous_P(N, skew=3.5, p_median=0.15):
    """Row x column multiplicative observation probability matrix.
    P_ij = p_median * exp(-skew*(i/N-0.5)) * exp(-skew*(j/N-0.5)), clipped."""
    x = np.linspace(0, 1, N)
    r_raw = np.exp(-skew * (x - 0.5))
    c_raw = np.exp(-skew * (x - 0.5))
    r = r_raw / np.mean(r_raw)
    c = c_raw / np.mean(c_raw)
    P = p_median * np.outer(r, c)
    return np.clip(P, 0.015, 0.98).astype(np.float64)


def weighted_conformal_fast(cal_scores, cal_P, P_test_array, M_test_hat, M_test_true,
                             S_test, alpha_cp):
    """Vectorized weighted conformal: pre-sort scores, binary-search weights per entry.

    Returns (coverage, avg_length).
    Uses the efficient O(n_test * log n_cal) approach.
    """
    n_cal = len(cal_scores)
    n_test = len(M_test_hat)
    sort_idx = np.argsort(cal_scores)
    sorted_scores = cal_scores[sort_idx]
    sorted_P = cal_P[sort_idx]
    sorted_w = (1.0 - sorted_P) / (sorted_P + 1e-8)

    lower = np.zeros(n_test)
    upper = np.zeros(n_test)

    for k in range(n_test):
        w_test = (1.0 - P_test_array[k]) / (P_test_array[k] + 1e-8)
        total_w = np.sum(sorted_w) + w_test
        cumsum = np.cumsum(sorted_w / total_w)
        valid = np.searchsorted(cumsum, 1.0 - alpha_cp, side='left')
        Q = sorted_scores[min(valid, n_cal - 1)]
        margin = Q * S_test[k]
        lower[k] = M_test_hat[k] - margin
        upper[k] = M_test_hat[k] + margin

    covered = (M_test_true >= lower) & (M_test_true <= upper)
    return float(np.mean(covered)), float(np.mean(upper - lower)), covered


def unweighted_conformal_fast(cal_scores, M_test_hat, M_test_true, S_test, alpha_cp):
    """Single-pass unweighted conformal (same Q for all test entries)."""
    n_cal = len(cal_scores)
    q_idx = min(int(np.ceil((1.0 - alpha_cp) * (n_cal + 1))) - 1, n_cal - 1)
    Q = np.sort(cal_scores)[q_idx]

    lower = M_test_hat - Q * S_test
    upper = M_test_hat + Q * S_test
    covered = (M_test_true >= lower) & (M_test_true <= upper)
    return float(np.mean(covered)), float(np.mean(upper - lower)), covered


def run_one_trial(N, r_true, alpha_inf, alpha_cp, seed, n_test_sample=5000):
    """Single trial: generate data, fit, compare 3 methods on sampled test entries."""

    # ---- generate latent M* ----
    M_star, _, Y_obs, _ = generate_data(
        N, N, r_true, p=0.2, alpha=alpha_inf,
        noise_type='heavy_tail', model_type='general', seed=seed)

    # ---- heterogeneous observation mask ----
    P_matrix = make_heterogeneous_P(N, skew=3.5, p_median=0.15)
    rng = np.random.RandomState(seed + 9999)
    Omega = rng.rand(N, N) < P_matrix
    Y_obs[~Omega] = 0

    density = float(np.mean(Omega))

    # ---- split observed 60/20/20 ----
    oi, oj = np.nonzero(Omega)
    n_obs = len(oi)
    tr_idx, ct_idx = train_test_split(np.arange(n_obs), test_size=0.4, random_state=seed)
    cal_idx, _ = train_test_split(ct_idx, test_size=0.5, random_state=seed + 1)

    Omega_tr = np.zeros_like(Omega, dtype=bool)
    Omega_tr[oi[tr_idx], oj[tr_idx]] = True
    Omega_cal = np.zeros_like(Omega, dtype=bool)
    Omega_cal[oi[cal_idx], oj[cal_idx]] = True

    # ---- fit ----
    M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r=r_true, alpha=alpha_inf)
    S_matrix = compute_local_uncertainty(M_hat, Omega_tr)

    # ---- sample test entries (unobserved) ----
    all_test_i, all_test_j = np.nonzero(~Omega)
    n_unobs = len(all_test_i)
    n_sample = min(n_test_sample, n_unobs)
    samp_idx = rng.choice(n_unobs, n_sample, replace=False)
    test_i = all_test_i[samp_idx]
    test_j = all_test_j[samp_idx]

    M_test_hat  = M_hat[test_i, test_j]
    M_test_true = M_star[test_i, test_j]
    S_test      = S_matrix[test_i, test_j]
    P_test_arr  = P_matrix[test_i, test_j]

    # ---- calibration scores ----
    cal_r, cal_c = np.nonzero(Omega_cal)
    cal_scores = np.abs(M_hat[cal_r, cal_c] - M_star[cal_r, cal_c]) / (S_matrix[cal_r, cal_c] + 1e-8)
    cal_P_vec  = P_matrix[cal_r, cal_c]

    # ================================================================
    # 1. Asymptotic (z-intervals)
    # ================================================================
    z_val = 1.6448536269514722  # norm.ppf(0.95)
    asym_lo = M_test_hat - z_val * S_test
    asym_hi = M_test_hat + z_val * S_test
    asym_cov_vec = (M_test_true >= asym_lo) & (M_test_true <= asym_hi)
    a_cov = float(np.mean(asym_cov_vec))
    a_len = float(np.mean(asym_hi - asym_lo))

    # ================================================================
    # 2. C1B-MC unweighted
    # ================================================================
    u_cov, u_len, u_cov_vec = unweighted_conformal_fast(
        cal_scores, M_test_hat, M_test_true, S_test, alpha_cp)

    # ================================================================
    # 3. C1B-MC weighted (odds-ratio)
    # ================================================================
    w_cov, w_len, w_cov_vec = weighted_conformal_fast(
        cal_scores, cal_P_vec, P_test_arr, M_test_hat, M_test_true, S_test, alpha_cp)

    # ---- subgroup: split by median P_ij ----
    med_P = np.median(P_test_arr)
    lo = P_test_arr <= med_P
    hi = P_test_arr > med_P

    def subcov(vec, mask):
        return float(np.mean(vec[mask])) if np.sum(mask) > 0 else np.nan

    return {
        'density': density,
        'P_05': float(np.percentile(P_matrix[Omega], 5)),
        'P_95': float(np.percentile(P_matrix[Omega], 95)),
        'n_test': n_sample,
        # Aggregate
        'Asym_cov': a_cov, 'Asym_len': a_len,
        'Unif_cov': u_cov, 'Unif_len': u_len,
        'Wght_cov': w_cov, 'Wght_len': w_len,
        # Subgroup coverage
        'Asym_cov_lo': subcov(asym_cov_vec, lo), 'Asym_cov_hi': subcov(asym_cov_vec, hi),
        'Unif_cov_lo': subcov(u_cov_vec, lo),    'Unif_cov_hi': subcov(u_cov_vec, hi),
        'Wght_cov_lo': subcov(w_cov_vec, lo),    'Wght_cov_hi': subcov(w_cov_vec, hi),
        # Subgroup length (same across all entries for unif; per-entry for weighted)
        'Asym_len_lo': float(np.mean((asym_hi - asym_lo)[lo])),
        'Asym_len_hi': float(np.mean((asym_hi - asym_lo)[hi])),
    }


def main():
    N, r_true = 500, 5
    alpha_inf, alpha_cp = 1.0, 0.10
    n_trials = 10
    n_test = 5000

    cache_path = os.path.join(ROOT, 'results', 'heterogeneous_results.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            trials = json.load(f)
        print(f"Loaded {len(trials)} cached trials")
    else:
        trials = []

    existing_seeds = {t.get('seed', i) for i, t in enumerate(trials)}

    for t_idx in range(n_trials):
        seed = 500 + t_idx
        if seed in existing_seeds:
            print(f"[Trial {t_idx+1}/{n_trials}] skip (cached)", flush=True)
            continue
        print(f"[Trial {t_idx+1}/{n_trials}] seed={seed} ...", flush=True)
        res = run_one_trial(N, r_true, alpha_inf, alpha_cp, seed)
        res['seed'] = seed
        trials.append(res)
        existing_seeds.add(seed)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(trials, f, indent=2)

    # ============================================================
    # Aggregate summary
    # ============================================================
    def mean_std(key):
        vals = [t[key] for t in trials]
        return np.mean(vals), np.std(vals)

    print("\n" + "="*80)
    print("HETEROGENEOUS MISSINGNESS RESULTS")
    print("="*80)
    print(f"{'Method':<18} {'Coverage':>12} {'Length':>10}  {'Lo-P Cov':>10} {'Hi-P Cov':>10}")
    print("-"*80)
    for name, keys in [
        ('Asymptotic',  ('Asym_cov','Asym_len','Asym_cov_lo','Asym_cov_hi')),
        ('C1B-MC Unif', ('Unif_cov','Unif_len','Unif_cov_lo','Unif_cov_hi')),
        ('C1B-MC Wght', ('Wght_cov','Wght_len','Wght_cov_lo','Wght_cov_hi')),
    ]:
        m = [mean_std(k) for k in keys]
        print(f"{name:<18} {m[0][0]:>7.4f}±{m[0][1]:.4f}  {m[1][0]:>7.4f}±{m[1][1]:.4f}  "
              f"{m[2][0]:>7.4f}±{m[2][1]:.4f}  {m[3][0]:>7.4f}±{m[3][1]:.4f}")

    print(f"\nObservation density: {np.mean([t['density'] for t in trials]):.4f}")
    print(f"P_ij range (5-95%): [{np.mean([t['P_05'] for t in trials]):.4f}, "
          f"{np.mean([t['P_95'] for t in trials]):.4f}]")
    print(f"Samples per trial: {trials[0]['n_test']}")

    # ============================================================
    # Figure 1: Aggregate coverage + length bar chart
    # ============================================================
    methods = ['Asymptotic', 'Unweighted', 'Weighted']
    cov_keys = ['Asym_cov', 'Unif_cov', 'Wght_cov']
    len_keys = ['Asym_len', 'Unif_len', 'Wght_len']
    colors   = [COLOR_ASYM, COLOR_UNW, COLOR_W]
    labels   = ['Asymptotic\n(Chen 2023)', 'C1B-MC\nUnweighted', 'C1B-MC\nWeighted\n(Ours)']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(3)
    w = 0.50

    cov_means = [np.mean([t[k] for t in trials]) for k in cov_keys]
    cov_stds  = [np.std([t[k] for t in trials]) for k in cov_keys]
    len_means = [np.mean([t[k] for t in trials]) for k in len_keys]
    len_stds  = [np.std([t[k] for t in trials]) for k in len_keys]

    bars1 = ax1.bar(x, cov_means, w, color=colors, edgecolor='white', linewidth=0.8,
                    yerr=cov_stds, capsize=7, error_kw={'linewidth': 1.3})
    ax1.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Empirical Coverage', fontsize=12)
    ax1.set_ylim(0.65, 1.02)
    ax1.set_title('Coverage (target = 0.90)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars1, cov_means):
        ax1.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + 0.01, 1.01),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    bars2 = ax2.bar(x, len_means, w, color=colors, edgecolor='white', linewidth=0.8,
                    yerr=len_stds, capsize=7, error_kw={'linewidth': 1.3})
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Avg. Interval Length', fontsize=12)
    ax2.set_title('Interval Length', fontsize=12, fontweight='bold')
    for bar, val in zip(bars2, len_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    d_avg = np.mean([t['density'] for t in trials])
    p5, p95 = np.mean([t['P_05'] for t in trials]), np.mean([t['P_95'] for t in trials])
    fig.suptitle(f'Heterogeneous Missingness ($N={N}$, $r=5$, heavy-tail, {n_trials} trials)\n'
                 f'Obs. density = {d_avg:.3f},  $P_{{ij}}$ range = [{p5:.3f}, {p95:.3f}]',
                 y=1.005, fontsize=12, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(ROOT, 'paper', 'figures', 'heterogeneous_aggregate.pdf')
    fig.savefig(path1, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Figure 1 -> {path1}")

    # ============================================================
    # Figure 2: Subgroup coverage (lo-P vs hi-P)
    # ============================================================
    fig2, ax = plt.subplots(figsize=(9, 5))
    lo_keys = ['Asym_cov_lo', 'Unif_cov_lo', 'Wght_cov_lo']
    hi_keys = ['Asym_cov_hi', 'Unif_cov_hi', 'Wght_cov_hi']

    lo_means = [np.mean([t[k] for t in trials]) for k in lo_keys]
    hi_means = [np.mean([t[k] for t in trials]) for k in hi_keys]
    lo_stds  = [np.std([t[k] for t in trials]) for k in lo_keys]
    hi_stds  = [np.std([t[k] for t in trials]) for k in hi_keys]

    w2 = 0.32
    bars_lo = ax.bar(x - w2/2, lo_means, w2, color=COLOR_LO, edgecolor=COLOR_HI, linewidth=1.0,
                     yerr=lo_stds, capsize=5, error_kw={'linewidth': 1.0},
                     label='Low-$P$ entries ($P_{ij} \\leq$ median)')
    bars_hi = ax.bar(x + w2/2, hi_means, w2, color=COLOR_HI, edgecolor='#1A2A4A', linewidth=1.0,
                     yerr=hi_stds, capsize=5, error_kw={'linewidth': 1.0},
                     label='High-$P$ entries ($P_{ij} >$ median)')
    ax.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Empirical Coverage', fontsize=12)
    ax.set_ylim(0.65, 1.02)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10, edgecolor='lightgray')
    ax.set_title('Subgroup Coverage: Low-$P$ vs High-$P$ Test Entries\n'
                 f'($N={N}$, $r=5$, heavy-tail, {n_trials} trials)',
                 fontsize=12, fontweight='bold')
    for bar, val in zip(bars_lo, lo_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLOR_HI)
    for bar, val in zip(bars_hi, hi_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1A2A4A')
    plt.tight_layout()
    path2 = os.path.join(ROOT, 'paper', 'figures', 'heterogeneous_subgroup.pdf')
    fig2.savefig(path2, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 2 -> {path2}")

    # ============================================================
    # Figure 3: P-matrix visualization
    # ============================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4.2))
    P_ex = make_heterogeneous_P(N, skew=3.5, p_median=0.15)

    im = ax3a.imshow(P_ex[:100, :100], cmap='YlOrRd', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')
    ax3a.set_title(r'$P_{ij}$ (top-left 100$\times$100)', fontsize=11, fontweight='bold')
    ax3a.set_xlabel('Column'); ax3a.set_ylabel('Row')
    plt.colorbar(im, ax=ax3a, shrink=0.82)

    rng_ex = np.random.RandomState(42)
    Omega_ex = rng_ex.rand(N, N) < P_ex
    P_obs = P_ex[Omega_ex]
    ax3b.hist(P_obs, bins=60, density=True, color='#E64B35', alpha=0.75, edgecolor='white')
    ax3b.axvline(x=np.median(P_obs), color='black', linestyle='--', linewidth=1.5,
                 label=f'Median = {np.median(P_obs):.3f}')
    ax3b.set_xlabel(r'$P_{ij}$ on observed entries')
    ax3b.set_ylabel('Density')
    ax3b.set_title('Distribution of $P_{ij}$', fontsize=11, fontweight='bold')
    ax3b.legend(fontsize=10)
    fig3.suptitle('Heterogeneous Observation Probability Design',
                  y=1.01, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path3 = os.path.join(ROOT, 'paper', 'figures', 'heterogeneous_P_design.pdf')
    fig3.savefig(path3, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 3 -> {path3}")

    print("\n>>> Complete.")


if __name__ == "__main__":
    main()
