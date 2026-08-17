"""
Publication-quality figures for the paper (revised).
Figure 1: Coverage vs Hypothesized Rank — legend upper-right, original blue/red
Figure 2: Sample Size Scaling — legend upper-right, original blue/red
Figure 3: Sparsity Robustness — legend upper-right, original blue/red
Figure 4: Score distribution histogram + QQ-plot (replaces old pointwise plot)
Figure S1: Coverage calibration plot (observed vs nominal)
"""
import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import norm as norm_dist, probplot

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9

COLOR_CMC   = '#0000FF'   
COLOR_ASYM  = '#FF0000'   
COLOR_TRUE  = 'green'
COLOR_GRAY  = 'gray'
PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'figures')


def find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, 'results')):
        parent = os.path.dirname(d)
        if parent == d: raise RuntimeError('Cannot find project root')
        d = parent
    return d


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================
# Figure 1: Coverage vs Hypothesized Rank
# ============================================================
def figure_1_coverage_vs_rank(root):
    files = {
        'Rasch + Logistic (Well-specified)': os.path.join(root, 'results/grid/res_N500_rasch_p0.2_calib0.2_a1.0_logistic.json'),
        'General + Heavy-tail (Misspecified)': os.path.join(root, 'results/grid/res_N500_general_p0.2_calib0.2_a1.0_heavy_tail.json'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    r_true = 5

    for row, (label, fpath) in enumerate(files.items()):
        if not os.path.exists(fpath):
            continue
        data = load_json(fpath)
        res = data['data']
        ranks = sorted([int(k) for k in res.keys()])
        str_ranks = [str(r) for r in ranks]

        c_covs = [res[r]['cmc_cov'] for r in str_ranks]
        a_covs = [res[r]['asym_cov'] for r in str_ranks]
        c_lens = [res[r]['cmc_len'] for r in str_ranks]
        a_lens = [res[r]['asym_len'] for r in str_ranks]

        ax_cov, ax_len = axes[row]

        # --- LEFT: Coverage ---
        ax_cov.plot(ranks, c_covs, '-o', color=COLOR_CMC, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='C1B-MC (Ours)')
        ax_cov.plot(ranks, a_covs, '-s', color=COLOR_ASYM, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='Asym (Chen 2023)')
        ax_cov.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.0)
        ax_cov.axvline(x=r_true, color='gray', linestyle=':', linewidth=1.0)
        ax_cov.set_ylabel('Empirical Coverage')
        ax_cov.set_title(label, fontsize=12, fontweight='bold')
        y_min = max(0.65, min(min(c_covs), min(a_covs)) - 0.08)
        ax_cov.set_ylim(y_min, 1.01)
        ax_cov.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray', fontsize=8)

        # --- RIGHT: Length, ylim [0.6, 1.5] ---
        ax_len.plot(ranks, c_lens, '-o', color=COLOR_CMC, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='C1B-MC (Ours)')
        ax_len.plot(ranks, a_lens, '-s', color=COLOR_ASYM, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='Asym (Chen 2023)')
        ax_len.axvline(x=r_true, color='gray', linestyle=':', linewidth=1.0)
        ax_len.set_ylabel('Avg. Interval Length')
        ax_len.set_ylim(0.6, 1.5)
        ax_len.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray', fontsize=8)

    axes[1, 0].set_xlabel(r'Hypothesized Rank ($r_{\mathrm{fit}}$)')
    axes[1, 1].set_xlabel(r'Hypothesized Rank ($r_{\mathrm{fit}}$)')
    fig.suptitle('Coverage and Interval Length vs. Hypothesized Rank ($N=500$, $p=0.2$)',
                 y=1.005, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure1_coverage_vs_rank.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 1 -> {path}")


# ============================================================
# Figure 2: Sample Size Scaling
# ============================================================
def figure_2_sample_size(root):
    files = {
        'Rasch + Logistic': os.path.join(root, 'results/grid/res_Server_SampleSize_rasch_logistic.json'),
        'General + Heavy-tail': os.path.join(root, 'results/grid/res_Server_SampleSize_general_heavy_tail.json'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for col, (label, fpath) in enumerate(files.items()):
        if not os.path.exists(fpath):
            continue
        data = load_json(fpath)
        res = data['data']
        N_list = data['N_list']
        str_N = [str(n) for n in N_list]

        c_covs = [res[n]['cmc_cov'] for n in str_N]
        a_covs = [res[n]['asym_cov'] for n in str_N]
        c_lens = [res[n]['cmc_len'] for n in str_N]
        a_lens = [res[n]['asym_len'] for n in str_N]

        ax_cov = axes[0, col]
        ax_len = axes[1, col]

        # --- Top: Coverage ---
        ax_cov.plot(N_list, c_covs, '-o', color=COLOR_CMC, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='C1B-MC (Ours)')
        ax_cov.plot(N_list, a_covs, '-s', color=COLOR_ASYM, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='Asym (Chen 2023)')
        ax_cov.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.0)
        ax_cov.set_ylabel('Empirical Coverage')
        ax_cov.set_title(label, fontsize=12, fontweight='bold')
        min_cov = min(min(c_covs), min(a_covs))
        ax_cov.set_ylim(min(0.84, min_cov - 0.05), 1.02)
        ax_cov.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray', fontsize=8)

        # --- Bottom: Length ---
        ax_len.plot(N_list, c_lens, '-o', color=COLOR_CMC, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='C1B-MC (Ours)')
        ax_len.plot(N_list, a_lens, '-s', color=COLOR_ASYM, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='Asym (Chen 2023)')
        ax_len.set_xlabel('Matrix Dimension ($N$)')
        ax_len.set_ylabel('Avg. Length')
        ax_len.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray', fontsize=8)

    fig.suptitle('Sample Size Scaling ($p=0.2$, calib ratio $0.2$)',
                 y=1.005, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure2_sample_size.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 2 -> {path}")


# ============================================================
# Figure 3: Sparsity Robustness
# ============================================================
def figure_3_sparsity(root):
    files = {
        r'Rasch + Logistic ($600\times400$)': os.path.join(root, 'results/grid/res_Sparsity_rasch_logistic.json'),
        r'General + Heavy-tail ($600\times400$)': os.path.join(root, 'results/grid/res_Sparsity_general_heavy_tail.json'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for col, (label, fpath) in enumerate(files.items()):
        if not os.path.exists(fpath):
            continue
        data = load_json(fpath)
        res = data['data']
        p_list = data['p_list']
        str_p = [str(p) for p in p_list]

        c_covs = [res[p]['cmc_cov'] for p in str_p]
        a_covs = [res[p]['asym_cov'] for p in str_p]
        c_lens = [res[p]['cmc_len'] for p in str_p]
        a_lens = [res[p]['asym_len'] for p in str_p]

        ax_cov = axes[0, col]
        ax_len = axes[1, col]

        # --- Top: Coverage ---
        ax_cov.plot(p_list, c_covs, '-o', color=COLOR_CMC, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='C1B-MC (Ours)')
        ax_cov.plot(p_list, a_covs, '-s', color=COLOR_ASYM, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='Asym (Chen 2023)')
        ax_cov.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.0)
        ax_cov.set_ylabel('Empirical Coverage')
        ax_cov.set_title(label, fontsize=12, fontweight='bold')
        min_cov = min(min(c_covs), min(a_covs))
        ax_cov.set_ylim(min(0.65, min_cov - 0.08), 1.02)
        ax_cov.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray', fontsize=8)

        # --- Bottom: Length ---
        ax_len.plot(p_list, c_lens, '-o', color=COLOR_CMC, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='C1B-MC (Ours)')
        ax_len.plot(p_list, a_lens, '-s', color=COLOR_ASYM, markersize=6, markeredgewidth=1,
                    markerfacecolor='white', label='Asym (Chen 2023)')
        ax_len.set_xlabel('Observation Rate ($p$)')
        ax_len.set_ylabel('Avg. Length')
        ax_len.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray', fontsize=8)

    fig.suptitle('Sparsity Robustness (Calibration ratio $0.2$)',
                 y=1.005, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure3_sparsity.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 3 -> {path}")


# ============================================================
# Figure 4: Standardized Residual Distribution
# ============================================================
def figure_4_score_distribution(root):
    """Histogram of standardized residuals for both settings, with N(0,1) overlay.
    Shows that C1B-MC scores are well-approximated by Gaussian under correct spec,
    but heavy-tailed under misspecification — explaining the Asym failure.
    """
    files = [
        ('Rasch + Logistic\n(Well-specified)', os.path.join(root, 'results/grid/res_N500_rasch_p0.2_calib0.2_a1.0_logistic.json')),
        ('General + Heavy-tail\n(Misspecified)', os.path.join(root, 'results/grid/res_N500_general_p0.2_calib0.2_a1.0_heavy_tail.json')),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, (label, fpath) in enumerate(files):
        ax = axes[idx]
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, 'Data missing', ha='center', va='center', transform=ax.transAxes)
            continue

        data = load_json(fpath)
        # Try to get scores from plot_data_r12, otherwise fall back to any available
        scores = None
        for key in ['plot_data_r12', 'plot_data_r8', 'plot_data_r5']:
            pd = data.get(key, {})
            if 'scores' in pd:
                scores = np.array(pd['scores'])
                break

        if scores is None:
            # Generate from data differences for r=5
            res = data['data']
            # Approximate scores: z = (coverage deviation) / se, not exact but illustrative
            ax.text(0.5, 0.5, 'Score data\nnot available', ha='center', va='center', transform=ax.transAxes)
            continue

        # Histogram
        sns.histplot(scores, stat="density", bins=35, ax=ax, color='#B0C4DE', alpha=0.6,
                     edgecolor='white', linewidth=0.5, label='Empirical scores')

        # N(0,1) overlay
        x_pdf = np.linspace(-5, 5, 200)
        ax.plot(x_pdf, norm_dist.pdf(x_pdf, 0, 1), color='#D2691E', linewidth=2.5,
                label=r'$\mathcal{N}(0,1)$')

        # KDE of scores
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(scores)
            ax.plot(x_pdf, kde(x_pdf), color=COLOR_CMC, linewidth=1.8, linestyle='--',
                    label='KDE')
        except Exception:
            pass

        ax.set_xlim(-5, 5)
        ax.set_xlabel(r'Standardized Score $(\widehat{M}_{ij} - M^\star_{ij}) / \mathbf{S}_{ij}$')
        if idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

        # Add divergence annotation
        ks_stat = np.max(np.abs(np.arange(1, len(scores)+1)/len(scores) - norm_dist.cdf(np.sort(scores))))
        ax.text(0.95, 0.95, f'KS vs N(0,1): {ks_stat:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    fig.suptitle('Figure 4: Distribution of Standardized Residuals ($N=500$, $p=0.2$, $r=12$)',
                 y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure4_score_distribution.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 4 -> {path}")


# ============================================================
# Figure S1: Aggregated coverage bar chart across all settings
# ============================================================
def figure_s1_coverage_summary(root):
    """Bar chart of coverage across all major experimental configurations,
    showing C1B-MC vs Asym side by side with the 0.90 target line."""
    configs = [
        ('Rasch+Log\nr=2',  'results/grid/res_N500_rasch_p0.2_calib0.2_a1.0_logistic.json', '2'),
        ('Rasch+Log\nr=5',  'results/grid/res_N500_rasch_p0.2_calib0.2_a1.0_logistic.json', '5'),
        ('Rasch+Log\nr=12', 'results/grid/res_N500_rasch_p0.2_calib0.2_a1.0_logistic.json', '12'),
        ('Gen+HT\nr=2',     'results/grid/res_N500_general_p0.2_calib0.2_a1.0_heavy_tail.json', '2'),
        ('Gen+HT\nr=5',     'results/grid/res_N500_general_p0.2_calib0.2_a1.0_heavy_tail.json', '5'),
        ('Gen+HT\nr=12',    'results/grid/res_N500_general_p0.2_calib0.2_a1.0_heavy_tail.json', '12'),
        ('Phase5\nr=5',     'results/grid/res_Phase5_general_p0.3_calib0.2_a5.0_heavy_tail.json', '5'),
        ('Phase5\nr=12',    'results/grid/res_Phase5_general_p0.3_calib0.2_a5.0_heavy_tail.json', '12'),
    ]

    labels = []
    cmc_vals, asym_vals = [], []
    for label, fpath, r_key in configs:
        full = os.path.join(root, fpath)
        if not os.path.exists(full):
            continue
        d = load_json(full)['data']
        if r_key in d:
            labels.append(label)
            cmc_vals.append(d[r_key]['cmc_cov'])
            asym_vals.append(d[r_key]['asym_cov'])

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w/2, cmc_vals, w, color=COLOR_CMC, alpha=0.85, edgecolor='white', label='C1B-MC (Ours)')
    bars2 = ax.bar(x + w/2, asym_vals, w, color=COLOR_ASYM, alpha=0.85, edgecolor='white', label='Asym (Chen 2023)')

    ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.5, label='Target (0.90)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Empirical Coverage')
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Add value annotations
    for bar, val in zip(bars1, cmc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7, rotation=90)
    for bar, val in zip(bars2, asym_vals):
        y_pos = bar.get_height() + 0.01 if val > 0.15 else 0.03
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7, rotation=90, color='white' if val < 0.3 else 'black')

    fig.suptitle('Figure S1: Coverage Summary Across All Experimental Configurations',
                 y=1.005, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure_s1_coverage_summary.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure S1 -> {path}")


if __name__ == "__main__":
    root = find_root()
    os.chdir(root)
    print(f"Project root: {root}\n")
    os.makedirs(PAPER_DIR, exist_ok=True)

    figure_1_coverage_vs_rank(root)
    figure_2_sample_size(root)
    figure_3_sparsity(root)
    figure_4_score_distribution(root)
    figure_s1_coverage_summary(root)

    print(f"\n>>> All figures saved to: {PAPER_DIR}")
