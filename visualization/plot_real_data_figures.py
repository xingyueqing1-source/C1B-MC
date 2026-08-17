"""
Real data figures for MovieLens section.
Figure R1: Heatmap of M_hat (50x50 dense sub-block) + observation mask
Figure R2: Three-panel diagnostics (histogram, coverage vs alpha, S vs |M|)
Figure R3: Sorted prediction confidence plot — |M_hat| with conformal quantile
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.special import expit

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

COLOR_CMC = 'blue'
COLOR_ASYM = 'red'
COLOR_SINGLETON = '#2166AC'
COLOR_DOUBLETON = '#D6604D'
PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'figures')


def find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, 'results')):
        parent = os.path.dirname(d)
        if parent == d: raise RuntimeError('Cannot find project root')
        d = parent
    return d


def prepare_data():
    """Load MovieLens data, fit model, return all needed arrays."""
    root = find_root()
    os.chdir(root)
    sys.path.insert(0, root)

    import urllib.request, zipfile
    from sklearn.model_selection import train_test_split
    from src.estimators import solve_1bit_mc_with_cv
    from src.inference import compute_local_uncertainty

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    extracted_dir = os.path.join(data_dir, "ml-100k")
    zip_path = os.path.join(data_dir, "ml-100k.zip")

    if not os.path.exists(extracted_dir):
        if not os.path.exists(zip_path):
            print("Downloading MovieLens 100K...")
            try:
                urllib.request.urlretrieve(
                    "https://files.grouplens.org/datasets/movielens/ml-100k.zip", zip_path)
            except Exception:
                print("Download failed, using synthetic proxy.")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)

    ratings_file = os.path.join(extracted_dir, "u.data")
    if os.path.exists(ratings_file):
        raw = np.loadtxt(ratings_file, dtype=int)
        n_users, n_movies = raw[:, 0].max(), raw[:, 1].max()
        R = np.full((n_users, n_movies), np.nan)
        for r in raw:
            R[r[0] - 1, r[1] - 1] = r[2]
        Y_obs = np.where(R >= 3, 1.0, -1.0)
        Y_obs[np.isnan(R)] = 0.0
        Omega = (~np.isnan(R)).astype(bool)
        P_matrix = np.full((n_users, n_movies), np.mean(Omega))
        dataset_label = "MovieLens 100K"
    else:
        from src.data_generator import generate_data
        n_users, n_movies = 300, 300
        _, Omega, Y_obs, P_matrix = generate_data(
            n_users, n_movies, r=2, p=0.15, alpha=1.0,
            noise_type='logistic', model_type='rasch', seed=123)
        dataset_label = "Synthetic Proxy"
        print("Using synthetic fallback data.")

    # Subsample
    max_n = 300
    if n_users > max_n or n_movies > max_n:
        row_act = np.sum(Omega, axis=1)
        col_pop = np.sum(Omega, axis=0)
        top_u = np.argsort(row_act)[-max_n:]
        top_m = np.argsort(col_pop)[-max_n:]
        Y_obs = Y_obs[np.ix_(top_u, top_m)]
        Omega = Omega[np.ix_(top_u, top_m)]
        P_matrix = P_matrix[np.ix_(top_u, top_m)] if P_matrix.shape == Omega.shape else np.full((max_n, max_n), np.mean(Omega))
        n_users, n_movies = max_n, max_n

    density = np.mean(Omega)
    print(f"Data: {n_users}x{n_movies}, density={density:.4f}")

    # Split
    omega_i, omega_j = np.nonzero(Omega)
    n_obs = len(omega_i)
    tr_cal_idx, test_idx = train_test_split(np.arange(n_obs), test_size=0.2, random_state=42)
    tr_idx, calib_idx = train_test_split(tr_cal_idx, test_size=0.25, random_state=42)

    Omega_tr  = np.zeros_like(Omega, dtype=bool); Omega_tr[omega_i[tr_idx], omega_j[tr_idx]] = True
    Omega_cal = np.zeros_like(Omega, dtype=bool); Omega_cal[omega_i[calib_idx], omega_j[calib_idx]] = True
    Omega_test = np.zeros_like(Omega, dtype=bool); Omega_test[omega_i[test_idx], omega_j[test_idx]] = True
    print(f"Train={len(tr_idx)}, Calib={len(calib_idx)}, Test={len(test_idx)}")

    # Fit model
    print("Fitting model...")
    M_hat = solve_1bit_mc_with_cv(Omega_tr, Y_obs, r=5, alpha=1.0)
    S_matrix = compute_local_uncertainty(M_hat, Omega_tr)

    return {
        'M_hat': M_hat, 'S_matrix': S_matrix, 'Y_obs': Y_obs,
        'Omega': Omega, 'Omega_tr': Omega_tr, 'Omega_cal': Omega_cal,
        'Omega_test': Omega_test, 'P_matrix': P_matrix,
        'n_users': n_users, 'n_movies': n_movies, 'density': density,
        'dataset_label': dataset_label
    }


def figure_r1_heatmap(data):
    M_hat = data['M_hat']
    Omega = data['Omega']
    n_u, n_m = data['n_users'], data['n_movies']

    row_density = np.sum(Omega, axis=1)
    col_density = np.sum(Omega, axis=0)
    top_rows = np.argsort(row_density)[-50:]
    top_cols = np.argsort(col_density)[-50:]

    M_block = M_hat[np.ix_(top_rows, top_cols)]
    Obs_block = Omega[np.ix_(top_rows, top_cols)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    im = axes[0].imshow(M_block, cmap='RdBu_r', aspect='auto', vmin=-1.0, vmax=1.0, interpolation='nearest')
    axes[0].set_title(r'Estimated Latent $\widehat{\mathbf{M}}$', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Movies (Top 50 most rated)')
    axes[0].set_ylabel('Users (Top 50 most active)')
    plt.colorbar(im, ax=axes[0], shrink=0.82)

    im2 = axes[1].imshow(Obs_block, cmap='Blues', aspect='auto', interpolation='nearest')
    axes[1].set_title(r'Observation Mask $\Omega$ (Same Block)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Movies (Top 50 most rated)')
    axes[1].set_ylabel('Users (Top 50 most active)')
    plt.colorbar(im2, ax=axes[1], shrink=0.82, ticks=[0, 1])

    fig.suptitle(f'R1: Latent Structure — {data["dataset_label"]} '
                 f'(${n_u}\\times{n_m}$, density={data["density"]:.3f})',
                 y=1.02, fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure_r1_heatmap.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] R1 -> {path}")


def figure_r2_diagnostics(data):
    M_hat = data['M_hat']
    Omega = data['Omega']
    Omega_cal = data['Omega_cal']
    Omega_test = data['Omega_test']
    Y_obs = data['Y_obs']
    S_matrix = data['S_matrix']

    calib_scores = -Y_obs[Omega_cal] * M_hat[Omega_cal]
    M_test_hat = M_hat[Omega_test]
    Y_test = Y_obs[Omega_test]
    S_test = S_matrix[Omega_test]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) Histogram of M_hat
    M_all_obs = M_hat[Omega]
    axes[0].hist(M_all_obs, bins=50, density=True, color='steelblue', alpha=0.75, edgecolor='white')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel(r'$\widehat{M}_{ij}$')
    axes[0].set_ylabel('Density')
    axes[0].set_title(r'$\widehat{\mathbf{M}}$ Distribution (Observed)', fontsize=11)

    # (b) Coverage + Singleton rate vs alpha
    alphas = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    coverages, singleton_rates = [], []
    n_cal = len(calib_scores)
    for a in alphas:
        q_level = min(np.ceil((1 - a) * (n_cal + 1)) / n_cal, 1.0)
        Q = np.quantile(calib_scores, q_level)
        sign_ok = np.abs(M_test_hat) > Q
        sets_single = np.sum(sign_ok)
        sets_double = len(M_test_hat) - sets_single
        cov_singleton = np.sum(sign_ok & (np.sign(M_test_hat) == Y_test))
        cov = (cov_singleton + sets_double) / len(M_test_hat)
        coverages.append(cov)
        singleton_rates.append(sets_single / len(M_test_hat))

    ax_cov = axes[1]
    ax_sr = ax_cov.twinx()
    ax_cov.plot(alphas, coverages, '-o', color=COLOR_CMC, markersize=5, label='Coverage')
    ax_sr.plot(alphas, singleton_rates, '-s', color='darkorange', markersize=5, label='Singleton Rate')
    ax_cov.plot([0, 0.35], [0, 0.35], '--', color='gray', linewidth=1, label='Target $1-\\alpha$')
    ax_cov.set_xlabel(r'$\alpha$ (miscoverage level)')
    ax_cov.set_ylabel('Coverage', color=COLOR_CMC)
    ax_sr.set_ylabel('Singleton Rate', color='darkorange')
    ax_cov.set_title(r'Coverage vs $\alpha$', fontsize=11)
    ax_cov.set_xlim(0, 0.32)
    ax_cov.set_ylim(0.6, 1.02)
    ax_sr.set_ylim(0, 1.05)
    lines1, labels1 = ax_cov.get_legend_handles_labels()
    lines2, labels2 = ax_sr.get_legend_handles_labels()
    ax_cov.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=7.5)

    # (c) S_ij vs |M_hat_ij|
    sample_step = max(1, len(M_test_hat) // 2000)
    axes[2].scatter(np.abs(M_test_hat[::sample_step]), S_test[::sample_step],
                    s=3, alpha=0.4, color='steelblue', edgecolors='none')
    axes[2].set_xlabel(r'$|\widehat{M}_{ij}|$')
    axes[2].set_ylabel(r'$\mathbf{S}_{ij}$ (Local SE)')
    axes[2].set_title('Uncertainty vs Signal', fontsize=11)

    fig.suptitle(f'R2: Inference Diagnostics — {data["dataset_label"]}',
                 y=1.03, fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure_r2_diagnostics.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] R2 -> {path}")


def figure_r3_sorted_confidence(data):
    """Replacement for old R3: entries sorted by |M_hat|, showing conformal quantile
    threshold. Clean visual separation of singleton vs doubleton regions."""
    M_hat = data['M_hat']
    Omega_test = data['Omega_test']
    Omega_cal = data['Omega_cal']
    Y_obs = data['Y_obs']
    S_matrix = data['S_matrix']

    M_test_hat = M_hat[Omega_test]
    Y_test = Y_obs[Omega_test]
    S_test = S_matrix[Omega_test]
    calib_scores = -Y_obs[Omega_cal] * M_hat[Omega_cal]

    alpha_show = 0.10
    n_cal = len(calib_scores)
    q_level = min(np.ceil((1 - alpha_show) * (n_cal + 1)) / n_cal, 1.0)
    Q_show = np.quantile(calib_scores, q_level)

    # Sort by |M_hat|
    sort_idx = np.argsort(np.abs(M_test_hat))
    M_sorted = M_test_hat[sort_idx]
    Y_sorted = Y_test[sort_idx]
    S_sorted = S_test[sort_idx]

    # Take 200 evenly spaced entries for clean visualization
    n_total = len(M_sorted)
    indices = np.linspace(0, n_total - 1, 200, dtype=int)
    M_show = M_sorted[indices]
    Y_show = Y_sorted[indices]

    abs_M = np.abs(M_show)
    is_singleton = abs_M > Q_show
    correct = (np.sign(M_show) == Y_show)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: sorted |M_hat| with Q threshold ---
    ax = axes[0]
    x_axis = np.arange(len(M_show))

    # Plot bars for |M_hat|, color by singleton/doubleton
    colors_sgl = [COLOR_SINGLETON if c else '#B0C4DE' for c in correct]
    colors_dbl = [COLOR_DOUBLETON if c else '#F4A582' for c in correct]

    for i in range(len(x_axis)):
        if is_singleton[i]:
            ax.bar(i, abs_M[i], width=1.0, color=colors_sgl[i], alpha=0.85, edgecolor='none')
        else:
            ax.bar(i, abs_M[i], width=1.0, color=colors_dbl[i], alpha=0.85, edgecolor='none')

    ax.axhline(y=Q_show, color='black', linestyle='--', linewidth=1.8, label=f'Conformal Q = {Q_show:.3f}')
    ax.set_xlabel('Test Entries (sorted by $|\\widehat{M}_{ij}|$)')
    ax.set_ylabel(r'$|\widehat{M}_{ij}|$')
    ax.set_title(r'Sorted $|\widehat{M}_{ij}|$ with Conformal Threshold', fontsize=12, fontweight='bold')

    # Legend
    l1 = mlines.Line2D([], [], color=COLOR_SINGLETON, linewidth=3, label='Singleton (correct)')
    l2 = mlines.Line2D([], [], color='#B0C4DE', linewidth=3, label='Singleton (wrong)')
    l3 = mlines.Line2D([], [], color=COLOR_DOUBLETON, linewidth=3, label='Doubleton (correct)')
    l4 = mlines.Line2D([], [], color='#F4A582', linewidth=3, label='Doubleton (wrong)')
    l5 = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.8, label=f'Q = {Q_show:.3f}')
    ax.legend(handles=[l1, l3, l2, l4, l5], loc='upper left', fontsize=8, ncol=1,
              framealpha=0.9, edgecolor='lightgray')

    # --- Right: error bar plot of M_hat with CI bounds ---
    ax2 = axes[1]
    # Take 40 entries from across the spectrum
    idx2 = np.linspace(0, n_total - 1, 40, dtype=int)
    M_40 = M_sorted[idx2]
    Y_40 = Y_sorted[idx2]
    x2 = np.arange(40)

    lower = M_40 - Q_show * S_sorted[idx2]
    upper = M_40 + Q_show * S_sorted[idx2]
    is_sgl_40 = np.abs(M_40) > Q_show

    for i in range(40):
        color = COLOR_SINGLETON if is_sgl_40[i] else COLOR_DOUBLETON
        ax2.plot([i, i], [lower[i], upper[i]], color=color, linewidth=2.5, alpha=0.8)
        marker = 'o' if Y_40[i] == 1 else 's'
        ax2.plot(i, M_40[i], marker, color='black', markersize=5, markeredgewidth=1.2,
                 markerfacecolor=color if is_sgl_40[i] else 'white')

    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.0)
    ax2.axhline(y=Q_show, color='green', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.axhline(y=-Q_show, color='green', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel('40 Test Entries (sorted by $|\\widehat{M}_{ij}|$)')
    ax2.set_ylabel(r'$\widehat{M}_{ij}$ with CI')
    ax2.set_title('Prediction Intervals (40 entries)', fontsize=12, fontweight='bold')

    l6 = mlines.Line2D([], [], color=COLOR_SINGLETON, linewidth=2.5, label='Singleton CI')
    l7 = mlines.Line2D([], [], color=COLOR_DOUBLETON, linewidth=2.5, label='Doubleton CI')
    l8 = mlines.Line2D([], [], color='green', linestyle='--', linewidth=1.0, label=r'$\pm$Q boundary')
    ax2.legend(handles=[l6, l7, l8], loc='upper left', fontsize=8, framealpha=0.9)

    singleton_pct = np.mean(is_singleton) * 100
    fig.suptitle(f'R3: Prediction Confidence — {data["dataset_label"]} '
                 f'($\\alpha={alpha_show}$, Singleton rate={singleton_pct:.1f}%)',
                 y=1.02, fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, 'figure_r3_intervals.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[OK] R3 -> {path}")


if __name__ == "__main__":
    root = find_root()
    os.chdir(root)
    os.makedirs(PAPER_DIR, exist_ok=True)

    data = prepare_data()
    figure_r1_heatmap(data)
    figure_r2_diagnostics(data)
    figure_r3_sorted_confidence(data)

    print(f"\n>>> All real-data figures saved to: {PAPER_DIR}")
