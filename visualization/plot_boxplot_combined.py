"""
Improved boxplot visualization using EXISTING experiment data.
Reads from results/grid/*.json and generates clean publication-quality boxplots.

Fixes from the teammate's version:
  - Clean grid layout with proper spacing
  - Nature color palette preserved
  - Median annotations
  - No overlapping text, proper font sizes
  - Saved as PDF for publication
"""
import json, os, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12})

# Nature-style palette (from teammate)
METHOD_COLORS = {
    'C1B-MC':  '#3C5488',  # dark blue
    'Asym':    '#E64B35',  # red
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAPER_DIR = os.path.join(PROJECT_ROOT, 'paper', 'figures')
GRID_DIR = os.path.join(PROJECT_ROOT, 'results', 'grid')


def load_all_results():
    """Load all grid search JSONs, extract per-rank coverage and width data."""
    files = glob.glob(os.path.join(GRID_DIR, '*.json'))
    if not files:
        print(f"ERROR: No JSON files found in {GRID_DIR}")
        return pd.DataFrame()

    rows = []
    for fp in files:
        with open(fp, 'r') as f:
            data = json.load(f)
        params = data.get('params', {})
        results = data.get('data', {})

        # Determine model type
        model = params.get('model_type', 'UNKNOWN')
        noise = params.get('noise', 'UNKNOWN')
        N = params.get('N', '?')
        p_val = params.get('p', '?')
        alpha = params.get('alpha_inf', '?')

        # Create a readable label
        if 'Server_SampleSize' in os.path.basename(fp) or 'Sparsity' in os.path.basename(fp):
            continue  # skip trend data, only want grid search with rank info

        label = f"{model}+{noise}"
        if N != '?':
            label += f"\nN={N}"

        for r_str, vals in results.items():
            if not isinstance(vals, dict):
                continue
            if 'cmc_cov' not in vals:
                continue
            rows.append({
                'Scenario': label,
                'Rank': int(r_str),
                'N': N,
                'p': p_val,
                'alpha': alpha,
                'C1B-MC': vals.get('cmc_cov', np.nan),
                'C1B-MC_len': vals.get('cmc_len', np.nan),
                'Asym': vals.get('asym_cov', np.nan),
                'Asym_len': vals.get('asym_len', np.nan),
                'file': os.path.basename(fp),
            })

    return pd.DataFrame(rows)


def plot_boxplots(df):
    """Generate improved combined coverage + width boxplot."""
    if df.empty:
        print("No data to plot.")
        return

    df['Scenario_short'] = df['Scenario'].apply(lambda x: x.replace('\n', ' '))

    os.makedirs(PAPER_DIR, exist_ok=True)

    # Reshape to long format
    df_long = df.melt(id_vars=['Scenario_short', 'Rank', 'N', 'p'],
                       value_vars=['C1B-MC', 'Asym'],
                       var_name='Method', value_name='Coverage')

    df_long_w = df.melt(id_vars=['Scenario_short', 'Rank', 'N', 'p'],
                         value_vars=['C1B-MC_len', 'Asym_len'],
                         var_name='Method', value_name='Width')
    df_long_w['Method'] = df_long_w['Method'].str.replace('_len', '')

    # ---- Combined dual-panel ----
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(18, 6))

    # Coverage
    bp3a = sns.boxplot(x='Scenario_short', y='Coverage', hue='Method',
                       data=df_long, ax=ax3a,
                       palette=METHOD_COLORS,
                       width=0.7, linewidth=1.2, fliersize=4,
                       showmeans=True,
                       meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markersize': 7})
    ax3a.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3a.set_ylabel('Empirical Coverage', fontsize=13)
    ax3a.set_xlabel('Scenario', fontsize=12)
    ax3a.set_ylim(0.2, 1.05)
    ax3a.legend(loc='lower right', framealpha=0.95, fontsize=11)
    ax3a.tick_params(axis='x', rotation=25, labelsize=10)
    ax3a.set_title('Coverage', fontsize=14, fontweight='bold')

    # Annotate
    for i, scen in enumerate(df_long['Scenario_short'].unique()):
        for j, method in enumerate(['C1B-MC', 'Asym']):
            subset = df_long[(df_long['Scenario_short'] == scen) & (df_long['Method'] == method)]
            if len(subset) > 0:
                med = subset['Coverage'].median()
                x_pos = i + (j - 0.5) * 0.4
                ax3a.annotate(f'{med:.3f}', xy=(x_pos, med + 0.015),
                             ha='center', va='bottom', fontsize=7.5,
                             color=METHOD_COLORS[method], fontweight='bold', rotation=90)

    # Width
    bp3b = sns.boxplot(x='Scenario_short', y='Width', hue='Method',
                       data=df_long_w, ax=ax3b,
                       palette=METHOD_COLORS,
                       width=0.7, linewidth=1.2, fliersize=4,
                       showfliers=False,
                       showmeans=True,
                       meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markersize': 7})
    ax3b.set_ylabel('Average Interval Length', fontsize=13)
    ax3b.set_xlabel('Scenario', fontsize=12)
    ax3b.legend(loc='upper right', framealpha=0.95, fontsize=11)
    ax3b.tick_params(axis='x', rotation=25, labelsize=10)
    ax3b.set_title('Interval Length', fontsize=14, fontweight='bold')

    fig3.suptitle('Aggregate Performance Summary Across All Configurations',
                  y=1.005, fontsize=15, fontweight='bold')
    plt.subplots_adjust(wspace=0.3)
    combo_path = os.path.join(PAPER_DIR, 'boxplot_combined.pdf')
    fig3.savefig(combo_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[OK] Combined boxplot -> {combo_path}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading from: {GRID_DIR}")
    df = load_all_results()
    if df.empty:
        print("No data found. Run grid search experiments first.")
        sys.exit(1)
    print(f"\nLoaded {len(df)} data points from grid search results.")
    plot_boxplots(df)
    print("\nDone.")
