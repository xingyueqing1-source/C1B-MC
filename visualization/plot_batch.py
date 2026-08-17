import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

def plot_all_jsons(grid_dir):
    files = glob.glob(os.path.join(grid_dir, 'res_*.json'))
    if not files:
        print("未找到结果文件！")
        return

    out_dir = os.path.join('results', 'figures_batch')
    os.makedirs(out_dir, exist_ok=True)
    
    for file in files:
        with open(file, 'r') as f: data = json.load(f)
        params, res = data['params'], data['data']
        
        ranks = sorted([int(k) for k in res.keys()])
        str_ranks = [str(r) for r in ranks]
        
        c_covs = [res[r]['cmc_cov'] for r in str_ranks]
        a_covs = [res[r]['asym_cov'] for r in str_ranks]
        c_lens = [res[r]['cmc_len'] for r in str_ranks]
        a_lens = [res[r]['asym_len'] for r in str_ranks]
        
        # 智能判定 True Rank
        model_type = params.get('model_type', 'UNKNOWN').upper()
        actual_true_rank = 2 if model_type == 'RASCH' else params['r_true']
        
        title_str = (f"Model: {model_type} | Noise: {params['noise'].upper()} | N={params['N']}\n"
                     f"p={params['p']}, Calib={params['calib_ratio']}, Alpha={params.get('alpha_inf', 1.0)}")
        base_name = os.path.basename(file).replace('.json', '')
        
        # --- 1. 宏观图 (Macro) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        plt.suptitle(title_str, fontsize=12, fontweight='bold', y=1.05)
        
        ax1.plot(ranks, c_covs, '-ob', label='CMC (Ours)')
        ax1.plot(ranks, a_covs, '-or', label='Asym (Chen 2023)')
        ax1.axhline(y=0.90, color='gray', linestyle='--')
        ax1.axvline(x=actual_true_rank, color='gray', linestyle='--', label=f'True Rank ({actual_true_rank})')
        ax1.set_xlabel('Hypothesized Rank ($r_{fit}$)')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_ylim(0.65, 1.05)
        
        ax2.plot(ranks, c_lens, '-ob', label='CMC (Ours)')
        ax2.plot(ranks, a_lens, '-or', label='Asym (Chen 2023)')
        ax2.axvline(x=actual_true_rank, color='gray', linestyle='--')
        ax2.set_xlabel('Hypothesized Rank ($r_{fit}$)')
        ax2.set_ylabel('Average Interval Length')
        max_len = max(max(c_lens), max(a_lens))
        ax2.set_ylim(0, max_len * 1.25)
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='lightgray')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Macro_{base_name}.pdf"), bbox_inches='tight')
        plt.close()

        # --- 2. 微观图 (Micro) ---
        plot_data_r12 = data.get('plot_data_r12', {})
        if 'pointwise' in plot_data_r12:
            pw = plot_data_r12['pointwise']
            scores = np.array(plot_data_r12['scores'])
            true_vals = np.array(pw['true'])
            x_axis = np.arange(len(true_vals))
            
            fig, (ax_hist, ax_bounds) = plt.subplots(1, 2, figsize=(12, 4.5))
            
            sns.histplot(scores, stat="density", bins=40, ax=ax_hist, color='#B0C4DE', alpha=0.7)
            x_pdf = np.linspace(-5, 5, 100)
            ax_hist.plot(x_pdf, norm.pdf(x_pdf, 0, 1), color='#D2691E', linewidth=2.5, label='Gaussian(0,1)')
            ax_hist.set_xlim(-5, 5); ax_hist.set_xlabel('Standardized Score'); ax_hist.set_ylabel('Density')
            ax_hist.legend(loc='upper left')
            
            for i in range(len(x_axis)):
                ax_bounds.plot([x_axis[i], x_axis[i]],[pw['asym_L'][i], pw['asym_U'][i]], color='red', linewidth=3, alpha=0.3)
                ax_bounds.plot([x_axis[i], x_axis[i]], [pw['cmc_L'][i], pw['cmc_U'][i]], color='blue', linewidth=1.5, alpha=0.8)
            ax_bounds.plot(x_axis, true_vals, '+', color='green', markersize=8, markeredgewidth=2)
            
            import matplotlib.lines as mlines
            l1 = mlines.Line2D([],[], color='blue', linewidth=1.5, label='CMC Bounds')
            l2 = mlines.Line2D([],[], color='red', linewidth=3, alpha=0.3, label='Asym Bounds')
            l3 = mlines.Line2D([],[], color='green', marker='+', linestyle='None', label='True Signal')
            ax_bounds.legend(handles=[l1, l2, l3], loc='upper right', fontsize=9)
            
            ax_bounds.set_xlabel('Randomly Sampled Unobserved Entries')
            ax_bounds.set_ylabel('Latent Value')
            
            plt.suptitle(f"Micro View (Over-parameterized $r=12$)\n{title_str}", y=1.05, fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"Micro_{base_name}.pdf"), bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir
    while not os.path.exists(os.path.join(root_dir, 'results')):
        parent = os.path.dirname(root_dir)
        if parent == root_dir: sys.exit(1)
        root_dir = parent
    os.chdir(root_dir)
    plot_all_jsons(os.path.join('results', 'grid'))