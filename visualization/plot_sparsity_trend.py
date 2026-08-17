import json, os, glob
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

def plot_sparsity_experiments(grid_dir):
    files = glob.glob(os.path.join(grid_dir, 'res_*Sparsity*.json'))
    if not files:
        print("未找到 Sparsity 结果文件！")
        return

    out_dir = os.path.join('results', 'figures_batch')
    os.makedirs(out_dir, exist_ok=True)
    
    for file in files:
        with open(file, 'r') as f: data = json.load(f)
        params, res = data['params'], data['data']
        
        p_list = data['p_list']
        str_p = [str(p) for p in p_list]
        
        c_covs = [res[p]['cmc_cov'] for p in str_p]
        a_covs = [res[p]['asym_cov'] for p in str_p]
        c_lens = [res[p]['cmc_len'] for p in str_p]
        a_lens = [res[p]['asym_len'] for p in str_p]
        
        model_type = params.get('model_type', 'UNKNOWN').upper()
        
        title_str = (f"Model: {model_type} | Dim: {params['n_rows']}x{params['n_cols']} | Noise: {params['noise'].upper()} \n"
                     f"Calibration Ratio={params['calib_ratio']}")
        base_name = os.path.basename(file).replace('.json', '')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        plt.suptitle(title_str, fontsize=13, fontweight='bold', y=1.05)
        
        # --- 左侧：覆盖率图 ---
        ax1.plot(p_list, c_covs, '-ob', label='CMC (Ours)')
        ax1.plot(p_list, a_covs, '-or', label='Asym (CV + Fisher)')
        ax1.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.2)
        ax1.set_xlabel('Observation Rate ($p$)', fontsize=12)
        ax1.set_ylabel('Empirical Coverage', fontsize=12)
        
        min_cov = min(min(c_covs), min(a_covs))
        ax1.set_ylim(min(0.50, min_cov - 0.05), 1.02) 
        
        # --- 右侧：长度图 ---
        ax2.plot(p_list, c_lens, '-ob', label='CMC (Ours)')
        ax2.plot(p_list, a_lens, '-or', label='Asym (CV + Fisher)')
        ax2.set_xlabel('Observation Rate ($p$)', fontsize=12)
        ax2.set_ylabel('Average Interval Length', fontsize=12)
        
        max_len = max(max(c_lens), max(a_lens))
        ax2.set_ylim(0, max_len * 1.25)
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='lightgray')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Trend_{base_name}.pdf"), bbox_inches='tight')
        plt.close()
        print(f"  已生成稀疏度相变图: Trend_{base_name}.pdf")

if __name__ == "__main__":
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir
    while not os.path.exists(os.path.join(root_dir, 'results')):
        parent = os.path.dirname(root_dir)
        if parent == root_dir: sys.exit(1)
        root_dir = parent
    os.chdir(root_dir)
    plot_sparsity_experiments(os.path.join('results', 'grid'))