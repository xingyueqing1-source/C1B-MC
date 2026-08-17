import json, os, glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

def plot_samplesize_experiments(grid_dir):
    files = glob.glob(os.path.join(grid_dir, 'res_*SampleSize*.json'))
    if not files:
        print("未找到 SampleSize 的结果文件")
        return

    out_dir = os.path.join('results', 'figures_batch')
    os.makedirs(out_dir, exist_ok=True)
    
    for file in files:
        with open(file, 'r') as f: data = json.load(f)
        params, res = data['params'], data['data']
        
        N_list = data['N_list']
        str_N = [str(n) for n in N_list]
        
        c_covs = [res[n]['cmc_cov'] for n in str_N]
        a_covs = [res[n]['asym_cov'] for n in str_N]
        c_lens = [res[n]['cmc_len'] for n in str_N]
        a_lens = [res[n]['asym_len'] for n in str_N]
        
        model_type = params.get('model_type', 'UNKNOWN').upper()
        title_str = f"Model: {model_type} | Noise: {params['noise'].upper()} | p={params['p']}, Calib={params['calib_ratio']}"
        base_name = os.path.basename(file).replace('.json', '')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
        plt.suptitle(title_str, fontsize=13, fontweight='bold', y=1.05)
        
        # --- 左侧：覆盖率图 ---
        ax1.plot(N_list, c_covs, '-ob', label='CMC (Ours)')
        ax1.plot(N_list, a_covs, '-or', label='Asym (Chen 2023)')
        ax1.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.2)
        ax1.set_xlabel('Matrix Dimension ($N$)', fontsize=11)
        ax1.set_ylabel('Empirical Coverage', fontsize=11)
        
        # 动态Y轴下限，展示小样本下 Asym 跌破 90%
        min_cov = min(min(c_covs), min(a_covs))
        ax1.set_ylim(min(0.85, min_cov - 0.05), 1.02) 
        
        # --- 右侧：长度图 ---
        ax2.plot(N_list, c_lens, '-ob', label='CMC (Ours)')
        ax2.plot(N_list, a_lens, '-or', label='Asym (Chen 2023)')
        ax2.set_xlabel('Matrix Dimension ($N$)', fontsize=11)
        ax2.set_ylabel('Average Interval Length', fontsize=11)
        
        max_len = max(max(c_lens), max(a_lens))
        ax2.set_ylim(0, max_len * 1.25)
        
        # 图例只保留在右图
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='lightgray')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Trend_{base_name}.pdf"), bbox_inches='tight')
        plt.close()
        print(f"  已生成相变图: Trend_{base_name}.pdf")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir
    while not os.path.exists(os.path.join(root_dir, 'results')):
        parent = os.path.dirname(root_dir)
        if parent == root_dir: sys.exit(1)
        root_dir = parent
    os.chdir(root_dir)
    plot_samplesize_experiments(os.path.join('results', 'grid'))