import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

def plot_from_grid_results(grid_dir='../results/grid'):
    files = glob.glob(os.path.join(grid_dir, '*.json'))
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            
        params = data['params']
        res = data['data']
        ranks = [int(k) for k in res.keys()]
        ranks.sort()
        
        c_covs = [res[str(r)]['cmc_cov'] for r in ranks]
        a_covs = [res[str(r)]['asym_cov'] for r in ranks]
        c_lens = [res[str(r)]['cmc_len'] for r in ranks]
        a_lens = [res[str(r)]['asym_len'] for r in ranks]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # 把参数设置体现清楚
        title_str = f"Noise: {params['noise']} | N={params['N']}, p={params['p']}, Split Train/Calib={params['split']}:{1-params['split']:.1f}"
        plt.suptitle(title_str, fontsize=12, fontweight='bold', y=1.05)
        
        # --- 左图：覆盖率 ---
        ax1.plot(ranks, c_covs, '-ob', label='cmc-1bit')
        ax1.plot(ranks, a_covs, '-or', label='asym-chen')
        ax1.axhline(y=0.90, color='gray', linestyle='--')
        ax1.axvline(x=params['r_true'], color='gray', linestyle='--') # 标记真实秩
        ax1.set_xlabel('Hypothesized Rank ($r_{fit}$)')
        ax1.set_ylabel('Empirical Coverage')
        ax1.legend(loc='lower left')
        
        # --- 右图：区间长度 ---
        ax2.plot(ranks, c_lens, '-ob', label='cmc-1bit')
        ax2.plot(ranks, a_lens, '-or', label='asym-chen')
        ax2.axvline(x=params['r_true'], color='gray', linestyle='--')
        ax2.set_xlabel('Hypothesized Rank ($r_{fit}$)')
        ax2.set_ylabel('Average Interval Length')
        ax2.legend()
        
        plt.tight_layout()
        save_name = os.path.basename(file).replace('.json', '.pdf')
        plt.savefig(f"../results/figures/Macro_{save_name}", bbox_inches='tight')
        plt.close()

def plot_micro_unsorted(true_vals, cmc_L, cmc_U, asym_L, asym_U):
    x_axis = np.arange(len(true_vals))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # 先画预测边界线
    for i in range(len(x_axis)):
        ax.plot([x_axis[i], x_axis[i]],[asym_L[i], asym_U[i]], color='red', linewidth=3, alpha=0.3)
        ax.plot([x_axis[i], x_axis[i]], [cmc_L[i], cmc_U[i]], color='blue', linewidth=1.5, alpha=0.8)
    
    # 真实值画在最上面
    ax.plot(x_axis, true_vals, '+', color='green', markersize=8, markeredgewidth=2, label='True Entry')
    
    # 手动添加图例
    import matplotlib.lines as mlines
    l1 = mlines.Line2D([],[], color='red', linewidth=3, alpha=0.3, label='Asym-Chen Bounds')
    l2 = mlines.Line2D([],[], color='blue', linewidth=1.5, label='CMC-1bit Bounds')
    l3 = mlines.Line2D([],[], color='green', marker='+', linestyle='None', markersize=8, label='True Signal')
    ax.legend(handles=[l2, l1, l3], loc='upper right')
    
    ax.set_xlabel('Randomly Sampled Unobserved Entries (Unsorted)')
    ax.set_ylabel('Latent Value')
    ax.set_title('Point-wise Intervals (Scattered View)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    print("="*50)
    print(">>> 正在进行可视化...")
    print("="*50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    print(f"[*] 当前工作目录已锁定至: {current_dir}")
    
    grid_dir = '../results/grid'
    abs_grid_dir = os.path.abspath(grid_dir)
    print(f"[*] 正在检索数据目录: {abs_grid_dir}")
    
    if not os.path.exists(grid_dir):
        print(f"\n[错误] 找不到文件夹: {grid_dir}")
        print("请先运行 experiments 目录下的实验脚本生成数据！")
        sys.exit(1)
        
    files = glob.glob(os.path.join(grid_dir, '*.json'))
    if len(files) == 0:
        print(f"\n[错误] 文件夹存在，但里面没有 .json 文件！")
        print("请检查实验脚本是否成功生成了结果文件。")
        sys.exit(1)
        
    # 4. 执行画图
    print(f"\n[成功] 扫描到 {len(files)} 个实验结果文件，开始渲染图像...\n")
    plot_from_grid_results(grid_dir)
    print("\n" + "="*50)
    print(">>> 所有的图像均已成功渲染并保存！")
    print("="*50)