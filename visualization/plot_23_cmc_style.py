import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.0

COLOR_CMC = 'blue'
COLOR_ASYM = 'red'
COLOR_TRUE = 'green'

def plot_figure_1_macro(data, r_true=5):
    fig, axes = plt.subplots(len(data), 2, figsize=(10, 4 * len(data)))
    
    for idx, scene in enumerate(data):
        ranks = scene['ranks']
        sc_data = scene['data']
        
        cmc_covs = [sc_data[str(r)]['cmc_cov'] for r in ranks]
        asym_covs =[sc_data[str(r)]['asym_cov'] for r in ranks]
        cmc_lens = [sc_data[str(r)]['cmc_len'] for r in ranks]
        asym_lens = [sc_data[str(r)]['asym_len'] for r in ranks]
        
        ax_cov, ax_len = axes[idx]
        
        # --- 覆盖率图 ---
        ax_cov.plot(ranks, cmc_covs, marker='o', markersize=5, color=COLOR_CMC, label='cmc-1bit')
        ax_cov.plot(ranks, asym_covs, marker='o', markersize=5, color=COLOR_ASYM, label='asym-chen')
        ax_cov.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.2)
        ax_cov.axvline(x=r_true, color='gray', linestyle='--', linewidth=1.2)
        
        ax_cov.set_ylim(0.7, 1.0)
        ax_cov.set_xlabel('hypothesized rank')
        ax_cov.set_ylabel('AvgCov')
        ax_cov.set_title(scene['scenario'], y=-0.25, fontsize=12)
        ax_cov.legend(loc='lower left', fontsize=9)
        
        # --- 长度图 ---
        ax_len.plot(ranks, cmc_lens, marker='o', markersize=5, color=COLOR_CMC, label='cmc-1bit')
        ax_len.plot(ranks, asym_lens, marker='o', markersize=5, color=COLOR_ASYM, label='asym-chen')
        ax_len.axvline(x=r_true, color='gray', linestyle='--', linewidth=1.2)
        
        ax_len.set_xlabel('hypothesized rank')
        ax_len.set_ylabel('AvgLength')
        ax_len.legend(fontsize=9)
        
    plt.tight_layout()
    os.makedirs('../results/figures', exist_ok=True)
    plt.savefig('../results/figures/Figure_1_Macro_Trend.pdf', bbox_inches='tight')
    print(" 生成: Figure_1_Macro_Trend.pdf")

def plot_figure_2_micro_unsorted(data, target_scenario=1, target_r=8):
    scene = data[target_scenario]
    r_data = scene['data'][str(target_r)]
    
    scores = np.array(r_data['scores'])
    pw = r_data['pointwise']
    
    # 提取50个点，不排序
    true_vals = np.array(pw['true'])
    cmc_L = np.array(pw['cmc_L'])
    cmc_U = np.array(pw['cmc_U'])
    asym_L = np.array(pw['asym_L'])
    asym_U = np.array(pw['asym_U'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # --- Subplot 1: 直方图 vs 高斯钟形曲线 ---
    sns.histplot(scores, stat="density", bins=40, ax=ax1, color='#B0C4DE', alpha=0.7, label='empirical scores')
    x_pdf = np.linspace(-5, 5, 100)
    y_pdf = norm.pdf(x_pdf, 0, 1)
    ax1.plot(x_pdf, y_pdf, color='#D2691E', linewidth=2.5, label='Gaussian(0,1)')
    
    ax1.set_xlim(-5, 5)
    ax1.set_xlabel('Score') 
    ax1.set_ylabel('Density')
    ax1.set_title('Standardized Residuals Distribution', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    
    # --- Subplot 2: 上下界
    x_axis = np.arange(len(true_vals))
    
    for i in range(len(x_axis)):
        ax2.plot([x_axis[i], x_axis[i]],[asym_L[i], asym_U[i]], color=COLOR_ASYM, linewidth=3, alpha=0.3)
        ax2.plot([x_axis[i], x_axis[i]], [cmc_L[i], cmc_U[i]], color=COLOR_CMC, linewidth=1.5, alpha=0.9)
    
    ax2.plot(x_axis, true_vals, '+', color=COLOR_TRUE, markersize=8, markeredgewidth=2, label='true entry')
    
    # 手动添加图例
    import matplotlib.lines as mlines
    l1 = mlines.Line2D([],[], color=COLOR_CMC, linewidth=1.5, label='cmc-1bit')
    l2 = mlines.Line2D([],[], color=COLOR_ASYM, linewidth=3, alpha=0.3, label='asym-chen')
    l3 = mlines.Line2D([],[], color=COLOR_TRUE, marker='+', linestyle='None', markersize=8, label='true entry')
    ax2.legend(handles=[l1, l2, l3], loc='upper left', fontsize=9)
    
    ax2.set_xlabel('index (Randomly sampled)')
    ax2.set_ylabel('unobserved entries')
    ax2.set_title('Lower and upper bounds', fontsize=11)
    
    plt.suptitle(f"{scene['scenario']} (Hypothesized Rank: $r={target_r}$)", y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../results/figures/Figure_2_Micro_r{target_r}.pdf', bbox_inches='tight')
    print(f" 生成: Figure_2_Micro_r{target_r}.pdf")

if __name__ == "__main__":
    import sys
    print("="*50)
    print(">>> 正在进行可视化...")
    print("="*50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    json_path = '../results/exp_e_cmc_style.json'
    abs_json_path = os.path.abspath(json_path)
    
    print(f"[*] 正在检索数据文件: {abs_json_path}")
    
    if not os.path.exists(json_path):
        print(f"\n[错误] 找不到文件: {json_path}")
        print("请确认您是否成功运行了 exp_e_cmc_style.py！")
        sys.exit(1)
        
    # 执行画图
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print("\n[成功] 文件加载成功，开始渲染图像...\n")
    
    # 1. 渲染宏观图
    plot_figure_1_macro(data, r_true=5)
    
    # 2. 渲染微观图，选取 Setting 2: 重尾噪声, 且发生了过参数化 r=8 的场景
    plot_figure_2_micro_unsorted(data, target_scenario=1, target_r=8)
    
    print("\n" + "="*50)
    print(">>> 所有的图像均已成功渲染并保存至 results/figures/ 目录！")
    print("="*50)
    
    # 自动弹出图片展示
    plt.show()