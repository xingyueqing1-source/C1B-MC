import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})
METHOD_COLORS = {
    'Asym (Chen 2023)': '#E64B35',  # 砖红
    'CMC-1bit (Ours)':  '#3C5488'   # 藏蓝
}

def plot_json_with_nature_style(grid_dir):
    files = glob.glob(os.path.join(grid_dir, '*general*heavy_tail*.json'))
    if not files:
        print("未找到符合条件的结果文件！")
        return

    out_dir = os.path.join('results', 'figures_nature')
    os.makedirs(out_dir, exist_ok=True)
    
    # 遍历找到的符合条件的恶劣环境 JSON
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            
        if 'plot_data_r12' not in data or 'pointwise' not in data['plot_data_r12']:
            continue
            
        pw = data['plot_data_r12']['pointwise']
        params = data['params']
        
        # 提取 50 个测试点的区间长度
        cmc_lens = np.array(pw['cmc_U']) - np.array(pw['cmc_L'])
        asym_lens = np.array(pw['asym_U']) - np.array(pw['asym_L'])
        
        # 组装 DataFrame 适配 Seaborn
        df_cmc = pd.DataFrame({'Width': cmc_lens, 'Method': 'CMC-1bit (Ours)'})
        df_asym = pd.DataFrame({'Width': asym_lens, 'Method': 'Asym (Chen 2023)'})
        df = pd.concat([df_cmc, df_asym], ignore_index=True)
        
        method_order = ['Asym (Chen 2023)', 'CMC-1bit (Ours)']
        
        fig, ax_wid = plt.subplots(figsize=(6, 5))
        
        sns.boxplot(x='Method', y='Width', data=df, order=method_order, 
                    ax=ax_wid, palette=METHOD_COLORS, width=0.5, 
                    boxprops=dict(alpha=0.9, edgecolor='black', linewidth=1.2),
                    medianprops=dict(color='yellow', linewidth=2.5),
                    showfliers=False)
        
        sns.stripplot(x='Method', y='Width', data=df, order=method_order, 
                      ax=ax_wid, palette=METHOD_COLORS, alpha=0.5, jitter=True, size=4, edgecolor='black', linewidth=0.5)

        # 动态 Y 轴上限
        y_max = df['Width'].max() * 1.2
        ax_wid.set_ylim(0, y_max)
        
        ax_wid.set_xlabel('')
        ax_wid.set_xticklabels(method_order, fontsize=12, fontweight='bold')
        ax_wid.set_ylabel('Confidence Interval Width', fontweight='bold')
        
        title_str = f"Model: {params['model_type'].upper()} | Noise: {params['noise'].upper()} | N={params.get('n_rows', params.get('N'))}"
        ax_wid.set_title(f'Point-wise Interval Widths (Over-parameterized $r_{{fit}}=12$)\n{title_str}', pad=15)
        
        ax_wid.spines['top'].set_visible(False)
        ax_wid.spines['right'].set_visible(False)
        ax_wid.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # 保存图像
        base_name = os.path.basename(file).replace('.json', '')
        save_path = os.path.join(out_dir, f"Nature_Boxplot_{base_name}.pdf")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f" 成功生成箱线图: {save_path}")
        plt.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir
    while not os.path.exists(os.path.join(root_dir, 'results')):
        parent = os.path.dirname(root_dir)
        if parent == root_dir: 
            import sys
            sys.exit(1)
        root_dir = parent
    os.chdir(root_dir)
    
    plot_json_with_nature_style(os.path.join('results', 'grid'))