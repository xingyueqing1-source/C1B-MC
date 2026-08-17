import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": "#F8F9FA", 
    "grid.color": "#E5E5E5",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2
})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

METHOD_COLORS = {
    'Asymptotic':   '#D65F5F',  
    'CV+ CQR':      '#6ACCBC',  
    'CV+ Fisher':   '#10A37F',  
    'CV+ Bootstrap':'#4C72B0',  
    'Split-CP':     '#8172B2',  
    'C1B-MC':       '#2B4B8A',  
}

def render_2x4_grid_boxplot(data_path):
    if not os.path.exists(data_path):
        print(f"[错误]未找到数据文件: {data_path}")
        return

    # 兼容 JSON 或 CSV 的读取
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(data_path)

    desired_order = ['Asymptotic', 'CV+ CQR', 'CV+ Fisher', 'CV+ Bootstrap', 'Split-CP', 'C1B-MC']
    method_order = [m for m in desired_order if m in df['Method'].unique()]

    scenarios = df['scenario'].unique()
    p_scans = sorted(df['p_obs'].unique())

    out_dir = os.path.join(os.path.dirname(data_path), 'figures_nature')
    os.makedirs(out_dir, exist_ok=True)

    for scenario in scenarios:
        df_scen = df[df['scenario'] == scenario]
        if df_scen.empty: continue
        
        print(f"  -> 正在渲染场景: {scenario}")
        
        fig, axes = plt.subplots(2, len(p_scans), figsize=(4.5 * len(p_scans), 8))
        
        for j, p in enumerate(p_scans):
            df_p = df_scen[df_scen['p_obs'] == p]

            # ----------------------------------------------------
            # Row 0: 覆盖率
            # ----------------------------------------------------
            ax_cov = axes[0, j]
            sns.boxplot(x='Method', y='Coverage', data=df_p, order=method_order,
                        palette=METHOD_COLORS, ax=ax_cov, width=0.6,
                        boxprops=dict(alpha=0.9, edgecolor='black', linewidth=1.2),
                        medianprops=dict(color='yellow', linewidth=2.5),
                        showfliers=False)
            
            # 添加散点
            sns.stripplot(x='Method', y='Coverage', data=df_p, order=method_order,
                          color='black', alpha=0.2, size=3.5, jitter=True, ax=ax_cov)

            ax_cov.axhline(0.90, color='black', linestyle='--', linewidth=1.5, zorder=0)
            ax_cov.set_title(f'$p_{{obs}} = {p}$', fontsize=15, fontweight='bold', pad=12)
            ax_cov.set_xlabel('')
            ax_cov.set_xticklabels([]) 
            
            if j == 0:
                ax_cov.set_ylabel('Empirical Coverage Rate', fontsize=13, fontweight='bold')
            else:
                ax_cov.set_ylabel('')

            ax_cov.set_ylim(0.68, 1.05)

            # ----------------------------------------------------
            # Row 1: 区间长度
            # ----------------------------------------------------
            ax_wid = axes[1, j]
            sns.boxplot(x='Method', y='Width', data=df_p, order=method_order,
                        palette=METHOD_COLORS, ax=ax_wid, width=0.6,
                        boxprops=dict(alpha=0.9, edgecolor='black', linewidth=1.2),
                        medianprops=dict(color='yellow', linewidth=2.5),
                        showfliers=False)
            
            sns.stripplot(x='Method', y='Width', data=df_p, order=method_order,
                          color='black', alpha=0.2, size=3.5, jitter=True, ax=ax_wid)

            ax_wid.set_xlabel('')
            ax_wid.set_xticklabels(method_order, rotation=35, ha='right', fontsize=11, fontweight='bold')

            if j == 0:
                ax_wid.set_ylabel('Interval Width', fontsize=13, fontweight='bold')
            else:
                ax_wid.set_ylabel('')

            # 动态调整 Y 轴上限
            q95 = df_p['Width'].quantile(0.95)
            y_max = min(df_p['Width'].max() * 1.05, q95 * 1.5) if pd.notna(q95) else df_p['Width'].max() * 1.1
            ax_wid.set_ylim(0, max(y_max, 0.1))

        # ==========================================
        # 3. 图例
        # ==========================================
        sns.despine(fig)
        
        import matplotlib.patches as mpatches
        handles = [plt.Rectangle((0,0),1,1, color=METHOD_COLORS[m], ec='black') for m in method_order]
        fig.legend(handles, method_order, loc='upper center', bbox_to_anchor=(0.5, 1.08),
                   ncol=len(method_order), frameon=True, fontsize=13, edgecolor='black', facecolor='white')

        scene_name = "General Model + Heavy-tailed Noise" if scenario == "HIGH_RANK" else "Rasch Model + Logistic Noise"
        fig.suptitle(f'Coverage & Width Distributions | {scene_name}', y=1.15, fontsize=18, fontweight='bold')

        plt.tight_layout()
        
        save_path = os.path.join(out_dir, f'GridBoxplot_2x4_{scenario}.pdf')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f" 成功渲染并保存图片: {save_path}")

    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    while not os.path.exists(os.path.join(project_root, 'results')):
        project_root = os.path.dirname(project_root)
    os.chdir(project_root)
    data_file = os.path.join('results', 'boxplot_benchmark.json') 
   
    print("=" * 50)
    print(">>> 启动 2x4 顶会级箱线图渲染引擎...")
    render_2x4_grid_boxplot(data_file)
    print("=" * 50)