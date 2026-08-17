import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
COLOR_CMC = 'blue'
COLOR_ASYM = 'red'

def auto_plot_best_boxplot(grid_dir):
    # 针对 r=12 的微观数据绘制箱线图，寻找 General 模型 + 重尾噪声 的情况
    search_pattern = os.path.join(grid_dir, '*general*heavy_tail*.json')
    files = glob.glob(search_pattern)
    
    if not files:
        print("[错误] 未找到包含 'general' 和 'heavy_tail' 的实验数据。")
        return
    target_file = files[0]
    print(f"[*] 成功找到数据文件: {os.path.basename(target_file)}")
    
    with open(target_file, 'r') as f:
        data = json.load(f)
        
    # 提取 r=12 的微观推断数据
    if 'plot_data_r12' not in data or 'pointwise' not in data['plot_data_r12']:
        print("[错误] 该文件里没有 r=12 的微观数据 (plot_data_r12/pointwise)！")
        return
        
    pw = data['plot_data_r12']['pointwise']
    
    # 计算 50 个独立测试点的区间长度
    cmc_lens = np.array(pw['cmc_U']) - np.array(pw['cmc_L'])
    asym_lens = np.array(pw['asym_U']) - np.array(pw['asym_L'])
    
    # 组装 Pandas 数据框，供 Seaborn 渲染
    df_cmc = pd.DataFrame({'Interval Length': cmc_lens, 'Method': 'CMC-1bit (Ours)'})
    df_asym = pd.DataFrame({'Interval Length': asym_lens, 'Method': 'Asym (Chen 2023)'})
    df = pd.concat([df_cmc, df_asym], ignore_index=True)

    sns.set_theme(style="ticks")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    classic_teal = '#62A3A2' 
    
    sns.boxplot(x='Method', y='Interval Length', data=df, ax=ax, 
                color=classic_teal, width=0.45, 
                boxprops=dict(edgecolor='black', linewidth=1.2),
                medianprops=dict(color='black', linewidth=1.5, linestyle='-'), # 黑色实线中位数
                whiskerprops=dict(color='black', linewidth=1.2, linestyle='--'), 
                capprops=dict(color='black', linewidth=1.2),
                flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', 
                                markersize=4, markeredgewidth=1.0)) # 空心圆圈为异常值
    
    # 提取参数生成标题
    params = data['params']
    model_type = params.get('model_type', 'GENERAL').upper()
    title_str = f"Model: {model_type} | Noise: {params.get('noise', '').upper()} | $N={params.get('N', '')}$"
    
    # 调整字体大小
    ax.set_title(f'Distribution of Interval Lengths\n(Over-parameterized $r_{{fit}}=12$)\n{title_str}', 
                 fontsize=12, pad=15)
    ax.set_ylabel('Confidence Interval Length', fontsize=12)
    ax.set_xlabel('') 
    
    # 4. 坐标轴只留左和下，刻度向外
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.tick_params(width=1.0, direction='out', labelsize=11)
    
    # 动态调整 Y 轴，确保下限适当
    max_len = df['Interval Length'].max()
    ax.set_ylim(0, max_len * 1.1)
    
    # 保存图像
    out_dir = os.path.join('results', 'figures_batch')
    os.makedirs(out_dir, exist_ok=True)
    save_name = os.path.basename(target_file).replace('.json', '_CLASSIC_BOXPLOT.pdf')
    save_path = os.path.join(out_dir, save_name)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  成功生成箱线图: {save_path}")
    plt.show()

if __name__ == "__main__":
    import sys
    print("="*50)
    print(">>> 开始进行箱线图绘制...")
    print("="*50)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir
    while not os.path.exists(os.path.join(root_dir, 'results')):
        parent = os.path.dirname(root_dir)
        if parent == root_dir: 
            print("[错误] 找不到 'results' 文件夹！")
            sys.exit(1)
        root_dir = parent
        
    os.chdir(root_dir)
    print(f"[*] 成功锁定项目根目录: {root_dir}")
    
    grid_dir = os.path.join('results', 'grid')
    if not os.path.exists(grid_dir):
        print(f"[错误] 找不到数据文件夹: {grid_dir}")
        sys.exit(1)
        
    # 调用画图函数
    auto_plot_best_boxplot(grid_dir)