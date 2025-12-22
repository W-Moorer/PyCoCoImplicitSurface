import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import pandas as pd # 不再需要 pandas 读取 CSV
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl
import sys
import os

# ==========================================
# Configuration: 请在这里设置文件路径和常量
# ==========================================
# TODO: 1. 请将下面的路径替换为你存放 radii.txt 的实际绝对路径或相对路径
RADII_TXT_PATH = r'E:\PyCharm\PyCoCoImplicitSurface\output\cfpu_input\leftGear_surface_cellnormals_cfpu_input\radii.txt'

# TODO: 2. 设置默认策略的常数半径值 (根据你之前的统计数据填入)
# 基于你提供的信息: min = median = max = 0.00023320759726015442
DEFAULT_RADIUS_CONST = 0.00023320759726015442

# --- 检查文件是否存在，不存在则生成一个样例文件用于演示 ---
if not os.path.exists(RADII_TXT_PATH):
    print(f"[Warning] 文件 '{RADII_TXT_PATH}' 不存在。")
    print("正在当前目录下生成一个样例 'radii_sample.txt' (仅含 new_radius 列) 用于演示...")
    # 生成一些基于之前统计特征的假数据用于演示
    N_SAMPLE = 2743
    # 模拟 delta=0.6 的分布特征
    np.random.seed(42)
    sim_adaptive = np.random.lognormal(mean=np.log(0.00021), sigma=0.35, size=N_SAMPLE)
    sim_adaptive = np.clip(sim_adaptive, DEFAULT_RADIUS_CONST * 0.25, DEFAULT_RADIUS_CONST * 1.75)
    
    RADII_TXT_PATH = 'radii_sample.txt'
    # 保存为纯文本，一行一个数字
    np.savetxt(RADII_TXT_PATH, sim_adaptive, fmt='%.18e')
    print(f"样例文件已生成: {os.path.abspath(RADII_TXT_PATH)}")
    print("请在运行后查看结果，然后修改脚本中的 RADII_TXT_PATH 为你的真实文件路径。")
    print("-" * 30)


# ==========================================
# 0. 设置全局字体为 Times New Roman
# ==========================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ==========================================
# 1. 数据加载与生成
# ==========================================

# --- 1.1 加载真实的 Panel A 数据 (从纯文本文件) ---
print(f"正在读取文件: {RADII_TXT_PATH}...")
try:
    # [修改点]: 使用 np.loadtxt 读取纯文本数值文件
    # 假设文件只有一列 new_radius 值
    radii_adaptive = np.loadtxt(RADII_TXT_PATH)
    
    N_PATCHES = len(radii_adaptive)
    print(f"成功加载 {N_PATCHES} 个 Patch 的新半径数据。")

    # [修改点]: 根据常量生成默认半径数组
    print(f"使用常数默认半径: {DEFAULT_RADIUS_CONST}")
    radii_default = np.full(N_PATCHES, DEFAULT_RADIUS_CONST)

except FileNotFoundError:
    print(f"错误: 找不到文件 '{RADII_TXT_PATH}'。请检查路径设置。")
    sys.exit(1)
except ValueError as e:
    print(f"错误: 文件格式不正确，无法解析为数值。请确保文件只包含数字，每行一个。\n详细信息: {e}")
    sys.exit(1)
except Exception as e:
    print(f"读取文件时发生未知错误: {e}")
    sys.exit(1)

# 计算用于标注的真实统计量
default_val = radii_default[0]
# 计算比率，处理可能的除零异常
with np.errstate(divide='ignore', invalid='ignore'):
    ratios = radii_adaptive / radii_default
    ratios = ratios[np.isfinite(ratios)]

if len(ratios) > 0:
    median_ratio = np.median(ratios)
else:
    median_ratio = 0.0 # Fallback

# --- 1.2 模拟 Panel B 数据 (基于统计指标) ---
# 注意：由于未提供节点计数的真实文件，此处仍使用基于 delta=0.6 统计数据的模拟。
print("正在根据统计指标模拟节点计数数据 (Panel B)...")
# Default Stats: Median=67, Mean=75.5, Max=396, Std=47.9
np.random.seed(42) # 保证可复现性
nodes_default = np.random.gamma(shape=2.5, scale=30, size=N_PATCHES) + 18
nodes_default = np.clip(nodes_default, 19, 396)

# Adaptive Stats (delta=0.6): Min=40, Median=56, Mean=68.4, Max=201, Std=30.3
# 使用移位的 Gamma 分布模拟右偏且有硬下限的分布
nodes_adaptive = np.random.gamma(shape=1.1, scale=27, size=N_PATCHES) + 40
nodes_adaptive = np.clip(nodes_adaptive, 40, 201)

# ==========================================
# 2. 绘图设置
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1, rc={"font.family": "serif", "font.serif": ["Times New Roman"]})
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
colors = ["#34495e", "#e74c3c"]

# ==========================================
# 3. Panel A: 半径分布对比 (Violin Plot - 真实数据)
# ==========================================
ax_a = axes[0]
data_radii = [radii_default, radii_adaptive]
labels_radii = ['Default Strategy\n(Constant Radius)', 'Adaptive Strategy\n(radii.txt, $\delta=0.6$)']

# 绘制小提琴图
sns.violinplot(data=data_radii, ax=ax_a, palette=colors, inner="box", linewidth=1.5, cut=0)

ax_a.set_ylabel('Patch Radius Value ($R_i$)', fontsize=12, fontweight='bold')
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(labels_radii, fontsize=11)
ax_a.set_title('(A) Comparison of Patch Radius Strategies (Actual Data)', fontsize=14, loc='left', pad=15)
ax_a.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_a.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# --- 标注文本 (基于真实计算) ---
# 标注默认值
ax_a.annotate('Fixed Constant\n(std = 0.0)', 
              xy=(0, default_val), xytext=(0.3, default_val*1.25),
              arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, ha='center')

# 标注自适应中位数 (动态位置)
adaptive_median_pos = np.median(radii_adaptive)
# 确定文本和箭头的垂直位置，避免重叠
text_y_pos = max(adaptive_median_pos * 1.35, default_val * 1.1)

ax_a.annotate(f'Adaptive Distribution\n(Median ratio $\\approx$ {median_ratio:.2f}, Wider Spread)', 
              xy=(1, adaptive_median_pos), xytext=(1.35, text_y_pos),
              arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, ha='center')


# ==========================================
# 4. Panel B: Patch内节点数分布 (KDE Plot - 统计模拟)
# ==========================================
ax_b = axes[1]

# 绘制密度图
sns.kdeplot(nodes_default, ax=ax_b, color=colors[0], fill=True, alpha=0.3, linewidth=2, label='Default Strategy', bw_adjust=0.8)
# 增加 clip 防止 KDE 越过硬边界
sns.kdeplot(nodes_adaptive, ax=ax_b, color=colors[1], fill=True, alpha=0.4, linewidth=2, label='Adaptive Strategy ($\delta=0.6$)', bw_adjust=0.8, clip=(40, 205))

# 添加目标参考线 N=40
ax_b.axvline(x=40, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
y_max = ax_b.get_ylim()[1]
ax_b.text(42, y_max*0.85, 'Design Target\nMin $N_i \\geq$ 40 (Achieved)', color='black', fontsize=10, ha='left')

ax_b.set_xlabel('Number of Nodes per Patch ($N_i$)', fontsize=12, fontweight='bold')
ax_b.set_ylabel('Density (Frequency)', fontsize=12, fontweight='bold')
ax_b.set_title('(B) Resulting Node Count Distribution per Patch (Simulated)', fontsize=14, loc='left', pad=15)
# 设置X轴范围，确保覆盖到默认策略的最大值
ax_b.set_xlim(0, max(nodes_default.max(), nodes_adaptive.max()) + 50) 
ax_b.legend(fontsize=11)

# --- 标注文本 ---
ax_b.annotate('High Variance\n(Range: 19-396, std $\\approx$ 48)', xy=(150, 0.002), xytext=(180, 0.006),
              arrowprops=dict(arrowstyle='->', color=colors[0]), color=colors[0], fontsize=10)

# 寻找峰值进行标注
kde_lines = ax_b.get_lines()
adaptive_line = next((line for line in kde_lines if 'Adaptive' in line.get_label()), None)

if adaptive_line:
    kde_x, kde_y = adaptive_line.get_data()
    peak_idx = np.argmax(kde_y)
    peak_x = kde_x[peak_idx]
    peak_y = kde_y[peak_idx]

    ax_b.annotate(f'Lower-Bound Constrained\n(Mean $\\approx$ 68, std $\\approx$ 30, Max $\\approx$ 201)', 
                  xy=(peak_x, peak_y), xytext=(peak_x+50, peak_y-0.005),
                  arrowprops=dict(arrowstyle='->', color=colors[1]), color=colors[1], fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", fc=colors[1], ec=colors[1], lw=2, alpha=0.1))

# ==========================================
# 5. 保存和显示
# ==========================================
output_filename = 'figs/fig1/radius_strategy_comparison_real_A_sim_B'
print(f"正在保存图像到: {output_filename}.png/.pdf")
plt.savefig(f'{output_filename}.png', dpi=1200, bbox_inches='tight')
plt.savefig(f'{output_filename}.pdf', bbox_inches='tight')
print("完成。")
plt.show()