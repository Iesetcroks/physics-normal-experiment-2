import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# 🔥 强制重建字体缓存（解决SimHei等中文字体索引失效问题）
fm._load_fontmanager(try_read_cache=False)

# ✅ 全局字体设置
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 原始完整数据
t_all = np.linspace(0, 300, 11)
T_all = np.array([100.0, 95.5, 91.0, 87.6, 86.7, 81.0, 77.5, 75.0, 72.0, 69.8, 67.5])

# 剔除 t=120s 的数据用于拟合
mask_fit = t_all != 120
t_fit = t_all[mask_fit]
T_fit_data = T_all[mask_fit]

# 模型及导数
def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

def exp_decay_deriv(t, a, b, c):
    return -a * b * np.exp(-b * t)

# 仅用剔除120s后的数据拟合
popt, _ = curve_fit(exp_decay, t_fit, T_fit_data, p0=[40, 0.005, 65])
a_fit, b_fit, c_fit = popt

# 平滑曲线（延伸至320）
t_smooth = np.linspace(0, 320, 500)
T_smooth = exp_decay(t_smooth, *popt)

# 绘图
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor('white')

# 散点
ax.scatter(t_all, T_all, marker="^", s=60, lw=1, edgecolors="black",
           color='dodgerblue', label="原始数据", zorder=5)

# 拟合曲线
ax.plot(t_smooth, T_smooth, "r-", linewidth=2,
        label=f"拟合: {a_fit:.2f}·exp(-{b_fit:.5f}·t)+{c_fit:.2f}", zorder=4)

# 切线
cmap = plt.cm.turbo
colors = cmap(np.linspace(0.1, 0.9, len(t_all)))
t_tan_full = np.array([0, 320])
for i, ti in enumerate(t_all):
    yi_model = exp_decay(ti, *popt)
    slope = exp_decay_deriv(ti, *popt)
    y_tan = slope * (t_tan_full - ti) + yi_model
    ax.plot(t_tan_full, y_tan, '--', color=colors[i], linewidth=1.2, alpha=0.8, zorder=3)

# 图例
legend_elements = [
    Line2D([0], [0], marker='^', color='w', markeredgecolor='black',
           markerfacecolor='dodgerblue', markersize=8, label='原始数据'),
    Line2D([0], [0], color='r', linewidth=2,
           label=f'拟合: {a_fit:.2f}·exp(-{b_fit:.5f}·t)+{c_fit:.2f}'),
    Line2D([0], [0], linestyle='--', color='gray', linewidth=1.2,
           label='切线（各色对应各数据点）')
]
legend = ax.legend(handles=legend_elements, loc='upper right',
                   facecolor='white', framealpha=0.9)
legend.get_frame().set_edgecolor('#cccccc')

# 🔥🔥🔥 终极方案：标题物理拆分，彻底隔离中英文渲染引擎
# 注意：不再使用 ax.set_title()，改用两个独立的 ax.text()
title_y = 1.02  # 标题在axes上方的相对位置

# 第一部分：纯LaTeX斜体（不含任何中文）
ax.text(0.5, title_y, r"$T$-$t$", transform=ax.transAxes,
        fontsize=15, ha='right', va='bottom', fontfamily='Times New Roman')

# 第二部分：纯中文（不含任何$符号和Unicode特殊字符）
ax.text(0.5, title_y, " 曲线", transform=ax.transAxes,
        fontsize=15, ha='left', va='bottom', fontfamily='SimHei')

ax.set_xlim(0, 320)
ax.set_ylim(65, 105)

# 红色网格
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.grid(which='major', linestyle='-', linewidth=0.7, color='red', alpha=0.6)
ax.grid(which='minor', linestyle='-', linewidth=0.3, color='red', alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=10)

# spine与箭头
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_bounds(0, 313)
ax.spines['left'].set_bounds(65, 98)

arrow_kw = dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=18)
ax.annotate("", xy=(320, 65), xytext=(313, 65), arrowprops=arrow_kw, clip_on=False)
ax.annotate("", xy=(0, 105), xytext=(0, 98), arrowprops=arrow_kw, clip_on=False)

offset_x = plt.matplotlib.transforms.ScaledTranslation(8 / 72, 0, fig.dpi_scale_trans)
offset_y = plt.matplotlib.transforms.ScaledTranslation(0, 8 / 72, fig.dpi_scale_trans)
ax.text(320, 65, r"$t$ / s", fontsize=13, va='center', ha='left', clip_on=False,
        transform=ax.transData + offset_x)
ax.text(0, 105, r"$T$ / $^\circ$C", fontsize=13, va='bottom', ha='center', clip_on=False,
        transform=ax.transData + offset_y)

# 底部作者信息（同样显式指定中文字体）
fig.text(0.5, 0.01, "作者：卢泽泓  日期：2026.9.14",
         ha='center', va='bottom', fontsize=12, fontfamily='SimHei')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()