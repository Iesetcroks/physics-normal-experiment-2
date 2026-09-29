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

# ✅ 数据
x_data = np.array([1.833, 1.803, 1.771, 1.745, 1.738, 1.690, 1.658, 1.633, 1.602, 1.577, 1.550])
y_data = np.array([-0.812, -0.854, -0.879, -0.910, -0.947, -0.979, -1.009, -1.018, -1.071, -1.102, -1.149])

# 线性拟合 (用于绘制趋势线)
popt, _ = curve_fit(lambda x, a, b: a * x + b, x_data, y_data)
slope, intercept = popt
x_smooth = np.linspace(1.540, 1.860, 200)
y_smooth = slope * x_smooth + intercept

# 绘图
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor('white')

# 散点
ax.scatter(x_data, y_data, marker="^", s=60, lw=1, edgecolors="black",
           color='dodgerblue', label="原始数据", zorder=5)

# 拟合直线
ax.plot(x_smooth, y_smooth, "r-", linewidth=2,
        label=f"线性拟合: y={slope:.3f}x+{intercept:.3f}", zorder=4)

# 图例
legend_elements = [
    Line2D([0], [0], marker='^', color='w', markeredgecolor='black',
           markerfacecolor='dodgerblue', markersize=8, label='原始数据'),
    Line2D([0], [0], color='r', linewidth=2,
           label=f'线性拟合: y={slope:.3f}x+{intercept:.3f}')
]
legend = ax.legend(handles=legend_elements, loc='upper right',
                   facecolor='white', framealpha=0.9)
legend.get_frame().set_edgecolor('#cccccc')

# 🔥🔥🔥 终极方案：标题物理拆分，彻底隔离中英文渲染引擎
title_y = 1.02

# 第一部分：纯LaTeX数学符号（已改为分式 \frac）
ax.text(0.5, title_y, r"$\lg\left|\frac{\Delta T}{\Delta t}\right|$-$\lg(T-T_0)$", transform=ax.transAxes,
        fontsize=15, ha='right', va='bottom', fontfamily='Times New Roman')

# 第二部分：纯中文（不含任何$符号和Unicode特殊字符）
ax.text(0.5, title_y, " 曲线", transform=ax.transAxes,
        fontsize=15, ha='left', va='bottom', fontfamily='SimHei')

# ✅ 坐标轴范围（竖轴已修正为 -1.200 ~ -0.800）
ax.set_xlim(1.550, 1.850)
ax.set_ylim(-1.200, -0.800)

# 红色网格
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.grid(which='major', linestyle='-', linewidth=0.7, color='red', alpha=0.6)
ax.grid(which='minor', linestyle='-', linewidth=0.3, color='red', alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=10)

# spine与箭头
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

arrow_kw = dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=18)
ax.annotate("", xy=(1.850, -1.200), xytext=(1.830, -1.200), arrowprops=arrow_kw, clip_on=False)
ax.annotate("", xy=(1.550, -0.800), xytext=(1.550, -0.820), arrowprops=arrow_kw, clip_on=False)

offset_x = plt.matplotlib.transforms.ScaledTranslation(8 / 72, 0, fig.dpi_scale_trans)
offset_y = plt.matplotlib.transforms.ScaledTranslation(0, 8 / 72, fig.dpi_scale_trans)

# 横轴标签
ax.text(1.850, -1.200, r"$\lg(T-T_0)$", fontsize=13, va='center', ha='left', clip_on=False,
        transform=ax.transData + offset_x)

# 🔥 竖轴标签（同步改为分式）
ax.text(1.550, -0.800, r"$\lg\left|\frac{\Delta T}{\Delta t}\right|$", fontsize=13, va='bottom', ha='center', clip_on=False,
        transform=ax.transData + offset_y)

# 底部作者信息（显式指定中文字体）
fig.text(0.5, 0.01, "作者：卢泽泓  日期：2026.9.14",
         ha='center', va='bottom', fontsize=12, fontfamily='SimHei')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()