import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# 🔥 强制重建字体缓存
fm._load_fontmanager(try_read_cache=False)

# ✅ 全局字体设置
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# ==================== 真实实验数据 ====================
deltaT = np.array([18.4, 23.5, 28.3, 33.3, 37.8])
deltaL = np.array([0.0867, 0.1093, 0.1309, 0.1547, 0.1742])

# 线性拟合模型
def linear_model(x, k, b):
    return k * x + b

popt, _ = curve_fit(linear_model, deltaT, deltaL)
k_fit, b_fit = popt

# 平滑拟合曲线
dT_smooth = np.linspace(18.0, 38.0, 500)
dL_smooth = linear_model(dT_smooth, *popt)

# ======================== 绘图 ========================
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor('white')

# 原始散点
ax.scatter(deltaT, deltaL, marker="^", s=80, lw=1.2, edgecolors="black",
           color='dodgerblue', label="原始数据", zorder=5)

# 拟合曲线
ax.plot(dT_smooth, dL_smooth, "r-", linewidth=2,
        label=f"线性拟合: ΔL = {k_fit:.5f}·ΔT + {b_fit:.4f}", zorder=4)

# 图例
legend_elements = [
    Line2D([0], [0], marker='^', color='w', markeredgecolor='black',
           markerfacecolor='dodgerblue', markersize=8, label='原始数据'),
    Line2D([0], [0], color='r', linewidth=2,
           label=f'线性拟合: ΔL = {k_fit:.5f}·ΔT + {b_fit:.4f}'),
]
legend = ax.legend(handles=legend_elements, loc='upper left',
                   facecolor='white', framealpha=0.9)
legend.get_frame().set_edgecolor('#cccccc')

# 🔥 标题物理拆分
title_y = 1.02
ax.text(0.5, title_y, r"$\Delta L$-$\Delta T$", transform=ax.transAxes,
        fontsize=15, ha='right', va='bottom', fontfamily='Times New Roman')
ax.text(0.5, title_y, " 曲线", transform=ax.transAxes,
        fontsize=15, ha='left', va='bottom', fontfamily='SimHei')

# ✅ 更新：坐标轴范围
ax.set_xlim(18.0, 38.0)       # 横轴起点 18.0
ax.set_ylim(0.08, 0.1750)     # 竖轴起点改为 0.08

# ✅ 网格刻度
# 横轴：大格 1，小格 0.1
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
# 竖轴：大格 0.01，小格 0.001
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(0.001))

ax.grid(which='major', linestyle='-', linewidth=0.7, color='red', alpha=0.6)
ax.grid(which='minor', linestyle='-', linewidth=0.3, color='red', alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=11)

# spine与箭头（边界同步更新）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_bounds(18.0, 38.0)
ax.spines['left'].set_bounds(0.08, 0.1750)

arrow_kw = dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=18)
ax.annotate("", xy=(38.0, 0.08), xytext=(36.5, 0.08), arrowprops=arrow_kw, clip_on=False)
ax.annotate("", xy=(18.0, 0.1750), xytext=(18.0, 0.1680), arrowprops=arrow_kw, clip_on=False)

offset_x = plt.matplotlib.transforms.ScaledTranslation(8 / 72, 0, fig.dpi_scale_trans)
offset_y = plt.matplotlib.transforms.ScaledTranslation(0, 8 / 72, fig.dpi_scale_trans)

# 带单位的坐标轴标签
ax.text(38.0, 0.08, r"$\Delta T$ / $^\circ$C", fontsize=13, va='center', ha='left',
        clip_on=False, transform=ax.transData + offset_x)
ax.text(18.0, 0.1750, r"$\Delta L$ / mm", fontsize=13, va='bottom', ha='center',
        clip_on=False, transform=ax.transData + offset_y)

# 日期：2026.9.21
fig.text(0.5, 0.01, "作者：卢泽泓  日期：2026.9.21",
         ha='center', va='bottom', fontsize=12, fontfamily='SimHei')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()