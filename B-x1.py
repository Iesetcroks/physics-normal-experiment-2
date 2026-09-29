import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  # ✅ 新增 FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# 🔥 强制重建字体缓存
fm._load_fontmanager(try_read_cache=False)

# ✅ 全局字体设置
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# ==================== 真实实验数据 ====================
x_data = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float)
B_data = np.array([316, 324, 336, 342, 348, 345, 348, 350, 345, 342, 338, 329, 318], dtype=float)


# ==================== 亥姆霍兹线圈理论模型（含原点偏移） ====================
def helmholtz_B(x, A, R, d, x0):
    xc = x - x0
    term1 = 1.0 / (R**2 + (xc - d / 2.0)**2)**1.5
    term2 = 1.0 / (R**2 + (xc + d / 2.0)**2)**1.5
    return A * (term1 + term2)


p0_guess = [350 * 5.5**3, 5.5, 5.5, 5.0]
popt, pcov = curve_fit(helmholtz_B, x_data, B_data, p0=p0_guess, maxfev=10000)
A_fit, R_fit, d_fit, x0_fit = popt

print(f"===== 亥姆霍兹线圈拟合结果 =====")
print(f"A   = {A_fit:.4f}")
print(f"R   = {R_fit:.4f} cm")
print(f"d   = {d_fit:.4f} cm")
print(f"x₀  = {x0_fit:.4f} cm")
print(f"d/R = {d_fit / R_fit:.4f}")

x_smooth = np.linspace(-1.0, 11.0, 500)
B_smooth = helmholtz_B(x_smooth, *popt)

# ======================== 绘图 ========================
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor('white')

# 原始散点
ax.scatter(x_data, B_data, marker="^", s=80, lw=1.2, edgecolors="black",
           color='dodgerblue', zorder=5)

# 理论拟合曲线（不加入图例）
ax.plot(x_smooth, B_smooth, "r-", linewidth=2, zorder=4)

# ✅ 图例仅保留原始数据
legend_elements = [
    Line2D([0], [0], marker='^', color='w', markeredgecolor='black',
           markerfacecolor='dodgerblue', markersize=8, label='原始数据'),
]
legend = ax.legend(handles=legend_elements, loc='upper left',
                   facecolor='white', framealpha=0.9)
legend.get_frame().set_edgecolor('#cccccc')

# 🔥 标题物理拆分
title_y = 1.02
ax.text(0.5, title_y, r"$B$-$x$", transform=ax.transAxes,
        fontsize=15, ha='right', va='bottom', fontfamily='Times New Roman')
ax.text(0.5, title_y, "曲线(间距R)", transform=ax.transAxes,
        fontsize=15, ha='left', va='bottom', fontfamily='SimHei')

# ✅ 坐标轴范围
xlim_min, xlim_max = -1.0, 11.0
ylim_min, ylim_max = 300, 360
ax.set_xlim(xlim_min, xlim_max)
ax.set_ylim(ylim_min, ylim_max)

# ✅ 网格刻度
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

# ✅ 新增：强制横纵轴主刻度显示一位小数
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.grid(which='major', linestyle='-', linewidth=0.7, color='red', alpha=0.6)
ax.grid(which='minor', linestyle='-', linewidth=0.3, color='red', alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=11)

# spine设置
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_bounds(xlim_min, xlim_max)
ax.spines['left'].set_bounds(ylim_min, ylim_max)

# ✅ 箭头：终点=轴端点，起点回退足够距离，箭尖自然覆盖轴尾
# 仅使用arrowstyle、color、lw、mutation_scale四个安全参数，杜绝报错
arrow_kw = dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=18)
# 横轴箭头
ax.annotate("", xy=(xlim_max, ylim_min), xytext=(xlim_max - 2.0, ylim_min),
            arrowprops=arrow_kw, clip_on=False)
# 竖轴箭头
ax.annotate("", xy=(xlim_min, ylim_max), xytext=(xlim_min, ylim_max - 10),
            arrowprops=arrow_kw, clip_on=False)

# 坐标轴标签偏移
offset_x = plt.matplotlib.transforms.ScaledTranslation(8 / 72, 0, fig.dpi_scale_trans)
offset_y = plt.matplotlib.transforms.ScaledTranslation(0, 8 / 72, fig.dpi_scale_trans)

ax.text(xlim_max, ylim_min, r"$x$ / cm", fontsize=13, va='center', ha='left',
        clip_on=False, transform=ax.transData + offset_x)
ax.text(xlim_min, ylim_max, r"$B$ / μT", fontsize=13, va='bottom', ha='center',
        clip_on=False, transform=ax.transData + offset_y)

# 日期
fig.text(0.5, 0.01, "作者：卢泽泓  日期：2026.9.28",
         ha='center', va='bottom', fontsize=12, fontfamily='SimHei')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()