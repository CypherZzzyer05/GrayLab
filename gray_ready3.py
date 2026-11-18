import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 建一个 64 × 96 的浮点数组
clean = np.zeros((64, 96), dtype=float)
# 按列均分三块，从左到右赋值 0.0、0.5、1.0
clean[:, 0:32] = 0.0
clean[:, 32:64] = 0.5
clean[:, 64:96] = 1.0

# 固定随机种子 42
rng = np.random.default_rng(42)
# 为每个像素独立生成均值 0、标准差 0.1 的高斯噪声
noise = rng.normal(loc=0.0, scale=0.1, size=clean.shape)
# 先相加
raw = clean + noise
# 再截断
noisy = np.clip(raw, 0.0, 1.0)
# 计算平均误差
residual = noisy - clean

# 第二个去噪算法——中值滤波
filtered = noisy.copy()
for n in range(1, 63):
    for m in range(1, 95):
        filtered[n, m] = np.median(noisy[n-1:n+2, m-1:m+2])

# 验证单个像素，与刚才的 3×3 窗口图对照
print(f"(10, 31) 的真实亮度：{clean[10, 31]:.3f}")
print(f"(10, 31) 中值滤波后：{filtered[10, 31]:.3f}")

# 用同一张干净图作为参考，计算中值滤波后的逐像素误差
median_error = filtered - clean

# 第 31 列是黑区最后一列，第 32 列是灰区第一列
# 去掉首尾两行，因为循环没有处理它们
edge_mse_before = (residual[1:-1, 31:33] ** 2).mean()
edge_mse_median = (median_error[1:-1, 31:33] ** 2).mean()

print(f"黑灰交界处 · 滤波前 MSE：{edge_mse_before:.6f}")
print(f"黑灰交界处 · 中值滤波后 MSE：{edge_mse_median:.6f}")

cols = np.arange(27, 38)
plt.figure(figsize=(8, 4))
plt.step(cols, clean[10, cols], where="mid", color="black",
         linestyle="--", label="Clean")
plt.plot(cols, noisy[10, cols], "o:", label="Noisy")
plt.plot(cols, filtered[10, cols], "s-", label="Median filtered")
plt.xticks(cols)
plt.xlabel("Column (row 10)")
plt.ylabel("Brightness")
plt.legend()
plt.tight_layout()
plt.show()

# 在同一张含噪图上做均值滤波，方便和中值滤波比较
mean_filtered = noisy.copy()
for n in range(1, noisy.shape[0] - 1):
    for m in range(1, noisy.shape[1] - 1):
        mean_filtered[n, m] = noisy[n-1:n+2, m-1:m+2].mean()

# 每一项是一列，依次写上图、上图标题、下图、下图标题
columns = [
    (clean,         "Clean",         noise,                 "Added Gaussian noise"),
    (noisy,         "Noisy",         residual,              "Noisy - Clean"),
    (mean_filtered, "Mean filter",   mean_filtered - clean, "Mean - Clean"),
    (filtered,      "Median filter", filtered - clean,      "Median - Clean"),
]

fig, axes = plt.subplots(2, 4, figsize=(13, 5), constrained_layout=True)

for col, (img, title, error, error_title) in enumerate(columns):
    axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[0, col].set_title(title)

    error_view = axes[1, col].imshow(error, cmap="gray", vmin=-0.3, vmax=0.3,
                                     interpolation="nearest")
    axes[1, col].set_title(error_title)

    axes[0, col].axis("off")
    axes[1, col].axis("off")

fig.colorbar(error_view, ax=axes[1, :], ticks=[-0.3, 0, 0.3],
             shrink=0.8, label="Brightness difference")

# 保存到当前 .py 文件旁边，README 可以直接引用
fig.savefig(Path(__file__).with_name("exp03_filter_comparison.png"), dpi=180)
plt.show()

for name, image in [("未滤波", noisy), ("均值滤波", mean_filtered), ("中值滤波", filtered)]:
    error = image - clean
    gray_mse = (error[1:-1, 33:63] ** 2).mean()
    edge_mse = (error[1:-1, 31:33] ** 2).mean()
    print(f"{name}：灰带内部 {gray_mse:.8f}，黑灰交界 {edge_mse:.8f}")