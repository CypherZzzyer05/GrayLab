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

# 第一个去噪算法——均值滤波
filtered = noisy.copy()
for n in range(1, 63):
    for m in range(1, 95):
        filtered[n, m] = noisy[n-1:n+2, m-1:m+2].mean()

# 灰带内部对比
gray_error = residual[1:-1, 33:63]
gray_mse_before = (gray_error ** 2).mean()
gray_error_after = filtered[1:-1, 33:63] - clean[1:-1, 33:63]
gray_mse_after = (gray_error_after ** 2).mean()
print(gray_mse_before)
print(gray_mse_after)

# 灰黑交接带对比
edge_error = residual[1:-1, 31:33]
edge_mse_before = (edge_error ** 2).mean()
edge_error_after = filtered[1:-1, 31:33] - clean[1:-1, 31:33]
edge_mse_after = (edge_error_after ** 2).mean()
print(edge_mse_before)
print(edge_mse_after)

fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

# 第一排看亮度：黑色是 0，白色是 1，三张图共用同一亮度范围
for ax, img, name in zip(axes[0], [clean, noisy, filtered],
                        ["clean", "noisy", "filtered"]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(name)

# 第二排看变化：中灰表示 0，偏白表示正值，偏黑表示负值
errors = [noise, residual, filtered - clean]
names = ["noise", "residual = noisy - clean", "filtered - clean"]
limit = max(np.abs(err).max() for err in errors)

for ax, err, name in zip(axes[1], errors, names):
    error_map = ax.imshow(err, cmap="gray", vmin=-limit, vmax=limit,
                          interpolation="nearest")
    ax.set_title(name)

fig.colorbar(error_map, ax=axes[1].tolist(), label="Signed error", shrink=0.8)
fig.savefig(Path(__file__).with_name("mean_filter_comparison.png"), dpi=150)
plt.show()
