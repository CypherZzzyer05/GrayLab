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

print(clean[0, [0, 31, 32, 63, 64, 95]])

print(noise.shape)
print("原像素、噪声、相加后、截断后：", clean[0, 1], noise[0, 1], raw[0, 1], noisy[0, 1])
print("含噪图范围：", noisy.min(), noisy.max())

print("黑区平均误差", residual[:, 0:32].mean())
print("灰区平均误差", residual[:, 32:64].mean())
print("白区平均误差", residual[:, 64:96].mean())

fig1, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)

# 上排看像素亮度：黑 = 0，白 = 1
axes[0, 0].imshow(clean, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[0, 0].set_title("clean | Original image")

axes[0, 1].imshow(noisy, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[0, 1].set_title("noisy | After clipping")

# 下排看带符号的变化：两张图共用同一范围，中灰 = 0
limit = np.abs(noise).max()
error_scale = dict(cmap="gray", vmin=-limit, vmax=limit, interpolation="nearest")

axes[1, 0].imshow(noise, **error_scale)
axes[1, 0].set_title("noise | Added noise")

error_map = axes[1, 1].imshow(residual, **error_scale)
axes[1, 1].set_title("residual | noisy - clean")

fig1.colorbar(error_map, ax=axes[1, :].tolist(), label="Signed change")
fig1.savefig(Path(__file__).with_name("exp01_clipping.png"), dpi=150)
plt.show()