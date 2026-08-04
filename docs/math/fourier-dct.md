# Fourier 变换与 DCT 深入

从傅里叶级数讲起，一路到 DCT 的能量压缩性质，真正理解：为什么对平滑信号（如机器人动作）做 DCT，能量会奇迹般集中到几个低频系数上。是 [Action Tokenization](../robotics/action-tokenization.md) 里把 DCT 当黑盒用的那一步的展开。

!!! note "一条主线"
    **任何信号都能写成一堆"波"的叠加。** 傅里叶家族研究的就是"用哪些波、各占多少"。DCT 是其中专为**实数、平滑信号**优化的一员，它的杀手锏是**能量压缩**——把信息挤进极少数低频系数，这正是 FAST / JPEG 压缩的根基。

---

## 0. 为什么要把信号变到"频率域"

同一段信号有两种看法：

- **时域（time domain）**："每个时刻的值是多少"。机器人动作原始就是这种：第 0 步、第 1 步……的关节角度。
- **频域（frequency domain）**："信号由哪些快慢不同的波组成、各占多少"。同一信息的另一种坐标系。

换到频域的好处：很多在时域看起来"密密麻麻"的信号，在频域**极其稀疏**（只有几个频率有值）。平滑 = 变化慢 = 只有低频 = 频域稀疏 → **可压缩**。

---

## 1. Fourier 级数：周期信号 = 正弦/余弦之和

傅里叶的洞见（1807）：任何**周期**函数都能写成不同频率正弦/余弦的加权和：

$$
f(t)=\frac{a_0}{2}+\sum_{n=1}^{\infty}\Big(a_n\cos(n\omega t)+b_n\sin(n\omega t)\Big)
$$

频率 $n\omega$ 越高的项变化越快。下面用**奇次谐波的正弦叠加逼近方波**——经典例子，也能看到"加越多越像、但边角永远过冲"的**吉布斯现象**。

<div class="ln-demo">
<div class="ln-demo-title">Demo · 傅里叶级数逼近方波</div>
<div class="ln-demo-hint">方波 = 无穷个奇次正弦谐波之和。拖滑块增加谐波个数，看叠加结果（红）如何逼近方波（灰）。</div>
<canvas id="cvSquare" width="900" height="280"></canvas>
<div class="ln-controls">
<label>谐波个数 <input type="range" id="sqN" min="1" max="40" value="3"> <span class="ln-val" id="sqNV">3</span></label>
</div>
<div class="ln-readout" id="sqOut"></div>
</div>

!!! warning "吉布斯现象"
    不连续点（方波的跳变）附近永远有约 9% 的过冲，无论加多少项。启示：**越不连续/越多高频的信号，越难用少数低频项表示**——反过来，**越平滑越好压**。机器人动作恰恰平滑。

---

## 2. 连续 Fourier 变换：从周期到任意信号

把周期推向无穷，级数的"离散频率求和"变成"连续频率积分"，就得到 Fourier 变换：

$$
F(\omega)=\int_{-\infty}^{\infty} f(t)\,e^{-i\omega t}\,dt
$$

这里 $e^{-i\omega t}=\cos\omega t - i\sin\omega t$（欧拉公式）把正弦余弦打包成**复指数**。$F(\omega)$ 是复数，其**模**表示该频率的强度、**幅角**表示相位。

!!! tip "复指数 = 旋转的相量"
    $e^{-i\omega t}$ 在复平面上是一个以角频率 $\omega$ 旋转的单位向量。傅里叶变换在问："信号里有多少成分，和这个以 $\omega$ 旋转的相量同步？" 同步的频率分量就大。

---

## 3. DFT 与 FFT：离散世界的傅里叶

计算机里信号是**有限个采样点** $x_0,\dots,x_{N-1}$，对应**离散傅里叶变换 DFT**：

$$
X_k=\sum_{n=0}^{N-1}x_n\,e^{-i2\pi kn/N},\quad k=0,\dots,N-1
$$

直接算是 $O(N^2)$。**FFT（快速傅里叶变换）**利用对称性把它降到 $O(N\log N)$——这是 20 世纪最重要的算法之一，让实时音视频处理成为可能。

| 名称 | 输入 | 输出 | 复杂度 |
|---|---|---|---|
| Fourier 级数 | 连续周期函数 | 离散系数 $a_n,b_n$ | — |
| Fourier 变换 (FT) | 连续非周期 | 连续谱 $F(\omega)$ | — |
| DFT | $N$ 个离散采样 | $N$ 个复系数 | $O(N^2)$ |
| FFT | 同 DFT（算法优化） | 同 DFT | $O(N\log N)$ |
| **DCT** | $N$ 个实数采样 | $N$ 个**实**系数 | $O(N\log N)$ |

---

## 4. 从 DFT 到 DCT：去掉复数与边界跳变

DFT 有两个对压缩不友好的地方，DCT 正是为修正它们而生：

1. **DFT 系数是复数**（有实部虚部），对实数信号有冗余。
2. **DFT 默认信号周期延拓**，首尾若值不同会在边界产生**人为跳变** → 制造大量高频 → 不利压缩。

!!! note "DCT 的两个 trick"
    ① **偶对称延拓**：DCT 先把信号"镜像"成偶函数再做变换。偶函数的傅里叶展开**只含余弦项** → 系数全是**实数**，没有虚部冗余。

    ② **镜像消跳变**：镜像延拓让边界**平滑续接**（不再有 DFT 那种首尾跳变）→ 高频被压低 → 能量更集中在低频。

    这两点合起来，使 DCT 对实数平滑信号的**能量压缩**显著优于 DFT。

最常用的是 **DCT-II**（就是 JPEG 和 FAST 用的那个）：

$$
X_k=\sum_{n=0}^{N-1}x_n\cos\!\Big[\frac{\pi}{N}\big(n+\tfrac12\big)k\Big]
$$

注意基底是纯余弦、系数 $X_k$ 是实数。$k=0$ 是直流（均值），$k$ 越大频率越高。逆变换 IDCT（即 DCT-III）用同样的余弦基重建，**完全可逆**。

---

## 5. DCT 的四种变体（知道有别即可）

按"在两端怎么做对称延拓"不同，DCT 有 I~IV 四型。实践中：

| 变体 | 用途 |
|---|---|
| **DCT-II** | 最常用。JPEG、MPEG、**FAST action tokenization** 都用它 |
| **DCT-III** | DCT-II 的逆变换（即 IDCT） |
| DCT-I | 端点处理不同，较少用 |
| DCT-IV | 用于 MDCT（音频，如 MP3/AAC 的重叠变换） |

!!! tip "norm=\"ortho\" 是什么"
    scipy 里常加 `norm="ortho"`：给系数乘上归一化因子（$k{=}0$ 乘 $\sqrt{1/N}$，其余乘 $\sqrt{2/N}$），使变换**正交且能量守恒**（Parseval：时域能量 = 频域能量）。这样 IDCT 就是 DCT 的转置，数值更干净。

---

## 6. DCT 基函数画廊

DCT 把信号分解到这一组**固定的余弦基向量**上。第 $k$ 个基 = 频率为 $k$ 的余弦采样。任何信号都是这些基的加权和，权重就是 DCT 系数。下面是前 8 个基（$N=32$）：

<div class="ln-demo">
<div class="ln-demo-title">DCT-II 基函数（k = 0…7）</div>
<div class="ln-demo-hint">k=0 是常数（直流/均值），k 越大振荡越快（频率越高）。鼠标移到任意一个高亮查看。</div>
<canvas id="cvBasis" width="900" height="320"></canvas>
<div class="ln-readout">每个小图是一个余弦基向量 $\cos[\pi/N\,(n+0.5)\,k]$。DCT 系数 $X_k$ = 信号在第 k 个基上的投影。</div>
</div>

---

## 7. 能量压缩：DCT 的杀手锏（亲手验证）

这是整页最重要的概念，也是 FAST 能工作的根本原因。**能量压缩（energy compaction）**：对平滑信号，DCT 把绝大部分能量集中到**极少数低频系数**，其余系数接近 0。

于是我们只需**保留前几个大系数、丢弃一大堆接近 0 的**，就能用很少的数据近乎完美地重建信号。下面亲手验证：

<div class="ln-demo">
<div class="ln-demo-title">Demo · 只保留前 k 个 DCT 系数的重建质量</div>
<div class="ln-demo-hint">上图：原始信号（蓝）vs 只用前 k 个 DCT 系数的重建（红）。下图：DCT 系数能量（保留的高亮）。拖 k，看平滑信号只需极少系数就几乎重合；再切换成"含高频/噪声"的信号对比。</div>
<canvas id="cvCompact" width="900" height="360"></canvas>
<div class="ln-controls">
<label>保留前 k 个系数 <input type="range" id="ckK" min="1" max="32" value="4"> <span class="ln-val" id="ckKV">4</span> / 32</label>
<button id="ckSmooth" class="on">平滑信号</button>
<button id="ckRough">含高频/噪声</button>
</div>
<div class="ln-readout" id="ckOut"></div>
</div>

!!! note "连接到 FAST"
    FAST 不显式"丢弃"系数，而是**对系数取整（量化）**：那些接近 0 的高频系数取整后**变成恰好 0**，效果等同于丢弃，还顺便产生大量重复的 0 供 BPE 压缩。能量压缩越强（信号越平滑），取整后 0 越多 → token 越少。

---

## 8. 为什么偏偏是 DCT（它接近"最优"变换）

理论上，对给定信号统计，**能量压缩最优**的变换是 **KLT（Karhunen–Loève 变换，即 PCA）**——它把信号投影到协方差矩阵的特征向量上。但 KLT 依赖数据统计、要现算特征向量，昂贵且不通用。

!!! tip "DCT ≈ 免费的 KLT"
    对于**一阶马尔可夫信号**（相邻采样高度相关——平滑信号正是如此），可以证明 **DCT 的基函数无限接近 KLT 的最优基**。也就是说：DCT 用一组**固定、无需训练**的余弦基，几乎达到了"为这类信号量身定制的最优变换"的压缩效果。这就是为什么 JPEG、MPEG、FAST 都选 DCT 而不是真的去算 KLT。

呼应 learned vs direct：DCT 是 **direct**（固定基、零训练），却逼近了需要数据统计的最优 learned 变换——**用确定性数学拿到了接近学习的收益**。

---

## 9. 应用：从 JPEG 到 FAST

| 系统 | 怎么用 DCT |
|---|---|
| **JPEG** | 图像切 8×8 块 → 2D DCT → 量化（丢高频）→ 熵编码。你看到的几乎所有照片都被 DCT 压过 |
| **MPEG / H.26x** | 视频帧（残差）做 DCT + 量化 |
| **MP3 / AAC** | 用 MDCT（DCT-IV 的重叠版）压音频 |
| **FAST（机器人）** | 动作 (T,D) 沿**时间轴**做 DCT → 量化取整 → BPE。平滑动作 → 能量集中 → 取整后大量 0 → 短 token 序列 |

!!! tip "同一个套路"
    JPEG 压图片、FAST 压动作，**骨架完全一样**：变到频域(DCT) → 量化丢弃不重要的高频 → 无损熵编码(BPE/Huffman)。理解了 JPEG，你就理解了 FAST 的一半。

---

## 延伸阅读

- [Action Tokenization](../robotics/action-tokenization.md) — 机器人动作离散化主线；FAST 用 DCT 把动作压成 token
- [自回归模型 · BERT / GPT](../ml/autoregressive-models.md) — 序列建模范式与 Prefix-LM，token 之后由谁消费
