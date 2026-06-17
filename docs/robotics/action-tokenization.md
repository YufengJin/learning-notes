# 机器人 Action Tokenization

把"连续动作 → 离散 token → 连续动作"这条链路，拆成每一块需要的前置知识。主线是 **π₀-FAST 风格的离散自回归路线**，并对照 FSQ、Binning 以及连续生成的 flow matching。

全程出现三种标签：

- <span class="ln-tag ln-direct">DIRECT</span> — 纯数学变换（确定性、无需训练）
- <span class="ln-tag ln-learned">LEARNED</span> — 从数据拟合（统计 fit 或神经网络）
- <span class="ln-tag ln-lossy">LOSSY</span> — 有信息损失

---

## 0. 大局观：到底在解决什么问题

一个机器人策略（policy）要做的事：**看到观测 $o$（图像 + 本体状态），输出一段未来动作 $a_{1:T}$**。动作是**连续**的实数向量（关节角度、末端位姿、夹爪开合）。

建模 $p(a_{1:T}\mid o)$ 有两条完全不同的路线：

| 路线 | 动作表示 | 代表 | 类比 |
|---|---|---|---|
| **连续生成** | 直接吐实数向量 | π₀（flow matching / 扩散） | 像图像扩散模型 |
| **离散自回归** | 先把动作变成 token，像 LLM 逐个预测 | π₀-FAST | 像 GPT 写句子 |

本笔记聚焦**第二条路线**。核心魔法：把连续动作"翻译"成一串整数 token，于是机器人控制问题就**变成了语言建模问题**——可以直接复用 LLM 的全套机器（Transformer、交叉熵、采样）。

!!! note "一句话直觉"
    Tokenization = 给连续动作发一本"字典"，把光滑曲线翻译成有限个"单词"。难点全在于：字典怎么建（learned 还是 direct？）、翻译会丢多少信息（lossy 在哪一步？）、翻译回去准不准（可逆性）。

**概念依赖图：**

```text
robot action（连续向量序列）
   │  为什么不直接回归？ → mode averaging 问题（§2.1）
   ▼
离散化的动机 ───────────────┐
   │                        │
   ├─ FAST 路线：           ├─ 信号处理：DCT（§4）压缩能量
   │   DCT→量化→BPE          ├─ 量化/取整（§5）唯一的有损步
   │                        └─ BPE（§6）无损压缩字典
   │
   ├─ FSQ 路线：神经自编码器 + VQ/FSQ 码本（§8、§9）
   │
   └─ Binning 路线：均匀分桶（§10，最朴素基线）
                ▼
        都接到 自回归 Transformer（§11）
        用 交叉熵 训练 → 天然 多模态（§12）
        靠 采样 + temperature（§12.1）取出不同解
        靠 注意力掩码（§13）区分 prompt / action
```

---

## 1. 什么是 robot action（被 tokenize 的对象）

一个动作样本是一个二维矩阵：

```text
actions.shape == (T, D)
            ┌─ T = action_horizon：一次预测未来多少步（如 50 步 ≈ 1 秒@50Hz）
            └─ D = action_dim：每步的自由度数（如 7 = 6 关节 + 1 夹爪，或 14 = 双臂）
```

两个关键性质：

- **① 时间上平滑**：相邻时刻动作变化不大（机器人不会瞬移）→ 这是 DCT 能压缩的前提。
- **② 各维量纲不同**：关节角范围可能 ±3 rad，夹爪 0~1 → 必须先**归一化**到统一范围。

!!! tip "为什么是'一段'而不是'一步'"
    预测一整段 **action chunk**（而非单步）能减少累积误差、让动作更连贯，也给 DCT 提供了"时间轴"去做频域变换。单步动作没有频域可言。

### 归一化：第一步预处理 <span class="ln-tag ln-learned">LEARNED</span>（统计分位数，近似无损）

把每一维动作缩放到约 $[-1,1]$。常用**分位数（quantile）归一化**而不是简单的 min-max：取第 1% 和 99% 分位作为边界，对离群值更鲁棒。

```python
# 概念：用训练集统计出的分位数把 action 压到 [-1,1]
a_norm = 2 * (a - q01) / (q99 - q01) - 1   # 超出的 clip 掉
```

!!! note "为什么算 LEARNED 但不是神经网络"
    "learned" 在这里 = **从数据统计拟合出来的参数**（分位数），而不是梯度下降训练的权重。这个区分会贯穿全文：FAST 的 BPE 词表、归一化分位数都是这种"统计 fit"，**不是 neural network**。

---

## 2. 为什么要把连续动作离散化

最自然的想法是让网络直接输出实数动作，用 L2/MSE 回归监督。那为什么要费劲离散化成 token？三个理由：

1. **复用 LLM 的整套基础设施**：一旦动作是 token，就能直接用预训练 VLM（如 PaliGemma）的 Transformer、词表、交叉熵，省掉重新设计输出头。
2. **天然支持多模态**（§12 详讲）：离散分布能同时给"向左"和"向右"高概率，回归不行。
3. **压缩 + 高效自回归**：FAST 用 DCT+BPE 把一段动作压成很短的 token 序列，自回归推理更快。

### 2.1 核心动机：mode averaging（模式平均）灾难

这是**理解整条离散化路线最重要的一个直觉**。设想机器人面前有个障碍物，绕过去有两种同样正确的方式：**向左绕** 或 **向右绕**。训练数据里两种都有。

!!! danger "L2 回归的致命伤"
    MSE 回归本质上是在拟合一个**单峰高斯**，它的最优解是所有正确答案的**平均值**。"向左"和"向右"的平均 = **直直撞上障碍物**。这就是 mode averaging。

<div class="ln-demo">
<div class="ln-demo-title">Demo · mode averaging vs 多峰分布</div>
<div class="ln-demo-hint">数据里有两个正确动作（蓝点簇=向左，绿点簇=向右）。拖动滑块改变两簇距离，看 L2 回归的"最优预测"（红线）落在哪里。</div>
<canvas id="cvMode" width="900" height="280"></canvas>
<div class="ln-controls">
<label>两个模式的间距 <input type="range" id="modeGap" min="0" max="100" value="70"></label>
<label><input type="checkbox" id="showCat" checked> 显示离散分布（多峰）</label>
</div>
<div class="ln-readout" id="modeOut"></div>
</div>

间距越大，L2 的"最优预测"（两簇均值）越落在**无人区**——一个谁都没演示过、可能直接撞墙的动作。而离散分布能**同时**在左右两个 bin 上点亮高概率，完美保留两个模式。这就是为什么要把连续动作切成离散 token + 用分类交叉熵。

---

## 3. tokenization 概念（从 NLP 借来的）

在 NLP 里，tokenization = 把一段连续的文字符流切成有限词表里的整数 ID：

```text
"机器人很可爱"  →  ["机器","人","很","可爱"]  →  [8123, 442, 19, 5601]
   文本                 子词（subword）            整数 token id
```

这些整数随后被查 **embedding 表**变成向量喂进 Transformer。关键点：

- **词表（vocabulary）**是有限的（如 PaliGemma 有 257,152 个 token）。
- 每个 token id 对应词表里一行 embedding 向量。
- 模型输出是**词表上的概率分布**（softmax over 257,152 类）。

!!! note "迁移到机器人的核心 trick"
    PaliGemma 词表最后 **128 个** id 平时几乎用不到。可以**征用这 128 个槽位**当 action token——于是动作 token 和文字 token **共享同一张 embedding 表、同一个 softmax 输出头**。机器人动作真的被当成"一门新语言的单词"塞进了语言模型。

---

## 4. DCT 离散余弦变换 <span class="ln-tag ln-direct">DIRECT</span>

这是 FAST 的第一步，也是很多人最陌生的一块。

### 直觉：把一条曲线拆成"不同频率的余弦波之和"

任何一段离散信号（比如某个关节角度随时间的 50 个采样点），都可以**精确地**表示成一堆不同频率余弦波的加权和。DCT 就是算出"每个频率占多少权重"的那组系数。

!!! tip "类比：调音台"
    想象一个音频均衡器：低频（重低音）、中频、高频各有一个推子。DCT 就是把信号分解到这些"频率推子"上。机器人动作**平滑** → 几乎全是低频 → 只有最左边几个推子有值，右边一大片≈0。这就是"能量集中在低频"。

### 公式（DCT-II，知道长这样即可）

把长度 $N$ 的信号 $x_0,\dots,x_{N-1}$ 变成系数 $X_0,\dots,X_{N-1}$：

$$
X_k = \sum_{n=0}^{N-1} x_n \cos\!\left[\frac{\pi}{N}\left(n+\tfrac12\right)k\right]
$$

$k=0$ 是直流分量（整体平均），$k$ 越大频率越高。逆变换 IDCT 用同样的余弦基把系数加回去，**完全可逆**（除浮点误差）。沿**时间轴 T** 对每个动作维度 D 独立做 DCT。

<div class="ln-demo">
<div class="ln-demo-title">Demo · DCT 能量集中</div>
<div class="ln-demo-hint">上图是一段"动作信号"（可调平滑度），下图是它的 DCT 系数。看平滑信号如何把能量挤到最左边几个低频系数上。</div>
<canvas id="cvDct" width="900" height="360"></canvas>
<div class="ln-controls">
<label>信号平滑度 <input type="range" id="dctSmooth" min="1" max="20" value="3"></label>
<button id="dctReroll">🎲 换一条信号</button>
</div>
<div class="ln-readout" id="dctOut"></div>
</div>

!!! note "为什么 DCT 是离散化的'神助攻'"
    DCT 本身**不离散化、不丢信息**（纯可逆数学）。但它把信号能量**重新分布**：平滑动作 → 大部分系数≈0。下一步取整时，这些接近 0 的小系数会被压成**恰好 0**，产生大量重复的 0 → 给 BPE 提供了绝佳的压缩素材。**DCT 不压缩，但它让后面的压缩变得高效。**

---

## 5. 量化 / 取整 <span class="ln-tag ln-direct">DIRECT</span> <span class="ln-tag ln-lossy">LOSSY</span>

这是整条 FAST 链路里**唯一真正丢信息的一步**，也是"连续 → 离散"真正发生的地方。

连续值有无穷多种取值，token 词表只能表示有限个。量化 = **把连续值映射到最近的"格点"上**。最简单的形式：先乘一个缩放系数，再四舍五入取整。

```python
q = round(x * scale)        # 连续 x → 整数 q   （编码，有损）
x_hat = q / scale           # 整数 q → 近似连续 x̂（解码，回不到原值）
```

误差 $|x-\hat x|$ 最大约 $\frac{1}{2\,\text{scale}}$。scale 越大 → 格点越密 → 越精确，但整数范围越大、token 越多。这是**精度 vs 压缩率**的权衡。

<div class="ln-demo">
<div class="ln-demo-title">Demo · 量化的精度 / 压缩权衡</div>
<div class="ln-demo-hint">蓝线是原始连续信号，红色阶梯是量化后重建的信号。拖动"量化级数"看：级数越少越省（token 少）但越失真。</div>
<canvas id="cvQuant" width="900" height="300"></canvas>
<div class="ln-controls">
<label>量化级数（bins） <input type="range" id="qBins" min="2" max="64" value="8"> <span class="ln-val" id="qBinsV">8</span></label>
</div>
<div class="ln-readout" id="quantOut"></div>
</div>

!!! warning "FAST 的精妙之处"
    FAST 不是直接量化原始动作，而是量化 **DCT 系数**。因为高频系数≈0，取整后变成一大片 0——既丢得起（高频对平滑动作几乎没贡献），又制造了可压缩的重复模式。"在频域取整"比"在时域取整"聪明得多。

---

## 6. BPE 字节对编码 <span class="ln-tag ln-learned">LEARNED</span>（统计 fit，非 NN，无损）

取整后得到一长串整数（含大量重复，尤其是 0）。BPE 是一种**无损压缩**算法，把"高频重复的模式"合并成单个新 token，让序列变短。

反复执行："找出当前序列里**出现最频繁的相邻 pair**，把它合并成一个新符号"，直到达到目标词表大小。

```text
初始: a a b a a b a a b          （a a 这对出现很多）
合并 (a,a)→Z:  Z b Z b Z b        （现在 Z b 这对出现很多）
合并 (Z,b)→Y:  Y Y Y               ← 9 个符号压成 3 个，完全可还原
```

<div class="ln-demo">
<div class="ln-demo-title">Demo · BPE 合并压缩</div>
<div class="ln-demo-hint">下面是一段量化后的整数序列（很多重复）。点"合并一次"执行一步 BPE，看序列如何变短、字典如何增长。</div>
<canvas id="cvBpe" width="900" height="170"></canvas>
<div class="ln-controls">
<button id="bpeStep">▶ 合并最频繁的 pair</button>
<button id="bpeReset">↺ 重置</button>
</div>
<div class="ln-readout" id="bpeOut"></div>
</div>

"哪些 pair 该合并、合并顺序如何" = **BPE 词表/合并规则**，它是**从训练数据统计出来的**（所以是 learned），但**只是查表式的统计规则，不含任何神经网络/梯度**。给定词表，编码和解码都**完全无损**。

!!! note "FAST 名字的由来"
    FAST = **F**requency-space **A**ction **S**equence **T**okenization。DCT+BPE 是它的算法本体——纯信号处理 + 统计压缩，没有神经网络。

---

## 7. 把 FAST 拼起来（全景）

编码方向（连续 → token）：

```text
归一化的 action 矩阵 A (T, D)，约 [-1,1]
   │
   ① DCT（沿时间轴）          DIRECT 无损，能量挤向低频
   ▼
频域系数矩阵 C (T, D)，高频≈0
   │
   ② Scale + Round            DIRECT · LOSSY ← 离散化在此发生
   ▼
稀疏整数矩阵（大量 0）
   │
   ③ Flatten 拍平成 1D        DIRECT
   │
   ④ BPE 合并重复模式         LEARNED 词表 · 无损
   ▼
最终离散 token 序列（长度可变，通常远短于 T×D）
   │
   ⑤ 映射到 PaliGemma 词表尾部 128 槽   DIRECT 纯算术
```

解码方向就是**完全反过来**：BPE 解码 → reshape → 除以 scale → IDCT → 反归一化。

!!! danger "最容易记混的结论"
    **FAST 不是神经网络！** 它是"确定性可逆压缩（DCT+取整+Flatten）+ 一个统计拟合的 BPE 词表"。唯一的信息损失来自步骤②的取整。这正是它相对 VQ/FSQ 的卖点——**不用训练神经自编码器**，开箱即用。

<div class="ln-demo">
<div class="ln-demo-title">Demo · FAST 完整往返流水线（连续 ↔ 离散，可调 scale）</div>
<div class="ln-demo-hint">一段平滑的多维动作轨迹，沿时间轴做 DCT → 量化取整 → IDCT 还原。拖动量化 scale 看四个面板同时变化——这是"连续怎么变离散、损失从哪来、token 多少"的最直观演示。</div>
<canvas id="cvRT" width="900" height="250"></canvas>
<canvas id="cvTrade" width="900" height="200" style="margin-top:14px"></canvas>
<div class="ln-controls">
<label>量化 scale（越大越精细） <input type="range" id="rtScale" min="1" max="64" value="10" step="0.5"> <span class="ln-val" id="rtScaleV">10</span></label>
<button id="rtReroll">🎲 换一条轨迹</button>
</div>
<div class="ln-readout" id="rtOut"></div>
</div>

### 7.1 一个完整的数值示例：看清每一步的 shape 与数据

很多人对"DCT 输入输出到底是什么 shape、scale+round 怎么落地"没有具象感。下面用**真实计算的数字**走一遍（为看清楚取 `D=1` 单维、`T=8` 个时间步；多维时每一列独立重复同样的过程）。

**输入：归一化后的动作矩阵 A**，形状 `(T, D) = (8, 1)`，值已归一化到 $[-1,1]$：

```text
A = [ 1.000, 0.816, 0.494, 0.111, -0.258, -0.555, -0.755, -0.853 ]   # shape (8,1)
     t=0    t=1    t=2    t=3    t=4     t=5     t=6     t=7         ← 时间轴
这是一条平滑下降的曲线（机器人动作的典型样子）。
```

**① DCT —— 沿时间轴变换，输入输出 shape 不变**（`(T,)→(T,)`，对 `(T,D)` 就是每列各做一次）：

```text
C = DCT(A) = [ 0.00, 1.889, 0.159, -0.00, -0.00, 0.00, 0.00, -0.00 ]   # 仍是 (8,1)
              k=0   k=1    k=2    k=3    k=4   k=5   k=6   k=7         ← 频率轴
              直流  低频────────→                       高频
能量几乎全集中在 k=1（=1.889）！k≥3 的高频系数已经≈0。
```

!!! tip "关键认知"
    DCT **不改变元素个数**：进去 8 个数，出来 8 个数。它只是**换了一组坐标**——从"时刻"换到"频率"。信息一个没丢（可逆），但被**重新分布**到少数几个低频系数上。

**② Scale + Round —— 离散化在这里发生**（取 `scale = 8`）：

```text
C × scale = [ 0.00, 15.115, 1.273, -0.00, -0.00, 0.00, 0.00, -0.00 ]
round(·)  = [   0,    15,     1,     0,     0,    0,    0,    0    ]   = Cq（整数！）
                                                                       8 个里只有 2 个非零
```

这一步：① 把连续小数变成**整数**（真正的"离散化"）；② 那些 ≈0 的高频系数被**压成恰好 0**。损失就来自这里（`15.115→15`、`1.273→1` 的舍入）。

**③ Flatten** 把 `(T,D)` 拍平成 1D 整数序列 `[0, 15, 1, 0, 0, 0, 0, 0]`（`D>1` 时按 `T×D` 拉直）。

**④ BPE** 把高频重复模式（如"连续 5 个 0"）合并成单个 token，序列更短，最后 **⑤** 映射到词表最高 128 个槽位的 id。

**解码：原路返回，看损失有多小**

```text
token → BPE解码 → [0,15,1,0,0,0,0,0] → reshape(8,1) → ÷scale → IDCT → 反归一化
重建 A_rec = [0.977, 0.803, 0.497, 0.125, -0.241, -0.545, -0.756, -0.862]
原始 A     = [1.000, 0.816, 0.494, 0.111, -0.258, -0.555, -0.755, -0.853]
重建 MSE ≈ 1.7e-4   ← 仅用 2 个非零系数就几乎完美还原 8 个值！
```

!!! note "这个例子说明了一切"
    8 个连续值 → DCT → 只有 **2 个**有效（非零）整数系数 → 重建误差仅 1.7e-4。**这就是 FAST 高效压缩的全部秘密**：平滑动作经 DCT 后能量极度集中，取整制造大量 0，BPE 再把 0 的游程压掉。想深入 DCT 的能量压缩数学，见 [Fourier 与 DCT · 能量压缩](../math/fourier-dct.md#7-dct)。

---

## 8. VQ-VAE 与码本（理解 FSQ 的前置）<span class="ln-tag ln-learned">LEARNED</span>（真神经网络）

FSQ 路线是**真正的神经网络方案**。要懂 FSQ，先懂它的前身 VQ-VAE 的"码本"思想。

VQ-VAE（Vector Quantized VAE）的想法：维护一本**可学习的"码本"**——比如 256 个向量，每个叫一个 codeword（码字），编号 0~255。

```text
编码器(NN) → 连续向量 z
         ↓ 在码本里找最近的码字
最近码字的编号 = token（如 #42）          ← 离散化
         ↓
解码器(NN) 用 #42 对应的码字向量重建动作
```

整个编码器、解码器、**码本里的 256 个向量本身**都是**梯度训练**出来的（用重建 MSE 损失）。这就是真正的"端到端学习的离散表示"。

!!! warning "VQ-VAE 的痛点"
    ① **码本坍缩**：训练中很多码字从没被用到，浪费容量、难训练。

    ② 取最近邻不可导 → 需要 **straight-through estimator**（直通梯度）的 trick 才能反向传播。

    这些痛点正是 **FSQ** 要解决的。

### straight-through estimator（直通梯度）

"取最近邻 / 四舍五入"是阶梯函数，导数处处为 0，梯度传不回编码器。技巧：**前向用离散值，反向假装它是恒等函数**（梯度直接穿过去）。

```python
z_q = z + stop_gradient(quantize(z) - z)
#   前向：z_q == quantize(z)（离散）
#   反向：d z_q / d z == 1   （梯度像没量化一样直通回去）
```

---

## 9. FSQ 有限标量量化 <span class="ln-tag ln-learned">LEARNED</span>

FSQ（Finite Scalar Quantization）是 VQ-VAE 的优雅替代：**砍掉可学习码本，改成"在极少数几维上各自做固定网格的四舍五入"**。它无需学习码本，却天然不会坍缩。

核心 trick：

1. 编码器把动作投影到**极低维**（比如只有 3 维）。
2. 每一维用 $\tanh$ 压到 $[-1,1]$，然后**四舍五入到该维固定的几个格点**（如第 1 维 8 个格点、第 2 维 6 个、第 3 维 5 个）。
3. 多维格点的组合用**混合进制**编码成**单个整数** token（$8\times6\times5=240\approx256$ 个码字）。

```python
# FSQ encode（概念）
x = proj_down(z)                      # 投影到 ~3 维      [LEARNED]
zc = tanh(x)                          # 压到 [-1,1]
digits = round((zc+1)*(bases-1)/2)    # 每维四舍五入到格点 [DIRECT, LOSSY]
token = undigitize(digits)            # 多维 digit → 单整数（混合进制）[DIRECT]
```

!!! tip "FSQ vs VQ 一句话"
    VQ：码本是**学出来的一堆任意向量**，要小心坍缩。

    FSQ：码本是**固定的规则网格**（每维等距格点的笛卡尔积），不用学、不会坍缩。只有投影编码器/解码器是神经网络。

<div class="ln-demo">
<div class="ln-demo-title">Demo · FSQ 的 2D 固定网格码本</div>
<div class="ln-demo-hint">把动作压到 2 维后，FSQ 的"码字"就是这些规则网格点。移动鼠标（或拖动），看连续点被吸附到最近的网格码字（=一个整数 token）。调每维格点数看码本大小变化。</div>
<canvas id="cvFsq" width="420" height="420" style="margin:auto"></canvas>
<div class="ln-controls">
<label>第1维格点数 <input type="range" id="fsqB1" min="2" max="9" value="6"> <span class="ln-val" id="fsqB1V">6</span></label>
<label>第2维格点数 <input type="range" id="fsqB2" min="2" max="9" value="5"> <span class="ln-val" id="fsqB2V">5</span></label>
</div>
<div class="ln-readout" id="fsqOut"></div>
</div>

---

## 10. Binning 均匀分桶（最朴素基线）<span class="ln-tag ln-direct">DIRECT</span>

RT-2 / OpenVLA 风格，**零学习**：每个维度、每个时间步**独立**地把 $[-1,1]$ 均匀切成 256 个桶，取整成桶编号。

```python
token = round((a+1)/2 * n_bins)     # 编码
a_hat = token / n_bins * 2 - 1      # 解码
```

- **优点**：简单、无需任何训练、完全可逆（除取整）。
- **缺点**：token 数 = T×D（不压缩，序列很长）；精度只有 256 级且**不利用时间相关性**。是论文里的对照基线。

三者放一起看：

| | Binning | FAST | FSQ |
|---|---|---|---|
| 连续→离散 | 均匀分桶取整 | DCT→取整→BPE | Transformer→FSQ 量化 |
| 是否 NN | ❌ | ❌（统计 fit） | ✅ |
| 需预训练 | ❌ | 仅 BPE 词表/分位数 | ✅ 需 checkpoint |
| token 数 | 多（T×D） | 少（压缩） | 中（固定） |
| 利用时间相关性 | ❌ | ✅（DCT） | ✅（注意力） |

---

## 11. 自回归建模 + 交叉熵

无论用哪个 tokenizer，动作变成 token 后，**训练目标和 GPT 一模一样**：预测下一个 token。

序列长什么样：

```text
Task: pick up the cup, State: <离散化的本体状态>;
Action: <a₁><a₂>...<aₙ>|
└──────── 前缀 prefix（条件，不算损失）───────┘└─ 后缀 postfix（算损失）─┘
```

注意：连本体 **state 也被离散化**（切 256 桶）一起塞进 prompt。

训练目标 = next-token 交叉熵（负对数似然）：

$$
\mathcal{L} = -\frac{1}{|\text{mask}|}\sum_{t}\; \text{mask}_t \cdot \log p_\theta(a_t \mid a_{<t}, o)
$$

```python
targets = one_hot(tokens[:, 1:], vocab=257152)   # 右移一位当 target
logp    = log_softmax(logits)
token_logp = sum(targets * logp, axis=-1)        # 正确 token 的 log 概率
loss = -sum(token_logp * loss_mask) / sum(loss_mask)  # 只在 action 段算损失
```

!!! note "三个要点"
    ① **没有任何 MSE/回归项**——纯交叉熵。

    ② `loss_mask` 只在后缀 `Action:...|` 为 True → 只对 action token 计损失，prompt/state 只当条件。

    ③ 推理时自回归逐 token 采样，遇到 `|` 或 EOS 停止，再用 tokenizer 反变换回连续动作。

---

## 12. 多模态是怎么"免费"得到的

回到 §2.1 的 mode averaging 问题。离散 token + 交叉熵**从三个层面**解决它：

1. **离散 categorical 天然多峰**：每个 token 位置输出整个词表的 softmax，可以同时给"向左对应的 token"和"向右对应的 token"高概率。不再被迫平均。
2. **自回归分解表达联合多模态**：$p(a_{1:n}\mid o)=\prod_t p(a_t\mid a_{<t},o)$。一串 categorical 的连乘能表示复杂的多峰联合分布——"向左"是一条连贯 token 序列，"向右"是另一条。
3. **推理时靠采样取出不同的解**（下一节）。

### 12.1 采样与 temperature

有了多峰分布，怎么取出动作？对每个 token 的 logits 做带温度的采样：

$$
p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

- $\tau=0$：取 argmax → 确定性、单一最可能解。
- $\tau>0$：从分布随机采样 → 不同 RNG 得到不同可行解；$\tau$ 越大越多样。

<div class="ln-demo">
<div class="ln-demo-title">Demo · temperature 如何调节多模态采样</div>
<div class="ln-demo-hint">某个 token 位置的双峰 logits（"向左"和"向右"两个高概率区）。拖动 temperature 看 softmax 分布怎么变；点"采样 20 次"看实际抽到的 token 分布。</div>
<canvas id="cvTemp" width="900" height="280"></canvas>
<div class="ln-controls">
<label>temperature τ <input type="range" id="tempT" min="0" max="200" value="100"> <span class="ln-val" id="tempTV">1.00</span></label>
<button id="tempSample">🎲 采样 20 次</button>
</div>
<div class="ln-readout" id="tempOut"></div>
</div>

!!! tip "直觉"
    τ→0：分布变成一根尖针（只取最高峰，丢掉另一个解）。τ 大：分布变平，两个峰都有机会被抽到 → 这次绕左、下次绕右。**多模态的"解"是靠随机采样从保留下来的多峰分布里取出来的。**

---

## 13. 注意力掩码：prefix 双向 vs action 因果

序列里 prompt/state（前缀）和 action（后缀）扮演不同角色，用不同的注意力掩码（`ar_mask`）：

- **前缀 = 双向注意力（像 encoder）**：prompt + state 是已知条件，互相都能看到（BERT 式），让模型充分理解任务上下文。
- **后缀 = 因果注意力（像 decoder）**：action token 只能看到自己**左边**的（GPT 式），保证自回归生成时"不偷看未来"。

这种"前缀双向 + 后缀因果"的混合掩码是 **prefix-LM** 结构，PaliGemma / π₀-FAST 都用它。

---

## 14. 对比：另一条路线 flow matching（π₀）

为了把离散路线放进坐标系，简单看看连续路线。π₀ **不 tokenize**，用 **flow matching**（扩散家族）直接生成连续动作：从高斯噪声出发，沿学到的"速度场"积分，逐步把噪声"流"成动作。

| | π₀-FAST（本笔记主线） | π₀（flow matching） |
|---|---|---|
| 动作表示 | 离散 token | 连续向量 |
| 建模 | 自回归 + 交叉熵 | 速度场 + 积分 |
| 多模态来源 | 多峰 categorical + 采样 | 从噪声出发的随机积分路径 |
| 复用 LLM | ✅ 直接复用词表/Transformer | 需专门的 flow 头 |
| 推理 | 逐 token（快慢取决于 token 数） | 少步积分 |

!!! note "同一个问题，两种答案"
    两条路线都在解"一个观测对应多个合理动作"的多模态难题。FAST 用"离散化 + 分类"，π₀ 用"连续随机生成"。理解了离散路线为什么需要 tokenization（mode averaging + 复用 LLM），就理解了这个领域的核心张力。

---

## 15. 三种 tokenizer 重建效果对比

"连续 → token → 连续"必然有损（来自量化取整）。三种 tokenizer 的**重建误差**和**token 数**各有取舍。先看一张实测对比，再亲手调参验证。

!!! warning "一个事实澄清"
    实践中通常**只有 FAST 真正把 action 编码成 token**；Binning / FSQ 更多作为**推理基线**（只负责把已有 token 解码回动作）。下面的"编码-解码往返对比"是**按各自算法的数学定义**做的概念实测，用来理解三者的精度/压缩取舍。

<div class="ln-demo">
<div class="ln-demo-title">Demo · FAST vs Binning vs FSQ 重建往返对比</div>
<div class="ln-demo-hint">同一段动作（可调平滑度/噪声），三种方法各自编码再解码。上图叠加三条重建曲线，下图用柱状对比「重建 MSE」和「token 数」。FSQ 为<b>示意</b>（无训练 checkpoint，用「固定 token 数 + 网格量化」近似其行为）。</div>
<canvas id="cvCmp" width="900" height="250"></canvas>
<canvas id="cvCmpBar" width="900" height="190" style="margin-top:14px"></canvas>
<div class="ln-controls">
<label>信号平滑度 <input type="range" id="cmpSmooth" min="1" max="12" value="2"> <span class="ln-val" id="cmpSmoothV">2</span></label>
<label><input type="checkbox" id="cmpNoise"> 加噪声</label>
<label>FAST scale <input type="range" id="cmpScale" min="2" max="40" value="12"> <span class="ln-val" id="cmpScaleV">12</span></label>
<label>FSQ token 数 <input type="range" id="cmpFsqN" min="4" max="24" value="12"> <span class="ln-val" id="cmpFsqNV">12</span></label>
</div>
<div class="ln-readout" id="cmpOut"></div>
</div>

| | FAST | FSQ | Binning |
|---|---|---|---|
| 重建误差来源 | DCT 系数取整 | FSQ 网格量化 + 神经重建误差 | 每个标量独立 256 桶取整 |
| 误差随平滑度改善 | ✅ 越平滑越准（DCT 利用时间相关性） | ✅ 编码器能学到时间结构 | ❌ 与平滑度无关，固定量化地板 |
| token 数 | 少且**可变**（压缩，≪ T×D） | 中且**固定**（=num_tokens） | 多且固定（=T×D，每标量一个） |
| 同等 token 预算下精度 | 高（平滑信号） | 高（需训练好） | 低（浪费在高频） |
| 是否需训练 | 仅 BPE 词表/分位数（统计 fit） | ✅ 需训练神经 checkpoint | ❌ 零训练 |

!!! tip "怎么读这张表"
    核心权衡是**「token 数 ↔ 精度 ↔ 是否利用时间结构」**。Binning 最朴素：每标量一个 token、误差恒定但 token 最多、不懂时间相关性。FAST 用 DCT 把时间相关性变成低频稀疏，**用更少 token 拿到更高精度**（前提：动作平滑）。FSQ 用神经网络学一个紧凑潜空间，token 数固定可控，但要先训练。

---

## 16. π₀-FAST 模型全貌：基础模型 / 输入 / 输出

tokenizer 把动作变成 token 之后，是谁在消费这些 token？这一节把 π₀-FAST 的模型骨架讲清楚。

### 16.1 基础模型：PaliGemma（Gemma-2B 解码器 + SigLIP 视觉）

π₀-FAST 的骨干是 **PaliGemma**——一个视觉语言模型（VLM），由两部分组成：

| 组件 | 是什么 | 规格 |
|---|---|---|
| 视觉编码器 | **SigLIP So400m/14**（ViT） | patch=14，224×224 → 16×16 = **256 个图像 token / 张** |
| 语言模型 (LLM) | **Gemma-2B，decoder-only** | width=2048, depth=18, heads=8, vocab=**257,152** |

!!! note "是 Decoder-only 吗？——是，但用 prefix-LM 掩码"
    Gemma 本身是 **decoder-only** 的语言模型。但 π₀-FAST 通过**注意力掩码**把它用成 **prefix-LM**：图像 + prompt + state 这段前缀用**双向**注意，action 这段后缀用**因果**注意。所以"decoder-only 架构 + prefix-LM 掩码"两句话都对。掩码细节与可交互对比见 [自回归模型 · 注意力掩码](../ml/autoregressive-models.md#2)。

### 16.2 输入与多模态 embed 顺序

模型输入含：多路相机图像、各自的 mask、本体 state、以及已 tokenize 的文本 prompt。它们被拼成**一条 token 序列**，顺序是：

<div class="ln-fig">
<svg viewBox="0 0 860 230" role="img" aria-label="π0-FAST 输入序列结构">
  <text x="10" y="20" fill="#6B675F" font-size="13">输入 token 序列（沿序列轴 concat）：左→右</text>
  <rect x="10" y="36" width="150" height="56" rx="8" fill="rgba(63,185,80,.13)" stroke="#3fb950"/>
  <text x="85" y="60" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="700">① 图像 token</text>
  <text x="85" y="78" text-anchor="middle" fill="#6B675F" font-size="10">base_0 / base_1 / wrist_0</text>
  <rect x="166" y="36" width="330" height="56" rx="8" fill="rgba(63,185,80,.13)" stroke="#3fb950"/>
  <text x="331" y="58" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="700">② 文本前缀</text>
  <text x="331" y="76" text-anchor="middle" fill="#6B675F" font-size="10">"Task: …, State: ⟨离散化state⟩;\nAction: "</text>
  <rect x="502" y="36" width="250" height="56" rx="8" fill="rgba(88,166,255,.13)" stroke="#58a6ff"/>
  <text x="627" y="58" text-anchor="middle" fill="#58a6ff" font-size="12" font-weight="700">③ action token + "|"</text>
  <text x="627" y="76" text-anchor="middle" fill="#6B675F" font-size="10">FAST 离散 token（要预测的）</text>
  <line x1="10" y1="104" x2="496" y2="104" stroke="#3fb950" stroke-width="2"/>
  <text x="253" y="122" text-anchor="middle" fill="#3fb950" font-size="12">前缀 prefix · 双向注意 · 不计损失</text>
  <line x1="502" y1="104" x2="752" y2="104" stroke="#58a6ff" stroke-width="2"/>
  <text x="627" y="122" text-anchor="middle" fill="#58a6ff" font-size="12">后缀 suffix · 因果 · 计损失</text>
  <text x="10" y="158" fill="#6B675F" font-size="12">embed 来源：</text>
  <text x="85" y="178" text-anchor="middle" fill="#bc8cff" font-size="11">SigLIP 视觉编码器</text>
  <text x="430" y="178" text-anchor="middle" fill="#bc8cff" font-size="11">Gemma 词表 embedding（文本 + action 共享同一张表）</text>
  <text x="10" y="208" fill="#6B675F" font-size="11">注：state 不是单独向量，而是被离散成 256 桶后写进文本 prompt 里（字符串）。</text>
</svg>
<div class="ln-fig-cap">先图像、后文本前缀，再 action；前缀双向、后缀因果。</div>
</div>

- **先图像、后文本**：先把每张图过 SigLIP 得到一串图像 token，**再**把 tokenized 文本 prompt 的 embedding 接在后面。
- **state 在文本里**：本体 state 被离散成 256 桶、转成字符串拼进 `"State: …;"`，所以它走的是**文本 embedding**，不是单独通道。
- **图像与文本/action 共享同一个 Transformer**，但图像 token 之间是双向注意。

### 16.3 输出：每步预测一个 token，自回归直到结束

模型输出是**词表（257,152 类）上的概率分布**，即"下一个 token 是谁"。生成是**逐 token 自回归**的：

```text
while 未遇到 EOS 且 步数 < 上限(256):
    token = 从 last_logit 采样          # 每步只产出【1 个】token（argmax 或带温度采样）
    把 token embed 回去，预测下一个     # 接回输入，继续
遇到 EOS 或到达上限则停止
```

!!! note "每次预测一个还是多个？"
    **每次预测一个 action token**（每步取一个）。action token 的**总数是可变的**（FAST 压缩后通常远少于 T×D），靠遇到 `"|"` / EOS 终止。训练时则不同：是**一次前向 teacher forcing**——整条序列并行预测每个位置的下一个 token，只在 action 段算交叉熵。

### 16.4 解码完是不是 (T, D)？——是

生成的一串变长 action token 被还原：取 `"Action: "` 与 `"|"` 之间的 token → 映射回 FAST token 空间 → 用 `time_horizon=T, action_dim=D` 解码，reshape 成 **`(T, D)`**。

!!! tip "闭环"
    无论中间 token 数是多少，解码端都用 `time_horizon` / `action_dim` 强制 reshape 回 `(T, D)`，正好对上 §1 里 tokenizer 的**输入**形状——整条"连续 (T,D) → token → 连续 (T,D)"的环就闭合了。

---

## 术语表（速查）

| 术语 | 含义 |
|---|---|
| action chunk / horizon (T) | 一次预测的未来动作步数 |
| action_dim (D) | 每步动作的自由度数 |
| tokenization | 连续/文本 → 有限词表的整数 ID |
| DCT / IDCT | 离散余弦（逆）变换，时域↔频域，可逆 direct |
| quantization | 连续值映射到有限格点（取整），唯一有损步 |
| BPE | 字节对编码，合并高频 pair 的无损压缩 |
| codebook / codeword | 码本 / 码字，离散表示的"字典"及其条目 |
| VQ-VAE | 带可学习码本的离散自编码器 |
| FSQ | 有限标量量化，用固定网格代替可学习码本 |
| straight-through estimator | 前向离散、反向恒等的梯度直通技巧 |
| mode averaging | 回归把多个正确解平均成错误解的现象 |
| categorical 分布 | 离散类别上的概率分布，可多峰 |
| cross-entropy | 分类任务的负对数似然损失 |
| temperature τ | 采样温度，调节分布尖锐度/多样性 |
| ar_mask | 区分双向(prefix)/因果(action)注意力的掩码 |
| prefix-LM | 前缀双向+后缀因果的混合 Transformer |
| flow matching | 连续动作生成路线（π₀），扩散家族 |

---

## 延伸阅读

- [FAST Tokenization 论文快照](/paper-snapshots/fast-tokenization/) — paper-snapshots 上对 FAST 原论文的精读
- [Fourier 变换与 DCT](../math/fourier-dct.md) — 把 §4 的 DCT 讲透：能量压缩、为何接近最优(KLT)、JPEG 到 FAST 的同源套路
- [自回归模型 · BERT / GPT](../ml/autoregressive-models.md) — §11/§13 的展开：三大范式、BERT 双向 vs GPT 因果、为何 VLA 选 Prefix-LM
