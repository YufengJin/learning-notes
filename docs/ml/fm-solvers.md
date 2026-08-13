# 流匹配 ODE 求解器 · 图解

<div class="ln-byline">2026-08-06 · 阅读约 17 分钟 · Yufeng Jin</div>

<p class="ln-lead" markdown>采样一张图 = 数值积分一条 ODE。这篇笔记把五个经典求解器（<span class="fms-t-euler">Euler</span>、<span class="fms-t-midpoint">Midpoint</span>、<span class="fms-t-heun">Heun</span>、<span class="fms-t-rk4">RK4</span>、<span class="fms-t-dopri5">Dopri5</span>）拆开讲清楚——每个动画都是**真实积分**，每张数据图都来自可复现实验。</p>

<figure class="fms-panel">
  <canvas id="heroCv" height="380" aria-label="流匹配向量场动画：粒子从高斯噪声流向八个模式"></canvas>
  <figcaption><b>题图 · </b>512 个粒子沿 8-Gaussians 流匹配速度场的精确闭式解实时积分：从中心的高斯噪声流向八个红 × 标记的模式。这不是示意动画，是浏览器里逐帧算出来的真实轨迹。</figcaption>
</figure>

---

<div class="ln-eyebrow">引子</div>

## 1 · 引言

你训练好了一个 flow matching 模型。采样的时候，代码里总有那么一行 `x = solver.step(x, t)`——但那一行里到底发生了什么？为什么 Stable Diffusion 3 默认走 28 步而不是 1000 步？为什么有人用 Euler、有人用 Heun、有人用听起来很高级的 DPM-Solver？换一个求解器，到底能省多少算力？

这篇笔记用一个**完全可复现的 2D 玩具实验**把这些问题讲透。我们训练一个真实的流匹配模型（两份 2D 数据、15000 步、固定种子），然后让五种经典 ODE 求解器——<span class="fms-t-euler">Euler</span>、<span class="fms-t-midpoint">Midpoint</span>、<span class="fms-t-heun">Heun</span>、<span class="fms-t-rk4">RK4</span>、<span class="fms-t-dopri5">Dopri5</span>——在同一个场上同台竞技。

阅读方式随意，但有两个约定：页面里的每个交互图都是**浏览器内的真实数值积分**（不是预渲染的动图），每张数据图的数字都来自配套代码 `fm_solvers.py` 的真实运行。读完你应该能回答：*给定一个 NFE 预算，我该选哪个求解器，为什么。*

<div class="ln-howto"><b>读法建议</b>：已经熟悉 ODE 与数值积分的读者可直接跳到
第 4 节；只想要结论的，看第 8.3 节的五条与第 9 节的决策表即可。
第 4–6 节是方法主线，每节都按「直觉 → 机制/数学 → 实测 → 教训」展开。</div>

<div class="ln-eyebrow">引子 · 为什么值得学</div>

## 2 · 动机：为什么生成模型工程师要懂求解器

三个理由，一个比一个实际：

- **NFE 就是钱。** 生成一张图的成本 ≈ 网络前向次数 × 单次前向的开销。模型定了之后，单次前向的开销就定了——求解器决定的是*同样的质量要付多少次前向*。
- **差距是 3 倍量级，而且是免费的。** 本文实验里，同一个训练好的模型要达到接近采样地板的质量：Euler 要 NFE=40，RK4 只要 12（第 8 节）。不改模型、不重训练，只换几行求解器代码。
- **结论并不显然。** 极低预算下高阶方法反而更差；场没训练好时 Euler 会反超 RK4；「自动挡」的自适应方法也不是全场最优。凭直觉选，很容易选错——所以本文所有结论都配数据。

<div class="ln-eyebrow">预备知识</div>

## 3 · 预备知识

### 3.1 · 什么是 ODE 与初值问题

一条常微分方程（ODE）长这样：

$$
\frac{dx}{dt} = f(x, t)
$$

它说的是一件很朴素的事：「一个点在位置 $x$、时刻 $t$ 时，它的**速度**是 $f(x,t)$」。$f$ 叫做**向量场**——空间里每一点都插着一支小箭头（页首题图里那些淡色短线就是）。再给定出发点 $x(0)=x_0$，就构成一个**初值问题**（IVP）：从 $x_0$ 出发、每时每刻顺着箭头走，走出来的轨迹 $x(t)$ 就是解。

只有极少数 $f$ 能写出解析解（比如第 5 节用来做测试的 $dx/dt=x^2$）。当 $f$ 是一个神经网络时，解析解无从谈起——**只能数值求解**，这正是本文的主题。

### 3.2 · 生成模型是怎么变成一条 ODE 的

把「生成」看成「求解 ODE」不是流匹配的发明，而是一条汇流的河：neural ODE / 连续正规化流直接把生成建模为积分一条学出来的场<sup>[[1]](#refs)</sup>；score-based 扩散模型<sup>[[3]](#refs)</sup>的随机采样过程被证明存在一条确定性的「probability flow ODE」，两者边际分布完全相同<sup>[[2]](#refs)</sup>；而 flow matching<sup>[[4]](#refs)</sup> / rectified flow<sup>[[5]](#refs)</sup> 干脆跳过 SDE，直接用回归目标训练一个速度场。本文用的就是最后这种（linear/OT 路径）：

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad \text{回归目标}\ \ v = x_1 - x_0
$$

训练时在噪声 $x_0$ 与数据 $x_1$ 的连线上随机取点，让网络 $v_\theta(x,t)$ 回归「平均往哪儿流」。采样时反过来，从 $x_0 \sim \mathcal{N}(0, I)$ 出发解初值问题：

$$
\frac{dx}{dt} = v_\theta(x, t), \qquad t: 0 \to 1
$$

$t=1$ 时粒子落在哪里，样本就是什么。于是：

**生成质量 = 场学得多准（模型误差） × 积分得多准（离散化误差）。** 前者是训练的事，后者是求解器的事——本文只讲后者：同一个场，不同求解器的差距。

### 3.3 · 数值积分入门：离散化与「阶」

数值求解的思路只有一句话：把 $t\in[0,1]$ 切成 $N$ 小步，每一步用**有限次**场评估近似真轨迹。两个关键概念：

- **局部截断误差**：单独一步犯的错。把方法的更新公式和真解的泰勒展开逐项对比，如果能匹配到 $dt^p$ 项，这一步的误差就是 $O(dt^{p+1})$。
- **全局误差**：走完全程累积的错。一共走 $N=1/dt$ 步，误差大致累加：$O(dt^{p+1})\cdot(1/dt) = O(dt^p)$。这样的方法称为 **$p$ 阶方法**。

「阶」是理解一切的钥匙：$p$ 阶方法步长减半、误差降 $2^p$ 倍，在 log-log 图上是一条斜率 $-p$ 的直线——第 5 节我们会当场实测。多说一句边界：本文的流匹配场都不「刚性」（stiff），显式方法完全够用；刚性问题需要隐式方法，那是另一本书的事<sup>[[7]](#refs)</sup>。

### 3.4 · 记账单位：NFE

比较求解器时，横轴永远用 **NFE**（number of function evaluations，网络前向次数）而非步数：一步 RK4 要 4 次前向，一步 Euler 只要 1 次。「10 步 RK4 vs 10 步 Euler」不公平，「NFE=40 的 RK4 vs NFE=40 的 Euler」才公平——*NFE 就是钱，本文所有图的横轴都是它。*

<div class="ln-eyebrow">方法 01 · 固定步长</div>

## 4 · 固定步长求解器：一步怎么走

预备知识就绪，进入正题。所有「单步法」都在回答同一个问题：*已知当前点的速度，往前跨一步 dt，落在哪？* 差别只在于「偷看」几次场，以及怎么组合这些偷看——偷看得越聪明，阶数越高。

与其看公式，不如亲手拆一步。图 1 的场是一个匀速旋转场（真解是圆弧，弯曲程度一目了然）：切换求解器、拖动 dt，看每种方法如何用 1–4 次「偷看」拼出这一步，以及单步误差怎样长大。

<figure class="fms-panel">
  <div class="fms-controls">
    <span class="fms-seg" id="anatomySeg" role="group" aria-label="选择求解器"></span>
    <label>步长 dt <input type="range" id="anatomyDt" min="0.15" max="1.35" step="0.05" value="0.9">
      <span class="fms-readout" id="anatomyDtVal"></span></label>
  </div>
  <canvas id="anatomyCv" height="360" aria-label="单步构造图：各级速度评估与真实圆弧对比"></canvas>
  <div class="fms-stats">
    <span><span class="k">本步 NFE</span> <b id="anatomyNfe"></b></span>
    <span><span class="k">单步误差</span> <b id="anatomyErr"></b></span>
    <span class="k" id="anatomyDesc"></span>
  </div>
  <figcaption><b>图 1 · 单步构造解剖（交互）。</b>匀速旋转场上的一步：灰色粗线是真解圆弧，彩色箭头是该求解器实际迈出的方向，灰色虚线箭头与小点是中间的「偷看」（各级评估 k₁…k₄），红色虚线段是单步误差。切换求解器可对比：同样的 dt，看的次数越多、组合越聪明，落点越贴近真解。</figcaption>
</figure>

### NFE 记账

| 求解器 | 阶 | 每步评估 | 一次评估的用途 | steps=10 时 NFE |
|---|---|---|---|---|
| <span class="fms-chip fms-c-euler"></span>Euler | 1 | 1 | 起点速度 k₁ 直接走满一步 | 10 |
| <span class="fms-chip fms-c-midpoint"></span>Midpoint | 2 | 2 | 先走半步探路，用**中点速度** k₂ 走全程 | 20 |
| <span class="fms-chip fms-c-heun"></span>Heun | 2 | 2 | 先走满一步探路，用**首尾平均** (k₁+k₂)/2 修正 | 20 |
| <span class="fms-chip fms-c-rk4"></span>RK4 | 4 | 4 | 两次中点 + 一次终点，加权 (k₁+2k₂+2k₃+k₄)/6 | 40 |

<div class="ln-eyebrow">方法 02 · 收敛阶</div>

## 5 · 收敛阶：为什么高阶「省钱」

3.3 节说过：$p$ 阶方法步长减半、误差降 $2^p$ 倍，log-log 图上是斜率 $-p$ 的直线。这不只是理论——配套代码 `test_solvers.py` 的收敛阶单元测试断言的就是这个比例（Euler 2×、Heun/Midpoint 4×、RK4 16×），而图 2 让你亲眼看到它。

图 2 在你的浏览器里现场求解 $dx/dt = x^2$, $x(0)=\tfrac12$（真解 $x(1)=1$），逐个 NFE 画出误差。注意横轴已经换算成 NFE：RK4 每步贵 4 倍，但斜率 −4 很快就把这笔成本挣回来了。

!!! note "一个有趣的坑"
    为什么不用更顺手的 $dx/dt = x$？因为在**线性** ODE 上 Midpoint 和 Heun 的一步更新代数恒等（都精确等于乘 $1+h+h^2/2$），两条线会逐位重合、看起来像图画错了——要看出「同阶、误差常数不同」，必须用非线性方程。这个 bug 我们真踩过。

<figure class="fms-panel">
  <div class="fms-legend" id="convLegend"></div>
  <canvas id="convCv" height="380" aria-label="收敛阶图：四种求解器在解析 ODE 上的误差随 NFE 的 log-log 曲线"></canvas>
  <figcaption><b>图 2 · 收敛阶实测（浏览器现场计算，交互：悬停读数）。</b>四种求解器在 dx/dt=x² 上的终点误差 vs NFE（log-log）。灰色虚线为斜率 −1/−2/−4 参考线：Euler 贴 −1，Midpoint 与 Heun 贴 −2（两条平行线、误差常数不同），RK4 贴 −4。同样 NFE=40 处，四条曲线相差可达 6 个数量级。</figcaption>
</figure>

<div class="ln-lesson"><b>教训</b>：阶数不是纸面参数——同样 NFE=40，四条曲线的终点误差
相差可达 6 个数量级。同阶的 Midpoint 与 Heun 是两条平行线：阶决定斜率，误差常数决定
截距。验证收敛阶时必须用<b>非线性</b>方程，线性 ODE 上 Midpoint 与 Heun 的一步更新
代数恒等，两条线会逐位重合，看起来像图画错了。</div>

<div class="ln-eyebrow">方法 03 · 自适应步长</div>

## 6 · 自适应步长：Dopri5 如何自己配速

固定步长有个尴尬：场平缓的地方步子迈小了浪费，弯急的地方步子迈大了翻车。Dormand–Prince 5(4)<sup>[[6]](#refs)</sup>——SciPy `RK45`、MATLAB `ode45`、torchdiffeq 默认求解器的本体——解法是**每一步同时算两个答案**：7 级评估拼出一个 5 阶解 $y_5$ 和一个 4 阶解 $y_4$，两者之差就是免费的误差估计：

$$
\mathrm{err} = \mathrm{rms}\!\left(\frac{y_5 - y_4}{\mathrm{atol} + \mathrm{rtol}\cdot\max(|x|, |y_5|)}\right)
$$

- $\mathrm{err} \le 1$：接受这一步，且末级导数下步复用（FSAL，7 级只花 6 次新评估）；
- $\mathrm{err} > 1$：**拒绝**，缩小 dt 重来（这一步的评估白花）；
- 无论接受与否，按 `dt ×= clip(0.9·err^(−1/5), 0.2, 5)` 更新步长（经典的步长控制器，安全系数 0.9，clip 防震荡<sup>[[7]](#refs)</sup>）。

图 3 在 8-Gaussians 的真实流场上跑一遍（512 个粒子共享步长，和 `fm_solvers.py` 的实现一致）。这个场前段近似直线、临近 $t=1$ 时粒子要挤进 σ=0.08 的窄峰——看 dt 如何自动「前松后紧」。

<figure class="fms-panel">
  <div class="fms-controls">
    <label>rtol
      <span class="fms-seg" id="dpSeg" role="group" aria-label="选择容差"></span>
    </label>
  </div>
  <canvas id="dpCv" height="300" aria-label="Dopri5 自适应步长条形图：每步 dt、接受与拒绝"></canvas>
  <div class="fms-stats">
    <span><span class="k">接受步</span> <b id="dpAcc"></b></span>
    <span><span class="k">拒绝步</span> <b id="dpRej"></b></span>
    <span><span class="k">总 NFE</span> <b id="dpNfe"></b></span>
    <span><span class="k">最小/最大 dt</span> <b id="dpRange"></b></span>
  </div>
  <figcaption><b>图 3 · Dopri5 的自适应配速（交互：切换 rtol）。</b>每根柱子是一次尝试：横轴是它覆盖的 t 区间，高度是 log(dt)；实心橙=接受，红色描边=拒绝后重试。前段（场近似直线）大步流星，临近 t=1（粒子挤进窄峰）自动碎步、出现拒绝重试。收紧 rtol，步子整体变密、NFE 上涨。</figcaption>
</figure>

### Butcher 表（Dormand–Prince 5(4)）

这些系数就是 `fm_solvers.py` 里 `_DP_A / _DP_B5 / _DP_B4` 的来源。第 $i$ 行表示第 $i$ 级在 $t + c_i\cdot dt$ 处、用前面各级的 $a_{ij}$ 线性组合位置去评估场：

<table class="fms-butcher">
<tr><td class="cl">0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td class="cl">1/5</td><td>1/5</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td class="cl">3/10</td><td>3/40</td><td>9/40</td><td></td><td></td><td></td><td></td></tr>
<tr><td class="cl">4/5</td><td>44/45</td><td>−56/15</td><td>32/9</td><td></td><td></td><td></td></tr>
<tr><td class="cl">8/9</td><td>19372/6561</td><td>−25360/2187</td><td>64448/6561</td><td>−212/729</td><td></td><td></td></tr>
<tr><td class="cl">1</td><td>9017/3168</td><td>−355/33</td><td>46732/5247</td><td>49/176</td><td>−5103/18656</td><td></td></tr>
<tr><td class="cl">1</td><td>35/384</td><td>0</td><td>500/1113</td><td>125/192</td><td>−2187/6784</td><td>11/84</td></tr>
<tr><td class="cl tp"><i>b</i><sub>5</sub></td><td class="tp">35/384</td><td class="tp">0</td><td class="tp">500/1113</td><td class="tp">125/192</td><td class="tp">−2187/6784</td><td class="tp">11/84</td></tr>
<tr><td class="cl"><i>b</i><sub>4</sub></td><td>5179/57600</td><td>0</td><td>7571/16695</td><td>393/640</td><td>−92097/339200</td><td>187/2100</td></tr>
</table>

*注意 $b_5$ 恰好等于最后一级的 $a$ 行——所以「本步的 5 阶解」正是「下一步的第 1 级评估点」，这就是 FSAL（First Same As Last）省一次评估的来历。b₄ 行末尾还有一个 1/40（第 7 级权重）。*

<div class="ln-lesson"><b>教训</b>：自适应不是「更高阶」，而是把「选步长」这件事从人肉扫
网格变成设一个 <code>rtol</code>。代价是每步同时算 5 阶与 4 阶两个解来拿免费的误差估计；
FSAL 让 7 级评估只花 6 次新的网络前向，被拒绝的步则是白花的评估。</div>

<div class="ln-eyebrow">动手 · 同场对比</div>

## 7 · 亲手采样：同一个场，五种走法

现在把第 4–6 节合起来玩真的：1500 个粒子从 $\mathcal{N}(0,I)$ 出发，在 8-Gaussians 的精确流场上积分到 $t=1$。换求解器、拖步数，看散点云如何从「一圈糊」凝聚成八个尖峰。推荐一条对比路线：*Euler·2 步 → Euler·8 步 → Midpoint·4 步（NFE=8）→ RK4·2 步（NFE=8）*——同样的 NFE，形状差很多，这就是「怎么花钱」的差别。

<figure class="fms-panel">
  <div class="fms-controls">
    <span class="fms-seg" id="pgSeg" role="group" aria-label="选择求解器"></span>
    <label id="pgStepsLab">步数 <input type="range" id="pgSteps" min="0" max="8" step="1" value="3">
      <span class="fms-readout" id="pgStepsVal"></span></label>
    <label><input type="checkbox" id="pgShowReal"> 叠加真实样本</label>
  </div>
  <canvas id="pgCv" height="430" aria-label="粒子终点散点：所选求解器与步数下的生成样本"></canvas>
  <div class="fms-stats">
    <span><span class="k">NFE</span> <b id="pgNfe"></b></span>
    <span><span class="k">落峰比例 (2σ 内)</span> <b id="pgHqf"></b></span>
    <span><span class="k">覆盖模式</span> <b id="pgModes"></b>/8</span>
  </div>
  <figcaption><b>图 4 · 采样试验场（交互：换求解器、拖步数、叠加真实样本）。</b>1500 个粒子在精确流场上的积分终点散点；红 × 为八个真实模式中心。「落峰比例」即实验指标 high_quality_frac（落在任一模式 2σ 半径内的比例）——注意完美复制的理论上限约 0.865 而非 1（2D 高斯本身只有 86.5% 的质量在 2σ 内）。Dopri5 档使用固定 rtol=10⁻³。</figcaption>
</figure>

<div class="ln-eyebrow">实验 · 真实模型</div>

## 8 · 实验：真实训练模型上的对比

玩具场好懂，但结论要在**真实训练出来的模型**上成立才算数。实验设置：在 8-Gaussians 与 Two-Moons 两份 2D 数据上各训练一个流匹配 MLP（15000 步、EMA、固定种子），冻结后让五种求解器从**同一批** 2000 个评测噪声出发采样（配对比较），横轴 NFE。完整代码与数据见 `fm_solvers.py` / `results.json`。

<div class="ln-chips">
  <span class="ln-chip">Two-Moons 到地板 NFE · Euler <b>40</b></span>
  <span class="ln-chip">Midpoint/Heun <b>16</b></span>
  <span class="ln-chip good">RK4 <b>12</b></span>
  <span class="ln-chip">8-Gaussians 采样地板 W2 <b>0.165</b></span>
  <span class="ln-chip bad">NFE=4 时 Euler <b>0.194</b> &lt; Heun <b>0.280</b></span>
</div>

### 8.1 · 分布质量 W2：很快就到「地板」

W2（2-Wasserstein 距离）衡量生成分布与真实数据的距离。图 5 里它先快速下降，然后**饱和**——剩下的差距来自模型误差与 2000 点采样噪声，再加 NFE 也买不动。求解器的差别只在「多快到地板」。

<figure class="fms-panel">
  <div class="fms-legend" id="w2Legend"></div>
  <canvas id="w2Cv" height="620" aria-label="两个数据集上 W2 随 NFE 变化的折线图"></canvas>
  <figcaption><b>图 5 · W2 vs NFE，两份数据集（真实实验数据；交互：悬停读数）。</b>虚线为 sample floor（两组独立真实样本之间的 W2，即采样噪声地板）。8-Gaussians 的地板 (0.165) 比模型收敛值还高，所以那一侧 W2 在 NFE≈6–8 就饱和、区分不了求解器；Two-Moons 侧能看清顺序：到地板 Euler 需 NFE≈40、Midpoint/Heun≈16、RK4≈12。</figcaption>
</figure>

### 8.2 · 纯数值精度 endpoint_err：教科书斜率现身

把 RK4 走 500 步（NFE=2000）当作「真答案」，量每个求解器终点偏了多少。这个指标剥离了模型误差，只剩离散化误差——于是第 5 节的教科书斜率在**训练出来的网络**上原样复现（图 6）。

<figure class="fms-panel">
  <div class="fms-legend" id="epLegend"></div>
  <canvas id="epCv" height="620" aria-label="两个数据集上 endpoint 误差随 NFE 变化的 log-log 折线图"></canvas>
  <figcaption><b>图 6 · endpoint 误差 vs NFE（log-log，真实实验数据；交互：悬停读数）。</b>灰虚线为斜率 −1/−2/−4 参考线：Euler 贴 −1，Midpoint/Heun 贴 −2，RK4 早期贴 −4，高阶方法最终触到 ~10⁻³ 的地板——那是 RK4-500 参考解自身的精度极限。<span class="fms-t-dopri5">★ Dopri5</span> 不用调步数，天然落在帕累托前沿。</figcaption>
</figure>

### 8.3 · 五条值得记住的结论

<div class="fms-takeaways">
  <div class="fms-tk"><div class="n">一</div><p class="hd">到地板的最小 NFE 差 3 倍</p>
    <p>Two-Moons 上达到 ≤1.05× 地板：Euler 40 · Midpoint 16 · Heun 16 · RK4 12。高阶方法用约 1/3 的算力买到同样的分布质量。</p></div>
  <div class="fms-tk"><div class="n">二</div><p class="hd">极低 NFE 时高阶未必赢</p>
    <p>NFE=4 处 8-Gaussians 的 W2：Euler 0.194 &lt; Heun 0.280。步数太少时，高阶法「每步昂贵」的代价压过了「每步精确」的收益——在图 4 里亲手试。</p></div>
  <div class="fms-tk"><div class="n">三</div><p class="hd">W2 有地板，endpoint 没谎</p>
    <p>分布指标会被采样噪声饱和（8G 地板 0.165），饱和后所有求解器看起来一样好。要比数值精度，得用配对的 endpoint 误差。</p></div>
  <div class="fms-tk"><div class="n">四</div><p class="hd">Dopri5 = 免调参的前沿</p>
    <p>给一个 rtol，它自动配速：粗容差 NFE≈37 就有不错的样本，紧容差平滑逼近参考解，全程不需要人肉扫步数网格。</p></div>
  <div class="fms-tk"><div class="n">五</div><p class="hd">场坏了，谁也救不了</p>
    <p>欠训练消融（1000 步训练）：W2 地板从 0.05 飙到 0.33+，加 NFE 无用；Two-Moons 上 Euler 反超 RK4——高阶法只是更精确地积分了一个<b>错误的场</b>。</p></div>
</div>

<div class="ln-eyebrow">收束 · 决策表</div>

## 9 · 实践指南

把全文压缩成一张决策表：

| 求解器 | 阶 | NFE/步 | 什么时候用 |
|---|---|---|---|
| <span class="fms-chip fms-c-euler"></span>Euler | 1 | 1 | 预算极低（NFE≤6）或场很糙（欠训练/蒸馏模型）时的保底选择 |
| <span class="fms-chip fms-c-midpoint"></span>Midpoint | 2 | 2 | 低 NFE 甜点区（8–16）；本实验里低预算下最稳的一档 |
| <span class="fms-chip fms-c-heun"></span>Heun | 2 | 2 | 同为 2 阶，误差常数略不同；扩散社区常用（EDM 的默认二阶法<sup>[[9]](#refs)</sup>） |
| <span class="fms-chip fms-c-rk4"></span>RK4 | 4 | 4 | 中高预算（NFE≥12）要高保真终点；到参考精度最快 |
| <span class="fms-chip fms-c-dopri5"></span>Dopri5 | 5(4) | 自适应 | 不想调步数：设 rtol 即可，自动落在帕累托前沿 |

再往前一步就出了本文的范围：真实的图像/视频生成里，低 NFE 区间还有一批**利用扩散 ODE 结构的专用求解器**——DDIM<sup>[[8]](#refs)</sup>、DPM-Solver++<sup>[[10]](#refs)</sup>、UniPC<sup>[[11]](#refs)</sup>，以及 SD3/Flux 这类流匹配模型实际采用的「Euler + 移位时间表」<sup>[[12]](#refs)</sup>。它们与通用求解器的关系，见问答最后一条。

*与代码对照：solver 实现在 `fm_solvers.py` 的 `solve_euler / solve_midpoint / solve_heun / solve_rk4 / solve_dopri5`；收敛阶与 NFE 记账由 `test_solvers.py` 把关；本页数据来自 `summary.csv`。*

<div class="ln-eyebrow">附录 · 问答</div>

## 10 · 问答

*来自学习过程中的真实提问，持续补充。点开每条查看解答。*

???+ question "图里的 y 轴是「精度」吗？它和生成质量是一回事吗？"

    图 2（第 5 节）与图 6（8.2 节）的 y 轴是**终点误差**：求解器算出的终点到真解（图 2 是解析解，图 6 是 RK4-500 参考解）的距离，log 刻度。它衡量的是**纯数值积分精度**——越低积分越准，log-log 斜率就是收敛阶。

    但它**不是生成质量**。质量看图 5（8.1 节）的 W2（生成分布 vs 真实数据）。两者的分工正是结论三：W2 会被采样噪声「饱和」，饱和后所有求解器看起来一样好；endpoint 误差剥离了模型误差与采样噪声，永远能分出高下。一句话：*W2 回答「样本像不像」，endpoint 误差回答「积分准不准」。*

??? question "rtol / atol 到底是什么？为什么叫「相对」容差？"

    容差 = 你允许求解器**每一步犯多大的错**。有两种定法：**atol（绝对）**——「误差不许超过 0.0001，不管数值多大」，一把固定刻度的尺子；**rtol（相对）**——「误差不许超过数值本身的千分之一」，按比例要求精度。

    为什么要「相对」？量珠穆朗玛峰差 1 米无所谓（占比 0.01%），量身高差 1 米就离谱——**同样的误差严不严重，取决于被测的东西多大**。但纯相对标准在数值趋近 0 时会失控（「0 的千分之一还是 0」，步长被逼到无限小），所以用 atol 兜底。实际公式：

    $$
    \text{允许误差} = \mathrm{atol} + \mathrm{rtol}\cdot|x|
    $$

    数值大时 rtol 项主导（按比例），数值趋近 0 时 atol 项接管（绝对底线）。本实验取 atol = rtol/100。

??? question "用大白话解释 RK4？"

    把积分一步想成**闭着眼开一段弯路**，睁眼看一次路 = 一次网络前向（1 NFE）。

    <span class="fms-t-euler">Euler</span>：出发前看一眼方向，闭眼开完全程——路一弯就冲出去（图 4 里低步数的「弥散环」就是这么来的）。

    <span class="fms-t-rk4">RK4</span> 聪明地看四次：出发一眼（k₁）→ 按 k₁ 开到**半路**再看（k₂）→ 退回起点按 k₂ 重开到半路又看（k₃）→ 按 k₃ 开到**终点**看最后一眼（k₄）。然后加权平均 (k₁+2k₂+2k₃+k₄)/6——半路两眼权重最大，因为半路的方向最能代表全程——用这个「平均方向」真正走完这一步。

    本质：四次采样拼出整段路的**平均速度**（数学上是步内做辛普森积分），所以弯路也能贴着走。代价是每步 4 NFE，回报是步长减半、误差降 16 倍——很快回本。在图 1 里切到 RK4 拖大 dt，能亲眼看到这四眼的位置。

??? question "用大白话解释 Dopri5？"

    RK4 是个好司机，但**步子多大得你提前定死**。Dopri5 = 好司机 + 自动巡航，四个招数：

    **① 一步看七眼，拼出两个版本的答案**——精细版（5 阶）与粗糙版（4 阶）。<br>
    **② 两版之差 ≈ 这步错了多少**——不知道真答案也能自我检查，而且免费。<br>
    **③ 据此自动调速**：两版几乎一样 → 路很直 → 下一步油门加大（dt 最多 ×5）；差太多 → 这步**作废重开**，步子收紧（最少 ×0.2）。像导航限速：高速 120，进村自动 30——图 3 的「前松后紧 + 红框重试」就是它在工作。<br>
    **④ 尾巴接头（FSAL）**：这步的最后一眼恰好落在下一步起点上，直接复用，七眼只花六眼的钱。

    你只需要说一句「我要千分之一的精度」（rtol），哪里快哪里慢它全自己决定——这就是「免调参」的含义。

??? question "实际用 diffusion / flow-matching 模型时，首选是不是 Dopri5？"

    **不是。** Dopri5 是「不想调参时的稳妥默认」（SciPy `RK45`、MATLAB `ode45`、torchdiffeq 的默认都是它），但生产级生成模型的低 NFE 区间被专用方法占据：

    | 场景 | 主流选择 |
    |---|---|
    | Stable Diffusion 3 / Flux（流匹配） | Euler + 移位时间表，NFE 20–30<sup>[[12]](#refs)</sup> |
    | 扩散模型低 NFE 采样 | DPM-Solver++ / UniPC（NFE 10–20）<sup>[[10]](#refs)</sup><sup>[[11]](#refs)</sup> |
    | EDM / Karras 系 | Heun + 精心设计的 σ 网格<sup>[[9]](#refs)</sup> |
    | 确定性快速采样的源头 | DDIM<sup>[[8]](#refs)</sup> |
    | 蒸馏模型（LCM / Turbo / Lightning） | Euler 1–4 步 |
    | 精确似然 / inversion / 参考解 | **Dopri5 ✓**<sup>[[6]](#refs)</sup> |

    生产不用它的四个原因：① 低 NFE 打不过专用法——本页数据里 Dopri5 最粗容差也要 NFE≈37 起步，而 Midpoint 在 NFE=16 就到地板；拒绝步在大模型上白烧钱。② DPM-Solver 系利用扩散 ODE 的**半线性结构**（线性部分解析积分），通用 RK 把网络当黑盒，放弃了这个先验。③ 自适应步长意味着每张图 NFE 不可预测，固定步数延迟恒定、好部署。④ 时间表（σ/timestep schedule）往往比 solver 阶数更重要，固定步长方法可以直接调它。

    **Dopri5 真正的主场**：面对陌生的场、没有调参预算、要高精度（算 likelihood、做 inversion、生成参考解）、NFE 预算 ≥30 且不在乎波动。一句话：*免调参的帕累托前沿，不是全场最优——预算紧张的生产采样，前沿在专用低 NFE 方法手里。*

## 参考文献 { #refs }

1. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018.
2. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.（probability flow ODE）
3. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
4. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* ICLR 2023.
5. X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023.
6. J. R. Dormand, P. J. Prince. *A family of embedded Runge–Kutta formulae.* Journal of Computational and Applied Mathematics, 6(1):19–26, 1980.
7. E. Hairer, S. P. Nørsett, G. Wanner. *Solving Ordinary Differential Equations I: Nonstiff Problems.* Springer, 2nd ed., 1993.（步长控制与刚性问题的标准教材）
8. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
9. T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022.（EDM，Heun 采样器与 σ 网格）
10. C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, J. Zhu. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS 2022；及 *DPM-Solver++*, arXiv:2211.01095.
11. W. Zhao, L. Bai, Y. Rao, J. Zhou, J. Lu. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.* NeurIPS 2023.
12. P. Esser, S. Kulal, A. Blattmann, et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024.（Stable Diffusion 3）

*本文实验部分（图 5–6 数据、训练配置、指标定义）完全由配套代码 `fm_solvers.py / plots.py / test_solvers.py` 生成，可一键复现；本页交互图均为浏览器内真实数值积分。*
