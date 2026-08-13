# 自回归模型：BERT / GPT 与 Prefix-LM 架构详解

<div class="ln-byline">2026-08-06 · 阅读约 9 分钟 · Yufeng Jin</div>

<p class="ln-lead" markdown>有哪些序列建模范式、BERT 与 GPT 到底差在哪、为什么机器人 VLA（如 π₀-FAST）用的是两者的混合体 Prefix-LM。是 [Action Tokenization](../robotics/action-tokenization.md) 里"自回归 + 交叉熵""prefix 双向 / action 因果掩码"的展开。</p>

!!! note "一句话先记住"
    所有差别都归结到**一件事**：**每个 token 能"看到"哪些其他 token**（注意力掩码）。**双向**看全部 → 擅长理解（BERT）；**因果**只看左边 → 能自回归生成（GPT）。架构其实是同一套 Transformer block，区别只在掩码和训练目标。

---

<div class="ln-eyebrow">预备知识</div>

## 预备知识

正文会反复用到下面这些概念。每条按「**中文名 = 英文名 = 最小定义**」给出；按本站惯例，正文中直接使用英文术语（与此处三元组一一对应）：

- **自注意力 = self-attention** = 序列中每个 token 以自己的 query 去匹配其它 token 的 key、按匹配权重汇聚其 value，从上下文提取信息的机制<sup>[[1]](#refs)</sup>。
- **注意力掩码 = attention mask** = 规定「每个 query 位置允许注意哪些 key 位置」的布尔矩阵；本文三种范式（双向 / 因果 / Prefix-LM）的差别全部落在这张矩阵上<sup>[[1]](#refs)</sup>。
- **交叉注意力 = cross-attention** = decoder 侧的 query 去注意 encoder 输出的 key/value，把输入条件引入生成侧的注意力形式<sup>[[1]](#refs)</sup>。
- **自回归 = autoregressive** = 按链式法则把联合概率拆成逐 token 的条件概率、用已生成的历史预测下一个 token 的建模方式<sup>[[2]](#refs)</sup>。
- **掩码语言建模 = masked language modeling（MLM）** = 遮住输入中的部分 token、用未遮住的左右上下文重建它们的训练目标<sup>[[3]](#refs)</sup>。
- **交叉熵 = cross-entropy** = 衡量预测分布与目标分布差异的损失函数；next-token 训练即对正确 token 取负对数似然<sup>[[2]](#refs)</sup>。

---

<div class="ln-eyebrow">框架 01 · 在估计什么</div>

## 0. 序列模型到底在估计什么

给一串 token $x_1,x_2,\dots,x_n$，语言模型本质是在估计它们的**联合概率** $p(x_1,\dots,x_n)$。怎么拆这个联合概率，决定了模型范式：

| 范式 | 怎么分解 / 训练目标 | 能生成吗 |
|---|---|---|
| **自回归 (AR)** | $p(x)=\prod_t p(x_t\mid x_{<t})$，预测下一个 token | ✅ 天生为生成而设计 |
| **自编码 / 掩码 (AE/MLM)** | 遮住一部分 $\tilde x$，重建 $p(x_{\text{masked}}\mid \tilde x)$ | ❌ 主要用于理解/表征 |
| **序列到序列 (seq2seq)** | 编码输入，解码输出 $p(y\mid x)$ | ✅ 翻译/摘要类 |

!!! tip "“自回归”名字的由来"
    auto-regressive = 用自己**已经生成的**历史 $x_{<t}$ 去回归预测**下一个** $x_t$，再把它接回输入，循环往复。和动作生成里"逐 token 采样、把 `|` 当终止符"是同一件事。

---

<div class="ln-eyebrow">框架 02 · 三种搭法</div>

## 1. 三大架构范式（同一种积木，三种搭法）

Transformer 的积木是「多头自注意力 + 前馈层」<sup>[[1]](#refs)</sup>。三大范式只是**怎么堆叠 + 用什么掩码**不同：

<div class="ln-fig">
<svg viewBox="0 0 820 250" role="img" aria-label="三种架构对比">
  <g>
    <text x="120" y="24" text-anchor="middle" fill="#3fb950" font-size="15" font-weight="700">Encoder-only（BERT）</text>
    <rect x="30" y="40" width="180" height="150" rx="10" fill="#FFFFFF" stroke="#3fb950"/>
    <rect x="55" y="60" width="130" height="34" rx="6" fill="rgba(63,185,80,.15)" stroke="#3fb950"/><text x="120" y="82" text-anchor="middle" fill="#201F1C" font-size="12">双向自注意力 ×N</text>
    <rect x="55" y="104" width="130" height="26" rx="6" fill="#F1EFEA" stroke="#D8D2C7"/><text x="120" y="121" text-anchor="middle" fill="#6B675F" font-size="11">前馈</text>
    <text x="120" y="160" text-anchor="middle" fill="#6B675F" font-size="11">输入全部可见</text>
    <text x="120" y="178" text-anchor="middle" fill="#6B675F" font-size="11">↑ 输出每个位置的表征</text>
  </g>
  <g>
    <text x="410" y="24" text-anchor="middle" fill="#58a6ff" font-size="15" font-weight="700">Decoder-only（GPT）</text>
    <rect x="320" y="40" width="180" height="150" rx="10" fill="#FFFFFF" stroke="#58a6ff"/>
    <rect x="345" y="60" width="130" height="34" rx="6" fill="rgba(88,166,255,.15)" stroke="#58a6ff"/><text x="410" y="82" text-anchor="middle" fill="#201F1C" font-size="12">因果自注意力 ×N</text>
    <rect x="345" y="104" width="130" height="26" rx="6" fill="#F1EFEA" stroke="#D8D2C7"/><text x="410" y="121" text-anchor="middle" fill="#6B675F" font-size="11">前馈</text>
    <text x="410" y="160" text-anchor="middle" fill="#6B675F" font-size="11">只能看左边</text>
    <text x="410" y="178" text-anchor="middle" fill="#6B675F" font-size="11">↑ 预测下一个 token</text>
  </g>
  <g>
    <text x="700" y="24" text-anchor="middle" fill="#bc8cff" font-size="15" font-weight="700">Encoder-Decoder（T5）</text>
    <rect x="600" y="40" width="92" height="150" rx="10" fill="#FFFFFF" stroke="#3fb950"/>
    <text x="646" y="115" text-anchor="middle" fill="#3fb950" font-size="12" transform="rotate(-90 646 115)">编码器(双向)</text>
    <rect x="708" y="40" width="92" height="150" rx="10" fill="#FFFFFF" stroke="#58a6ff"/>
    <text x="754" y="115" text-anchor="middle" fill="#58a6ff" font-size="12" transform="rotate(-90 754 115)">解码器(因果)</text>
    <line x1="692" y1="115" x2="708" y2="115" stroke="#bc8cff" stroke-width="2" marker-end="url(#ar)"/>
    <text x="700" y="208" text-anchor="middle" fill="#6B675F" font-size="11">交叉注意力连接</text>
  </g>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#bc8cff"/></marker></defs>
</svg>
<div class="ln-fig-cap">图 1 · 三大架构范式对比。同样的 Transformer block，区别只在「掩码方向」和「堆叠方式」。看每个盒子里注意力层的类型即可：绿色 = 双向注意力（encoder 侧），蓝色 = 因果注意力（decoder 侧），紫色箭头 = 连接两者的交叉注意力。结论：范式之别不在积木本身，而在掩码与堆叠。</div>
</div>

---

<div class="ln-eyebrow">机制 · 注意力掩码</div>

## 2. 核心机制：注意力掩码（亲手切换看区别）

自注意力让每个 token 去"查询"其他 token。**掩码**决定哪些查询被允许。这是 BERT / GPT / Prefix-LM 唯一的本质差别。下面这张图：行 = 正在计算的 token（query），列 = 被注意的 token（key），**亮格 = 允许注意**。

<div class="ln-demo">
<div class="ln-demo-title">图 2 · Demo：三种注意力掩码对比</div>
<div class="ln-demo-hint">点按钮切换。序列假设是 <code>[CLS] 任务 提示 ; A1 A2 A3 |</code>，前 4 个（含分隔符 <code>;</code>）是 prompt 前缀、后 4 个是要生成的 action。看什么：行 = query（谁在看），列 = key（被注意）；token 标签绿色 = prompt 前缀、蓝色 = action 后缀；亮格 = 允许注意（Prefix-LM 模式下前缀内部的双向区显示为绿色），灰格 = 被掩码禁止。结论：因果 = 下三角，双向 = 全亮，Prefix-LM = 左上双向块 + 右下因果三角。</div>
<canvas id="cvMask" width="460" height="460" style="margin:auto"></canvas>
<div class="ln-controls" style="justify-content:center">
<button id="mCausal" class="on">因果 (GPT)</button>
<button id="mBidir">双向 (BERT)</button>
<button id="mPrefix">Prefix-LM (π₀-FAST)</button>
</div>
<div class="ln-readout" id="maskOut"></div>
</div>

- **因果掩码（下三角）**：token $t$ 只能注意 $\le t$ 的位置。保证"预测下一个时不偷看未来"，是**自回归生成的前提**。GPT 全程用它。
- **双向掩码（全亮）**：每个 token 看到整句。理解任务（分类、抽取）需要全局上下文，但**无法直接逐个生成**（会偷看答案）。BERT 用它。

---

<div class="ln-eyebrow">架构 01 · BERT</div>

## 3. BERT — 编码器 / 双向 / 掩码语言模型

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers<sup>[[3]](#refs)</sup>。只用 Transformer 的**编码器**栈，全程双向注意力。

训练目标：**MLM（Masked Language Modeling，完形填空）**

```text
输入:  机器人 [MASK] 可爱        ← 随机遮住 15% 的 token
目标:  预测 [MASK] = "很"        ← 用左右两边的上下文一起猜
```

因为要"用两边猜中间"，所以必须双向。另一个经典目标是 NSP（判断两句是否相邻）<sup>[[3]](#refs)</sup>，后续工作（RoBERTa）发现可去掉<sup>[[4]](#refs)</sup>。

!!! warning "BERT 不能直接做生成"
    它擅长把整句"读懂"成向量表征（适合分类、检索、命名实体识别），但因为是双向的，**没法逐 token 自回归生成**——所以聊天/写作类任务用的是 GPT 那一支。

| 属性 | BERT |
|---|---|
| 结构 | Encoder-only |
| 注意力 | 双向 |
| 训练目标 | MLM（+NSP） |
| 强项 | 理解 / 表征（分类、抽取、检索） |
| 能否生成 | ❌ |
| 代表后代 | RoBERTa, ALBERT, DeBERTa, ELECTRA |

---

<div class="ln-eyebrow">架构 02 · GPT</div>

## 4. GPT — 解码器 / 因果 / 自回归

**G**enerative **P**re-trained **T**ransformer<sup>[[5]](#refs)</sup>。只用 Transformer 的**解码器**栈（去掉了对编码器的交叉注意力），全程因果掩码。

训练目标：**next-token prediction（预测下一个 token）**

$$
\mathcal{L}=-\sum_t \log p_\theta(x_t\mid x_{<t})
$$

这正是 π₀-FAST 把动作 token 当语言、做 next-token 交叉熵的目标。下面这个 demo 演示自回归是怎么"逐字吐出"的：

<div class="ln-demo">
<div class="ln-demo-title">图 3 · Demo：自回归生成（逐 token 采样）</div>
<div class="ln-demo-hint">点"生成下一个 token"，模型基于已生成的历史给出下一个 token 的概率分布并采样，再接回输入——这就是 GPT / 动作逐 token 采样的循环。看什么：上排方框 = 已生成的历史序列（新 token 依次接到右侧），下方紫色柱状图 = 下一个 token 的概率分布（柱顶标注各自概率）；拖动 temperature 滑杆可见分布被压平或变尖，采样随之更随机或更确定。结论：生成 = 「预测分布 → 采样 → 接回输入」的循环。</div>
<canvas id="cvGen" width="900" height="220"></canvas>
<div class="ln-controls">
<button id="genStep">▶ 生成下一个 token</button>
<button id="genReset">↺ 重置</button>
<label>temperature <input type="range" id="genT" min="10" max="150" value="80"><span class="ln-val" id="genTV">0.80</span></label>
</div>
<div class="ln-readout" id="genOut"></div>
</div>

| 属性 | GPT |
|---|---|
| 结构 | Decoder-only |
| 注意力 | 因果（单向） |
| 训练目标 | next-token 预测 |
| 强项 | 生成（对话、写作、代码） |
| 能否生成 | ✅（天生） |
| 代表家族 | GPT-2/3/4<sup>[[6]](#refs)[[7]](#refs)</sup>, LLaMA, Mistral, Qwen, Gemma, Claude |

---

<div class="ln-eyebrow">架构 03 · T5 / BART</div>

## 5. T5 / BART — 编码-解码（seq2seq）

把两支合起来：**编码器**双向读入输入（如英文句子），**解码器**因果地生成输出（如中文句子），中间用**交叉注意力**让解码器读到编码器的表征。天生适合"输入→输出"的转换任务（翻译、摘要）。

!!! tip "T5 的统一视角"
    T5 把**所有任务都变成"文本→文本"**：分类 = 生成标签词，翻译 = 生成译文<sup>[[8]](#refs)</sup>。BART 则用"破坏文本再重建"的去噪目标预训练<sup>[[9]](#refs)</sup>。两者都是 encoder-decoder。

---

<div class="ln-eyebrow">架构 04 · Prefix-LM</div>

## 6. Prefix-LM — 前缀双向 + 后缀因果（π₀-FAST 用的就是它）

这是连接本页和动作 tokenization 的关键。Prefix-LM（前缀语言模型）是 decoder-only 的一个变体<sup>[[8]](#refs)[[10]](#refs)</sup>：把序列分成**前缀**和**后缀**两段，用**一张混合掩码**：

- **前缀**（prompt + 图像 + 离散化 state）：内部**双向**注意（像 encoder，充分理解条件）。
- **后缀**（要生成的 action token）：**因果**注意（像 decoder，保证自回归）。后缀能看到全部前缀。

用上面 Demo（图 2）的"Prefix-LM"按钮可以直接看到这张掩码的形状：左上是一个全亮方块（前缀双向），右下是一个下三角（后缀因果）。

!!! note "为什么机器人 VLA 选 Prefix-LM"
    观测（图像/指令/状态）是**已知条件**，应当被充分双向理解；而动作是**要生成的**，必须因果。Prefix-LM 一张掩码同时满足两边——PaliGemma<sup>[[11]](#refs)</sup>、π₀-FAST<sup>[[12]](#refs)</sup> 都用它。

---

<div class="ln-eyebrow">速查 · 家族全表</div>

## 7. 模型家族全表（速查）

| 模型 | 范式 | 注意力 | 训练目标 | 典型用途 |
|---|---|---|---|---|
| BERT / RoBERTa | Encoder-only | 双向 | MLM | 理解、分类、检索 |
| GPT / LLaMA / Gemma / Claude | Decoder-only | 因果 | next-token | 生成、对话 |
| T5 / BART | Encoder-Decoder | 编码双向+解码因果 | 去噪 / span 重建 | 翻译、摘要 |
| PaLM / UL2 / PaliGemma | Prefix-LM / 混合 | 前缀双向+后缀因果 | (prefix) next-token | 多模态、条件生成 |
| π₀-FAST | Prefix-LM (VLA) | 前缀双向+动作因果 | action token 交叉熵 | 机器人动作生成 |

!!! warning "为什么现在主流是 Decoder-only"
    GPT 路线（decoder-only + 因果）用一个目标就能既学表征又能生成，规模化最简单，所以大模型几乎都走这支。BERT 这种 encoder-only 仍在"只需理解、不需生成"的场景（如检索、排序）很强。

---

<div class="ln-eyebrow">收束 · 回到 tokenization</div>

## 8. 回到 action tokenization

串起来（详见 [Action Tokenization](../robotics/action-tokenization.md)）：

1. 动作被 tokenizer（FAST/FSQ/Binning）变成离散 token。
2. π₀-FAST 用 **Prefix-LM**：前缀（观测+指令+state）双向、后缀（action）因果<sup>[[12]](#refs)</sup>。
3. 训练用 **GPT 式 next-token 交叉熵**，只在 action 段计损失<sup>[[12]](#refs)</sup>。
4. 推理用 **自回归采样**（本页 Demo，图 3），靠 temperature 取出多模态的不同解。

<div class="ln-lesson" markdown>
**现在你应该能回答**：为什么 VLA 不用纯 BERT？（不能生成）为什么不用纯因果 GPT 而要 Prefix-LM？（让观测条件被双向充分理解）为什么训练像 GPT？（next-token 交叉熵 + 天然多模态）。
</div>

---

## 延伸阅读

- [Action Tokenization](../robotics/action-tokenization.md) — 机器人动作离散化主线（FAST / FSQ / Binning + 自回归 + 多模态）
- [Fourier 变换与 DCT](../math/fourier-dct.md) — 频域变换与能量压缩，FAST 第一步的数学内核

---

## References { #refs }

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
2. Goodfellow, I., Bengio, Y., Courville, A. *Deep Learning*. MIT Press, 2016.
3. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019. arXiv:1810.04805.
4. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V. *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692.
5. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. *Improving Language Understanding by Generative Pre-Training*. OpenAI technical report, 2018.
6. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. *Language Models are Unsupervised Multitask Learners*. OpenAI technical report, 2019.
7. Brown, T. B., Mann, B., Ryder, N., et al. *Language Models are Few-Shot Learners*. NeurIPS 2020. arXiv:2005.14165.
8. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P. J. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR 21, 2020. arXiv:1910.10683.
9. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., Zettlemoyer, L. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020. arXiv:1910.13461.
10. Tay, Y., Dehghani, M., Tran, V. Q., et al. *UL2: Unifying Language Learning Paradigms*. arXiv:2205.05131.
11. Beyer, L., Steiner, A., et al. *PaliGemma: A versatile 3B VLM for transfer*. arXiv:2407.07726.
12. Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., Levine, S. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747.
