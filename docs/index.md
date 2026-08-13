---
hide:
  - navigation
  - toc
---

# Learning Notes

我的个人学习笔记知识库 —— 把读过、学过、调试过的东西沉淀成可检索的笔记。

<div class="grid cards" markdown>

-   :material-brain: __机器学习__

    ---

    模型、训练、调参与踩坑记录。

    [:octicons-arrow-right-24: 进入](ml/index.md)

-   :material-robot-industrial: __机器人__

    ---

    机器人学习 / 控制 / 感知 / 仿真。

    [:octicons-arrow-right-24: 进入](robotics/index.md)

-   :material-function-variant: __数学__

    ---

    线性代数 / 概率 / 优化等基础笔记。

    [:octicons-arrow-right-24: 进入](math/index.md)

-   :material-book-open-variant: __阅读__

    ---

    书籍 / 博客 / 课程的读书笔记。（筹备中）

    [:octicons-arrow-right-24: 进入](reading/index.md)

</div>

## 全部笔记

<!-- 按日期倒序；新增笔记时在此登记一行（标题 · 分区 · 日期 · 一句话导语） -->

- [Action Tokenization：把连续动作离散成 token](robotics/action-tokenization.md)
  <small>机器人 · 2026-08-06</small> —— 沿「连续动作 → 离散 token → 连续动作」这条链，拆开 π₀-FAST 式离散自回归路线的每个环节。
- [流匹配 ODE 求解器 · 图解](ml/fm-solvers.md)
  <small>机器学习 · 2026-08-06</small> —— Euler / Midpoint / Heun / RK4 / DoPri5 在流匹配采样里的精度-成本取舍，配交互实验。
- [自回归模型：BERT / GPT 与 Prefix-LM 架构详解](ml/autoregressive-models.md)
  <small>机器学习 · 2026-08-06</small> —— 序列建模范式的分野，以及机器人 VLA 为什么用 Prefix-LM 混合体。
- [Fourier 变换与 DCT：从频域到能量压缩](math/fourier-dct.md)
  <small>数学 · 2026-08-06</small> —— 从傅里叶级数到 DCT 能量压缩：为什么平滑信号的能量会集中到少数低频系数。
- [Lie Theory：面向优化与机器人的直观入门](math/lie-theory.md)
  <small>数学 · 2026-08-06</small> —— 流形、切空间与 ⊞/⊟ 算子：旋转与位姿上的优化为什么要这么做。

---

<small>:material-palette-outline: [样式模板 / Style Guide](style/index.md) —— 统一风格的元素参照。</small>
