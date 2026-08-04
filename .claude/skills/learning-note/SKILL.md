---
name: learning-note
description: Use when turning experiment results, benchmark numbers, draft notes, or an existing teaching blog into a learning-notes 教学笔记 — requests like 制作学习笔记 / 教学文章 / 学习blog / 把这篇 blog 搬进 learning-notes / update learning-notes。输出为本仓库的 MkDocs 笔记（md + 可选交互 demo js）。
---

# Learning Note：从材料与草稿制作教学笔记

## Overview

把「材料（实测数字/结果表/已有 blog）+ 草稿（想讲什么、猜想、图的构思）」编译成本仓库的教学笔记。
核心原则：**笔记是材料的忠实渲染，不是再创作**——每个数字可回溯，每个猜想带标签，
每个借来的论断带引用。

> 本 skill = 通用写作规范（源自 learning-blog-recipe，每条规则对应一次实测抓到的失败）
> ⊕ 本仓库的落地件（MkDocs / 分类 / demo / 双语 / 构建检查）。

## 落地件（本仓库的输出由这些部件构成）

1. **笔记 md** → `docs/<分类>/<kebab-case>.md`。分类四选一（拿不准就问，别猜）：
   - `ml/` 机器学习（模型、训练、生成模型、采样……）
   - `robotics/` 机器人（控制、感知、动作表示……）
   - `math/` 数学（纯数学工具：变换、群论、数值方法本体……）
   - `reading/` 阅读
   - 边界判据：**以读者会去哪里找它为准**。例：流匹配 ODE 求解器 → `ml/`
     （读者带着「flow matching 采样」的问题来）；Fourier/DCT → `math/`（纯数学工具）。
2. **交互 demo（如有）** → `docs/javascripts/<同名>.js`，规则见下节。
3. **注册三件套**（缺一不可，漏了页面就是孤儿）：
   - `mkdocs.yml` 的 `nav:` 加中文标题条目；
   - `extra_javascript:` 注册 demo js（如有）；
   - `i18n` 插件的 `nav_translations:` 加英文标题（站点双语，界面标题必须两份）。
4. **构建检查**：`.venv/bin/mkdocs build --strict` 零 WARNING 才算完成；
   同时确认 `site/<分类>/<slug>/` 与 `site/en/<分类>/<slug>/` 都生成了。

## 文章结构契约（按序，仅由这些部件构成）

1. **导语**：两三句话说清这篇是什么、读完得到什么（正文首段，不加 Byline——
   本站笔记不署名不标日期，git 历史即记录）。
2. **引言**：一个具体场景钩子 + 主题为什么值得学。
3. **动机**：这个问题难在哪、现有讲法缺什么。
4. **预备知识**：每个概念写成**三元组：中文名 = 英文名 = 最小定义**，写完逐条自查
   三者配对无误（术语配反是实测出现过的失败）；正文用词必须与此处一致；
   教科书级概念也要挂引用。
5. **正文章节**：每章固定四拍——直觉 → 机制/数学 → 实测（引材料数字）→ 教训。
   章节主题清单 = 草稿点名的主题，一一对应；不加「未来方向」等模板章节。
6. **图**：每图 = 编号 + 图注。图注写实质：读者该看什么、颜色/轴什么含义、结论是什么。
   草稿只给构思时，输出图位描述 + 完整图注。
7. **总结**（如有）：只回收各章已出现的「教训」，不引入新主张。
8. **References**：文末编号列表（作者、标题、venue/arXiv）；
   正文借来的论断处挂 `<sup>[[n]](#refs)</sup>`，参考文献标题带 `{ #refs }` 锚点。

## 三级论断标签（每个论断属于且仅属于一级）

| 级别 | 判定 | 写法 |
|---|---|---|
| 实测 | 数字在材料里 | 直接引用，数值一字不改 |
| 文献 | 来自论文/教科书 | 正文挂 [n]，进 References |
| 推测 | 草稿标注的猜想、或任何材料未验证的解释 | 全文恰好出现一次：显式标注「猜测/待验证」+ **一个句号以内**，下一句必须换话题 |

推测的镜像规则：**「教训/总结」段里出现的每一个解释，检查它是否有材料支撑**；
没有支撑的解释就是推测——要么删掉、要么并入那唯一一次带标签的陈述。
（实测失败：推测在标注小节里守规，却换个说法溜进「教训」段变成了结论。）

## 数字保真规则

- 输出里的每个数字都必须在材料中逐字存在。写完后**逐个数字回查材料**
  （批量搬运时用脚本抽查关键数字 + 数据块 diff，别靠眼睛）。
- 材料没给的量：写「材料未含」，不写「中等」「未测」之类的臆断占位。
- 数字连同其语境一起搬运：材料说 AUROC 0.548 就是 0.548，不改写成「几乎随机」——
  除非材料本身这么说。
- 解释机制前先核对任务类型（回归任务不谈 softmax；分类任务不谈方差头）。
- **范围词对齐证据面**：单一实验的结论写「在本实验中/在此设置下」，
  不写「所有架构/总是/根本上/证明了」。

## 本仓库的写法映射（通用规范 → MkDocs 落地）

| 原稿元素 | 本仓库写法 |
|---|---|
| 行内/显示公式 | MathJax：`$...$` / `$$...$$`（站点已配 arithmatex），不用 unicode 伪公式 |
| 提示/坑/旁注 | Material admonition：`!!! note "标题"` |
| 折叠问答 FAQ | `???+ question "问题"`（首条展开）/ `??? question`，正文缩进 4 空格 |
| 普通表格 | markdown 表格（Material 自带样式与横向滚动） |
| 特殊排版表（如 Butcher 表） | HTML table + 笔记前缀类名 |
| 交互 demo | `<figure class="<前缀>-panel">` 包 controls + canvas + stats + figcaption |
| 引用上标 | `<sup>[[n]](#refs)</sup>` |

## 交互 demo 规则（canvas js）

- **单一浅色主题**：站点不提供暗色，demo 一律白底 + 深色线条；
  移除来源页的一切 `data-theme` / `prefers-color-scheme` 监听。
- **命名空间隔离**：每篇笔记一个前缀（如 `fms-`）。CSS 类、CSS 变量（`--fms-*`）、
  动态创建的 DOM id 全部带前缀，追加进 `docs/stylesheets/extra.css` 底部并注明来源；
  **绝不写裸元素选择器**（`table{}` / `nav{}`）污染全局。
- **数据语义色**（如五种求解器五色）允许出现在 demo 图内——它们是内容不是站点色；
  站点 chrome 仍守极简近单色。
- **加载模式**（仓库惯例，见 `docs/javascripts/fourier-dct.js`）：
  ```js
  (function () {
    function run() {
      if (!document.getElementById("主canvas id")) return;  // 存在性守卫
      /* ... 全部模块，各自 try/catch 隔离 ... */
    }
    if (typeof document$ !== "undefined") document$.subscribe(run);
    else if (document.readyState !== "loading") run();
    else document.addEventListener("DOMContentLoaded", run);
  })();
  ```
- tooltip 等页面级浮层由脚本按需创建（不要求 md 里预置 div）。
- hi-DPI canvas 用 `devicePixelRatio` 缩放，缓存设计高度防复利放大
  （参考 `fm-solvers.js` 的 `prep()` 注释）。
- 移植已有 demo 时：脚本里的实验数据块（如 `REAL={...}`）**逐字节保留**，
  移植后 diff 数据块确认未被改动。

## Quick check（交稿前）

- [ ] 每个数字回查过材料，语境一致；数据块 diff 过
- [ ] 预备知识三元组逐条自查，正文用词一致
- [ ] 每个猜想恰好一次、带标签、一句话以内；教训/总结段无未标注解释
- [ ] 每个文献论断有 [n] 且 References 完整
- [ ] 每图有编号 + 实质图注
- [ ] 章节与契约部件一一对应，无模板填充章节
- [ ] 分类判据核对过；文件名 kebab-case
- [ ] nav + nav_translations + extra_javascript 三件套齐
- [ ] demo：存在性守卫、命名空间前缀、无暗色残留、id 与页面逐一比对过
- [ ] `mkdocs build --strict` 零 WARNING，zh/en 两份页面都生成

## Common Mistakes

- 把草稿的猜想写成「我们的分析表明…」并编出机制细节 → 一句话 + 标签。
- 对比表里给材料未含的格子填「中等/未测」→ 写「材料未含」。
- 概念讲解（哪怕教科书内容）不挂引用 → 预备知识同样要引。
- 用「改进方向」等模板章节填充篇幅 → 章节清单以草稿为准。
- demo 样式用裸选择器或通用类名 → 与站点/其它笔记冲突（本站踩过 `.en` 被语言切换误隐藏的坑）。
- 只加 nav 忘了 nav_translations → 英文站显示中文标题还不报错。
- 转换公式时「顺手改写」数值或不等式方向 → 公式与数字同受保真规则约束。
