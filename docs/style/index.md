# 样式模板 / Style Guide

这一页**枚举所有常用格式**，作为统一风格的参照。请逐节看，告诉我哪一项要调整（颜色 / 间距 / 字号 / 圆角等），我据此改 `docs/stylesheets/extra.css`，两个站点同步统一。

---

## 1. 标题层级

页面顶部的 `#` 是 H1（每页一个）。下面是 H2–H4：

## H2 二级标题（带下边线）
### H3 三级标题
#### H4 四级标题

---

## 2. 正文与行内元素

这是一段正文，用来看**行高**与**中英混排**的观感（measure / line-height）。可以包含 **加粗**、*斜体*、`行内代码`、[超链接](https://yufengjin.github.io/)、~~删除线~~、==高亮==、上标 H^^2^^O、脚注[^1]，以及键盘键 ++ctrl+c++ / ++cmd+v++。

缩写示例（悬停看提示）：HTML 与 CSS 是网页基础。

*[HTML]: HyperText Markup Language
*[CSS]: Cascading Style Sheets

[^1]: 这是一个脚注，渲染在页面底部。

---

## 3. 列表

**无序列表（含嵌套）**

- 第一项
- 第二项
    - 子项 A
    - 子项 B
- 第三项

**有序列表**

1. 步骤一
2. 步骤二
3. 步骤三

**任务清单**

- [x] 已完成项
- [ ] 待办项
- [ ] 另一个待办

**定义列表**

术语
:   术语的定义说明文字。

另一个术语
:   它的定义说明文字。

---

## 4. 引用

> 单层引用：用于摘录原文或强调一句话。石墨灰左边线。

> 带出处的引用。
>
> — 某位作者，《某书》

> 嵌套引用：
>
> > 内层引用。

---

## 5. 表格

| 方法 | 正则 | 闭式解 | 备注 |
|---|:---:|:---:|---|
| OLS | 无 | ✅ | 低维、无共线性 |
| Ridge | L2 | ✅ | 防过拟合 |
| Lasso | L1 | ❌ | 稀疏特征选择 |

---

## 6. 代码

行内：`pip install mkdocs-material`。

带语言高亮 + 标题 + 复制按钮：

```python title="ridge.py"
import numpy as np

def ridge(X, y, lam=1.0):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)
```

带行号 + 高亮特定行：

```python linenums="1" hl_lines="2 3"
def train(model, data):
    for x, y in data:          # 高亮
        model.step(x, y)       # 高亮
    return model
```

---

## 7. 标签页（Tabbed）

=== "Python"

    ```python
    print("hello")
    ```

=== "Rust"

    ```rust
    fn main() { println!("hello"); }
    ```

=== "说明"

    标签页适合并列展示同一内容的多种实现 / 多个角度。

---

## 8. 记录框 / 提示框（Admonitions）

这是**不同的记录格式**，请确认哪些保留、各自配色是否统一。每种都有「展开」与「可折叠」两种形态。

!!! note "笔记 note"
    一般性的补充说明。

!!! abstract "摘要 abstract"
    内容概要 / TL;DR。

!!! info "信息 info"
    中性信息提示。

!!! tip "技巧 tip"
    经验、窍门、推荐做法。

!!! success "成功 success"
    正确结论 / 通过项。

!!! question "问题 question"
    待解决的疑问 / 思考题。

!!! warning "警告 warning"
    需要注意的坑（用暖琥珀色区分主强调色）。

!!! failure "失败 failure"
    错误做法 / 不通过。

!!! danger "危险 danger"
    严重风险 / 必须避免。

!!! bug "Bug"
    已知缺陷记录。

!!! example "示例 example"
    例子演示。

!!! quote "引文 quote"
    引用型记录框。

**可折叠形态**（默认收起 / 默认展开）：

??? note "可折叠（默认收起）"
    点击标题展开。适合放冗长的推导或日志。

???+ tip "可折叠（默认展开）"
    带 `+` 默认展开。

---

## 9. 数学公式

行内 $e^{i\pi} + 1 = 0$；块级：

$$
\nabla_\theta \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} \nabla_\theta \ell\big(f_\theta(x_i), y_i\big)
$$

---

## 10. 网格卡片（Grid cards）

<div class="grid cards" markdown>

-   :material-clock-fast: __快速上手__

    ---

    一句话说明，配一个图标和「了解更多」链接。

    [:octicons-arrow-right-24: 了解更多](#)

-   :material-palette-outline: __统一风格__

    ---

    暖灰底 + 石墨灰点缀，极简。

    [:octicons-arrow-right-24: 样式](#)

</div>

---

## 11. 按钮

[默认按钮](#){ .md-button }
[主按钮（石墨灰）](#){ .md-button .md-button--primary }

---

## 12. 图片与图注

<figure markdown="span">
  ![占位图](https://dummyimage.com/720x300/f1efea/6b675f&text=figure){ width="100%" }
  <figcaption>图 1. 图注文字示例。</figcaption>
</figure>

---

## 13. 分割线

上面各节之间的 `---` 即分割线（hairline）。
