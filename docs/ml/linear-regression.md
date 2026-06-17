# 线性回归（示例页）

这是一篇**示例笔记**，用来展示统一主题下各种元素的样式：公式、代码、提示框、表格、标签页。可以直接删掉或改写。

## 模型

给定数据 $\{(x_i, y_i)\}_{i=1}^n$，线性回归拟合

$$
\hat{y} = w^\top x + b .
$$

最小二乘目标（带 L2 正则）：

$$
\min_{w}\; \frac{1}{n}\sum_{i=1}^{n}\bigl(w^\top x_i - y_i\bigr)^2 + \lambda \lVert w \rVert_2^2 .
$$

行内公式示例：当 $\lambda \to 0$ 时退化为普通最小二乘。

## 代码

```python
import numpy as np

def ridge(X, y, lam=1.0):
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)   # 闭式解
```

## 提示框

!!! note "正规方程"
    当特征维度不高时，闭式解 $w = (X^\top X + \lambda I)^{-1} X^\top y$ 简单可靠。

!!! warning "数值稳定性"
    $X^\top X$ 病态时优先用 `np.linalg.solve` 或 SVD，避免显式求逆。

## 对比表

| 方法 | 正则 | 闭式解 | 适用 |
|---|---|:---:|---|
| OLS | 无 | ✅ | 低维、无共线性 |
| Ridge | L2 | ✅ | 共线性、防过拟合 |
| Lasso | L1 | ❌ | 稀疏特征选择 |

## 标签页

=== "NumPy"

    ```python
    w = np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ y)
    ```

=== "scikit-learn"

    ```python
    from sklearn.linear_model import Ridge
    w = Ridge(alpha=lam).fit(X, y).coef_
    ```
