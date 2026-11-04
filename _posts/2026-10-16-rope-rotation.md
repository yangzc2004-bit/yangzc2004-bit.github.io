---
title: "读懂 RoPE：旋转矩阵背后的位置直觉"
description: "从二维旋转、复数乘法和 Query–Key 点积出发，理解 RoPE 如何自然表达相对位置。"
tags: [机器学习]
category: 深度学习
icon: orbit
math: true
---

RoPE 的公式看起来比正弦位置编码更复杂，但它的几何直觉非常直接：**位置每向前走一步，向量就转过相应的角度。**

位置不再是贴在词向量外面的一张标签，而是变成 Query 和 Key 的方向变化。

![RoPE 将位置表示为旋转角的示意图]({{ '/assets/images/posts/rope-rotation.svg' | relative_url }})
<p class="figure-caption">不同位置对应不同旋转角，点积最终只保留角度差。</p>

## 先看二维：旋转改变方向，不改变长度

二维旋转矩阵写作：

$$R(\theta)=
\begin{pmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{pmatrix}$$

它满足两个很直观的性质：

$$R(a)R(b)=R(a+b)$$

$$R(a)^T=R(-a)$$

第一个表示连续旋转可以相加，第二个表示转置等于反向旋转。RoPE 的相对位置能力，几乎就从这两条性质里长出来。

如果用复数表示二维向量，旋转还可以写成乘以 $e^{i\theta}$。矩阵和复数是同一直觉的两种语言：都在描述平面上的转动。

## 把位置放进 Query 和 Key

设位置 $m$ 上的原始 Query 为 $q$，位置 $n$ 上的原始 Key 为 $k$。RoPE 分别施加与位置有关的旋转：

$$q_m=R(m\theta)q,\qquad k_n=R(n\theta)k$$

计算注意力匹配时：

$$
\begin{aligned}
q_m^Tk_n
&=q^TR(m\theta)^TR(n\theta)k\\
&=q^TR(-m\theta)R(n\theta)k\\
&=q^TR((n-m)\theta)k
\end{aligned}
$$

绝对位置 $m$ 和 $n$ 不再分别留在结果里，剩下的只有相对距离 $n-m$。

这就是为什么 RoPE 能同时做到两件事：施加旋转时使用绝对位置，进入注意力点积后却得到显式的相对位置关系。

## 高维怎么办：分组旋转

实际模型的向量远不只二维。RoPE 把维度两两分组，第 $i$ 组使用频率：

$$\theta_i=10000^{-2i/d}$$

于是每一组都可以看成一个转速不同的二维平面。低组索引旋转更快，适合表达细粒度位置；高组索引旋转更慢，提供较长跨度的坐标。

这与正弦位置编码使用的多频思想一脉相承，只是这里不再把波形加到输入上，而是直接改变注意力中的向量方向。

一个最小的 NumPy 形式可以写成：

```python
import numpy as np

def apply_rope(x, positions, base=10000):
    # x: [sequence_length, dimension]，dimension 必须为偶数
    d = x.shape[-1]
    inv_freq = base ** (-np.arange(0, d, 2) / d)
    angles = positions[:, None] * inv_freq[None, :]
    cos, sin = np.cos(angles), np.sin(angles)

    even, odd = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = even * cos - odd * sin
    out[:, 1::2] = even * sin + odd * cos
    return out
```

实际实现还需要处理批次、注意力头、缓存和张量布局，但数学操作就是这些正弦、余弦与交叉相乘。

## 为什么这个设计有吸引力

**相对位置直接进入点积。** 不需要在注意力分数之外再额外添加一张相对位置偏置表。

**旋转保留向量范数。** 它改变方向，不因为位置变大就机械地放大向量长度。

**不依赖逐位置学习的嵌入表。** 位置由函数生成，而不是为每个训练位置分别存一组可学习参数。

**多频旋转保留不同尺度的信息。** 邻近位置与较大距离可以由不同频率共同表达。

RoFormer 论文还讨论了随相对距离增加而衰减的依赖特性。不过这不等于对所有数据都设置了固定的硬衰减，也不代表远距离信息一定无法被使用。

## 三个容易产生的误解

第一，RoPE 不是把位置编码加到词嵌入上。它主要作用于注意力中的 Query 和 Key。

第二，RoPE 不等于无限上下文。函数可以给新位置生成旋转角，但模型能否可靠处理超出训练分布的长度，是另一个问题。

第三，RoFormer 和 RoPE 不是同义词。前者是采用 RoPE 的模型，后者是位置表示方法。

## 我会怎样记住它

如果只保留一条主线，我会记住：

> 位置决定旋转角，旋转角决定点积中的相对关系。

矩阵公式是在严格表达这件事，复数形式是在简化表达这件事，多频分组则是在把这件事扩展到高维。至于代码，只是让每一对维度完成一次正确方向的旋转。

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
