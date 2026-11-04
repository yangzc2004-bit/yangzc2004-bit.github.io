---
layout: post
title: "从位置编码到 RoFormer：让 Transformer 理解顺序"
description: "从正弦位置编码的直觉出发，梳理相对位置信息，以及 RoPE 如何用旋转把位置融入注意力。"
tags: [机器学习, Transformer, 位置编码]
math: true
---

在这里写开头：读完《Attention Is All You Need》后，哪个问题最吸引你？这篇笔记想解决什么？

## 为什么 Transformer 需要位置信息

记录自注意力本身为什么无法区分词序，以及你最喜欢的直觉解释。

## 正弦位置编码在做什么

在这里写下你的理解，可以配合公式：

$$PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

## 从绝对位置走向相对位置

记录两种位置视角的差别，以及为什么相对位置对注意力有用。

## RoPE：用旋转表达相对位置

在这里整理旋转矩阵、复数视角或几何直觉。

## RoFormer 如何落地

记录 RoPE 如何作用于 Query 和 Key，以及它如何进入注意力计算。

## 小结

用自己的话回答：这一整条演进路线真正想解决的是什么？
