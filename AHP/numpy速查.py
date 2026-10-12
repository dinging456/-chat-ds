# -*- coding: utf-8 -*-
"""
把 ahp_core.py 里出现过的 numpy 操作，一个一个拆开演示。
运行：python numpy速查.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)   # 只影响打印，不影响计算

A = np.array([[1,   2,   5],
              [0.5, 1,   3],
              [0.2, 1/3, 1]])

print("===== 0. np.array：把列表变成矩阵 =====")
print("A =")
print(A)
print("类型是", type(A))
print()

print("===== 1. A.shape：矩阵是几行几列 =====")
print("A.shape =", A.shape)          # (行数, 列数)
print("A.shape[0] =", A.shape[0], "  A.shape[1] =", A.shape[1])
print()

print("===== 2. 取值：A[行, 列] =====")
print("A[0, 1] =", A[0, 1], "  <- 第0行第1列")
print("A[1, :] =", A[1, :], "  <- 第1行全部（冒号表示“全部”）")
print("A[:, 1] =", A[:, 1], "  <- 第1列全部")
print()

print("===== 3. sum：求和，axis 决定“顺着哪个方向加” =====")
print("A.sum() =", A.sum(), "  <- 全部9个数字加起来")
print("A.sum(axis=0) =", A.sum(axis=0), "  <- 竖着加，得到每一列的和")
print("A.sum(axis=1) =", A.sum(axis=1), "  <- 横着加，得到每一行的和")
print()

print("===== 4. 矩阵 除以 一行数：自动“每一行都除一遍” =====")
col = A.sum(axis=0)
print("col = A.sum(axis=0) =", col)
print("A / col =")
print(A / col)
print("（这就是“列归一化”，numpy 会自动把每一列都除以对应的那个数）")
print()

print("===== 5. np.linalg.eig：求特征值和特征向量 =====")
vals, vecs = np.linalg.eig(A)
print("vals（特征值）=", vals)
print("vecs（特征向量，每一列是一个特征向量）=")
print(vecs)
print()

print("===== 6. np.argmax：找出最大数在第几个位置 =====")
print("vals.real =", vals.real, "   （取实部，因为结果可能是复数）")
print("np.argmax(vals.real) =", np.argmax(vals.real), "  <- 最大值的下标是0")
print()

print("===== 7. 取出最大特征值 和 它对应的特征向量 =====")
idx = np.argmax(vals.real)
lam = vals.real[idx]
w = vecs[:, idx].real        # 取第 idx 列，再取实部
print("lam =", round(lam, 4))
print("w   =", w)
print("w / w.sum() =", w / w.sum(), "  <- 归一化，这就是权重")
print()

print("===== 8. @ 矩阵乘法 =====")
W = np.array([[0.5954, 0.4434, 0.5396, 0.6267],
              [0.2764, 0.3874, 0.2970, 0.2797],
              [0.1283, 0.1692, 0.1634, 0.0936]])
criteria = np.array([0.2910, 0.5022, 0.0648, 0.1420])
print("W 是 3行4列（3个方案 × 4个指标），criteria 是 4个数")
print("W @ criteria =", W @ criteria, "  <- 一步算出三个方案的最终得分")
print()

print("===== 9. 其他小工具 =====")
print("np.zeros((3, 4)) =")
print(np.zeros((3, 4)))
print("np.round([0.12345, 0.98765], 2) =", np.round([0.12345, 0.98765], 2))
print("sum([1, 2, 3]) =", sum([1, 2, 3]), "  <- 这是 Python 自带的，不是 numpy")
