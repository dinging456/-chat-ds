# -*- coding: utf-8 -*-
"""
实验：故意用一个极度矛盾的判断矩阵，看看程序会不会报错、会不会停下来。
运行：python 测试_一致性不通过会怎样.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ahp_core import ahp

np.set_printoptions(precision=4, suppress=True)

# 石头剪刀布型矩阵：1号赢2号9倍，2号赢3号9倍，3号又赢1号9倍，形成循环矛盾
A_bad = [[1,     9,     1 / 9],
         [1 / 9, 1,     9],
         [9,     1 / 9, 1]]

print("判断矩阵：")
for row in A_bad:
    print("   ", [round(x, 4) for x in row])

res = ahp(A_bad)
print(f"\nlambda_max = {res['lam_max']:.4f}")
print(f"CI = {res['CI']:.4f},  CR = {res['CR']:.4f}")
print(f"结论：{'通过一致性检验' if res['pass'] else '未通过，需修改判断矩阵'}")
print("算出来的权重 =", np.round(res["w_eig"], 4))

print()
print("=" * 60)
print("关键问题：程序停下来了没有？")
print("=" * 60)
print("没有。上面的 print 执行完之后，函数正常返回，")
print("如果你在后面还写了别的计算，那些计算也会照常跑完。")
print("整个过程不会报错，只会得到一组【不可信】的权重。")

print()
print("下面故意再写一段后续计算，证明它真的会继续跑：")
A_scheme = [[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]]
w_scheme = ahp(A_scheme)["w_eig"]
score = res["w_eig"][0] * w_scheme
print("后续计算照样出结果：", np.round(score, 4), " <- 只是这些数字不可信")
