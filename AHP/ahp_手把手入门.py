# -*- coding: utf-8 -*-
"""
层次分析法（AHP）手把手入门脚本
这个脚本不做别的，就是把 AHP 的每一步都打印出来，
你可以对着屏幕一步一步核对，看懂"为什么这么算"。
运行方式：python ahp_手把手入门.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
RI = {3: 0.58, 4: 0.90, 5: 1.12}          # 随机一致性指标 n=3,4,5


def line(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def explain(text):
    """把多行说明文字缩进打印，方便阅读"""
    for s in text.strip("\n").split("\n"):
        print("   " + s if s else "")


line("第 0 步：认识判断矩阵")
explain("""
AHP 的全部原料就是一张方阵，叫"判断矩阵"。
里面的数字不是分数，而是【倍数】：
    a_ij = 3   读作：i 的重要程度是 j 的 3 倍
三条铁律：
    对角线永远是 1        （自己和自己一样重要）
    a_ji = 1 / a_ij       （反过来看，就是倒数）
    所以 3x3 矩阵你只需要回答 3 个问题，其余数字自动确定。
""")

A_good = np.array([[1,   2,   6],
                   [1/2, 1,   3],
                   [1/6, 1/3, 1]], dtype=float)
print("\n先看一个【完全一致】的判断矩阵 A_good =")
print(A_good)
explain("""
它为什么叫"完全一致"？因为三组比较互相不打架：
    1 号是 2 号的 2 倍，2 号是 3 号的 3 倍
    那么 1 号必须是 3 号的 2x3 = 6 倍，矩阵里正好就填了 6。
""")

line("第 1 步：算术平均法求权重（手算三小步）")
col_sum = A_good.sum(axis=0)
print("(1) 每一列求和：", col_sum)
stand = A_good / col_sum
print("\n(2) 列归一化（每个数字除以它所在列的和）：")
print(stand)
explain("""
注意看：三列变成了一模一样！
这就是"完全一致"的直接证据：三列都指向同一个结论 (0.6, 0.3, 0.1)。
""")
w_good = stand.sum(axis=1) / 3
print("(3) 每一行求平均，得到权重：", w_good)
print("    验证：三个权重相加 = %.4f（必须是 1）" % w_good.sum())

line("第 2 步：把矩阵改一个数字，看看会发生什么")
A_bad = A_good.copy()
A_bad[0, 2] = 4          # 本来应该是 6，只填了 4，这就是"矛盾"
A_bad[2, 0] = 1 / 4      # 互反性必须保持
print("A_bad =")
print(A_bad)
explain("""
矛盾在哪里？
    1 号是 2 号的 2 倍，2 号是 3 号的 3 倍
    那么 1 号必须是 3 号的 2x3 = 6 倍
    但 a_13 填的是 4，前后对不上，这叫"判断矩阵存在矛盾"。
""")
col_sum2 = A_bad.sum(axis=0)
stand2 = A_bad / col_sum2
print("列归一化后（三列不再一样了）：")
print(stand2)
explain("""
三列分别给出三套权重：
    第 1 列说：(0.5714, 0.2857, 0.1429)
    第 2 列说：(0.6000, 0.3000, 0.1000)
    第 3 列说：(0.5000, 0.3750, 0.1250)
三张"选票"互相打架，那就取平均，这就是"算术平均法"名字的由来。
""")
w_bad = stand2.sum(axis=1) / 3
print("算术平均法得到的权重：", w_bad)

line("第 3 步：特征值法求权重（论文里最常用）")
for name, A in [("A_good（完全一致）", A_good), ("A_bad（有矛盾）", A_bad)]:
    eig_val, eig_vec = np.linalg.eig(A)
    k = int(np.argmax(eig_val.real))
    lam = eig_val.real[k]
    w = np.abs(eig_vec[:, k].real)
    w = w / w.sum()
    n = A.shape[0]
    CI = (lam - n) / (n - 1)
    CR = CI / RI[n]
    print(f"{name}：")
    print(f"    最大特征值 lambda_max = {lam:.4f}   （矩阵 {n}x{n}，n = {n}）")
    print(f"    特征值法权重 = {w}")
    print(f"    CI = (lambda_max - n)/(n-1) = {CI:.4f}")
    print(f"    CR = CI / RI = {CR:.4f}  ->  "
          f"{'通过（CR < 0.1）' if CR < 0.1 else '不通过，需要修改矩阵'}")
    print()
explain("""
为什么用最大特征值对应的特征向量当权重？
    因为"完全一致"的矩阵有一个数学性质：最大特征值恰好等于 n，
    对应的特征向量恰好就是真实权重。现实中的矩阵略有偏差，
    最大特征值就略微大于 n，特征向量也就"近似"是权重。
于是：
    lambda_max 离 n 有多远   <==>   矩阵有多矛盾
这就是一致性检验的全部逻辑。
""")

line("第 4 步：一个完整的小例子（选餐厅）")
explain("""
场景：班级聚餐选餐厅，三个指标：价格、口味、距离。
你的两两比较结果是：
    价格 比 口味 重要 2 倍   -> a_12 = 2
    价格 比 距离 重要 5 倍   -> a_13 = 5
    口味 比 距离 重要 3 倍   -> a_23 = 3
""")
A = np.array([[1,   2,   5],
              [1/2, 1,   3],
              [1/5, 1/3, 1]], dtype=float)
print("判断矩阵 A =")
print(A)
print("\n(1) 列求和 =", A.sum(axis=0))
stand = A / A.sum(axis=0)
print("(2) 列归一化 =")
print(stand)
w = stand.sum(axis=1) / 3
print("(3) 行平均，得到指标权重 =", w)
eig_val, eig_vec = np.linalg.eig(A)
k = int(np.argmax(eig_val.real))
lam = eig_val.real[k]
w_eig = np.abs(eig_vec[:, k].real)
w_eig = w_eig / w_eig.sum()
CI = (lam - 3) / 2
CR = CI / RI[3]
print(f"\n(4) 特征值法：lambda_max = {lam:.4f}, 权重 = {w_eig}")
print(f"(5) 一致性检验：CI = {CI:.4f}, CR = {CR:.4f} -> "
      f"{'通过（CR < 0.1）' if CR < 0.1 else '不通过'}")
explain(f"""
(6) 结果解读：
    价格占 {w_eig[0]:.1%}，口味占 {w_eig[1]:.1%}，距离占 {w_eig[2]:.1%}
    说明这次决策里价格最重要，距离最不重要。
    接下来只要再给三个餐厅各填一张"价格/口味/距离"的判断矩阵，
    算出每个餐厅在三项上的得分，做加权求和，就得到最终排名。
""")

line("总结：AHP 就干三件事")
explain("""
1. 把"谁比谁重要"变成一张矩阵     （两两比较 + 1-9 标度）
2. 把矩阵变成权重                 （列归一化求平均，或求特征向量）
3. 检查矩阵有没有自相矛盾         （CR < 0.1 才算合格）
对准则层、方案层各做一遍，最后加权求和，得到每个方案的最终得分。
""")
