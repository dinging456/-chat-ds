# -*- coding: utf-8 -*-
"""
层次分析法（AHP）纯 Python 版
不 import 任何东西，只用列表和 for 循环，每一行都看得懂。
运行：python ahp_纯python版.py
"""


def col_sum(A):
    """每一列求和，返回 [第1列的和, 第2列的和, 第3列的和]"""
    n = len(A)                 # len(A) = 矩阵有几行
    sums = []                  # 先准备一个空列表，用来装结果
    for j in range(n):         # j 是"列号"，从 0 到 n-1
        s = 0                  # 每一列开始时，和先归零
        for i in range(n):     # i 是"行号"
            s = s + A[i][j]    # A[i][j] 就是"第 i 行、第 j 列"的数字
        sums.append(s)         # 这一列算完了，塞进结果列表
    return sums


def arithmetic_weights(A):
    """算术平均法求权重：先列归一化，再每行求平均"""
    n = len(A)
    sums = col_sum(A)
    # ---- 第一步：列归一化。每个数字除以它所在列的和 ----
    stand = []                     # stand 装归一化以后的矩阵
    for i in range(n):
        row = []                   # 先准备一个空行
        for j in range(n):
            row.append(A[i][j] / sums[j])
        stand.append(row)          # 一行填满了，塞进大列表
    # ---- 第二步：每一行求平均，得到权重 ----
    w = []
    for i in range(n):
        total = 0
        for j in range(n):
            total = total + stand[i][j]
        w.append(total / n)
    return w, stand


def eigen_weights(A, steps=50, verbose=False):
    """
    幂法求最大特征值和特征向量（替代 numpy 的 eig）
    思路：先随便猜一组权重，然后用矩阵去"乘"它，再归一化，反复几十次，
         结果就会稳定下来，稳定的那组数就是权重，每一步的"放大倍数"就是最大特征值。
    """
    n = len(A)
    w = [1 / n] * n            # 第一次的猜测：大家一样重要
    for step in range(steps):
        new = []
        for i in range(n):
            s = 0
            for j in range(n):
                s = s + A[i][j] * w[j]
            new.append(s)
        total = sum(new)       # sum(列表) 就是把这个列表里的数全加起来
        w = [x / total for x in new]   # 归一化，让新权重的和等于 1
        if verbose and step < 3:
            print(f"    第 {step + 1} 次迭代后的权重："
                  f"[{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]")
    # 收敛之后，把 A 乘 w 的结果全部加起来，就是最大特征值
    lam = 0
    for i in range(n):
        for j in range(n):
            lam = lam + A[i][j] * w[j]
    return w, lam


def consistency(A, lam):
    """一致性检验：返回 CI 和 CR"""
    n = len(A)
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
          6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    CI = (lam - n) / (n - 1)
    CR = CI / RI[n]
    return CI, CR


def analyze(name, A):
    """把一张判断矩阵从头到尾算一遍，并把结果打印出来"""
    print("=" * 64)
    print(f"【{name}】")
    print("=" * 64)
    print("判断矩阵（外层是中括号里的一行行，内层是每行的数字）：")
    for row in A:
        print("   ", [round(x, 4) for x in row])
    n = len(A)
    print(f"这是 {n} 行 {n} 列，所以 n = {n}，只需要你填 "
          f"{n * (n - 1) // 2} 个数字（右上角），其余是倒数。")

    print("\n第一步：每一列求和 ->", col_sum(A))
    w1, stand = arithmetic_weights(A)
    print("第二步：列归一化后的矩阵：")
    for row in stand:
        print("   ", [round(x, 4) for x in row])
    print("第三步：每行求平均，算术平均法权重 =",
          [round(x, 4) for x in w1])

    print("\n特征值法（幂法，不用 numpy）：")
    w2, lam = eigen_weights(A, verbose=True)
    print(f"    收敛后的权重 = [{w2[0]:.4f}, {w2[1]:.4f}, {w2[2]:.4f}]")
    print(f"    最大特征值 lambda_max = {lam:.4f}")

    CI, CR = consistency(A, lam)
    print(f"    CI = ({lam:.4f} - {n}) / ({n} - 1) = {CI:.4f}")
    print(f"    CR = {CI:.4f} / RI = {CR:.4f}   ->   "
          f"{'通过（CR < 0.1）' if CR < 0.1 else '不通过，需要修改矩阵'}")
    print()
    return w2, lam, CR


# ============================================================
# 例子一：完全一致的矩阵，权重本该是 0.6 / 0.3 / 0.1
# ============================================================
print("\n############ 例子一：一个完全一致的矩阵 ############")
A_good = [
    [1,     2,     6],
    [1 / 2, 1,     3],
    [1 / 6, 1 / 3, 1],
]
analyze("完全一致的矩阵", A_good)

# ============================================================
# 例子二：把 a13 从 6 改成 4，制造一点矛盾
# ============================================================
print("############ 例子二：改一个数字，制造矛盾 ############")
A_bad = [
    [1,     2,     4],
    [1 / 2, 1,     3],
    [1 / 4, 1 / 3, 1],
]
analyze("有矛盾的矩阵", A_bad)

# ============================================================
# 例子三：选餐厅（价格、口味、距离）
# ============================================================
print("############ 例子三：选餐厅 ############")
A_restaurant = [
    [1,     2,   5],
    [1 / 2, 1,   3],
    [1 / 5, 1 / 3, 1],
]
w, lam, CR = analyze("餐厅判断矩阵", A_restaurant)
print("结论：价格占 {:.1%}，口味占 {:.1%}，距离占 {:.1%}".format(
    w[0], w[1], w[2]))

# ============================================================
# 例子四：演唱会例子的准则层（4 个指标）
# ============================================================
print("\n############ 例子四：演唱会准则层（4 个指标）############")
A_criteria = [
    [1,     1 / 2, 4, 3],
    [2,     1,     5, 5],
    [1 / 4, 1 / 5, 1, 1 / 4],
    [1 / 3, 1 / 5, 4, 1],
]
names = ["景色", "花费", "饮食", "男女比例"]
print("=" * 64)
print("【演唱会准则层】")
print("=" * 64)
for i, row in enumerate(A_criteria):
    print(f"   {names[i]:<6}", [round(x, 4) for x in row])
w4, lam4 = eigen_weights(A_criteria)
CI4, CR4 = consistency(A_criteria, lam4)
print(f"\n  n = 4，RI = 0.90")
print(f"  最大特征值 lambda_max = {lam4:.4f}")
print(f"  CI = ({lam4:.4f} - 4) / 3 = {CI4:.4f}")
print(f"  CR = {CR4:.4f}  ->  "
      f"{'通过（CR < 0.1）' if CR4 < 0.1 else '不通过'}")
print("\n  四个指标的权重：")
for name, x in zip(names, w4):
    print(f"    {name:<6}{x:.4f}   （占 {x:.1%}）")
