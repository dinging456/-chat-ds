# -*- coding: utf-8 -*-
"""
层次分析法（AHP）核心函数库
----------------------------------------------------
写论文时只需要改 main 里的判断矩阵，其余函数不用动。
输出：各矩阵的权重向量、最大特征值、CI、CR，以及层次总排序。
"""

import numpy as np

# 平均随机一致性指标 RI（Saaty 经典取值，n=1,2,...）
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
      7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51,
      12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}

# 输入：重要性倍数矩阵A
# 输出： np矩阵 w, 列归一化，行求平均之后的结果 n*1
def arithmetic_mean_weights(A):
    """算术平均法（和积法）求权重：列归一化后按行求平均。"""
    A = np.asarray(A, dtype=float)  # 二维数组 -> np矩阵对象
    # shape 返回（行数，列数），A.shape[0] 就是行数
    n = A.shape[0]  # 输出几行几列
    # axis=0 就是让"第 0 维（行）消失"
    col_sum = A.sum(axis=0)          # 每一列求和(行消失)
    stand = A / col_sum              # 每个元素除以所在列的和
    w = stand.sum(axis=1) / n        # 每行求和再平均
    return w


def eigenvector_weights(A):
    """特征值法求权重：最大特征值对应的特征向量归一化。"""
    A = np.asarray(A, dtype=float)
    # val 是特征值，是个向量：[3.0037, -0.0018+0.1053j, -0.0018-0.1053j]
    # vec 是特征向量，是个矩阵，每一列对应一个特征向量
    # A * vec = val * vec
    eig_val, eig_vec = np.linalg.eig(A)
    idx = np.argmax(eig_val.real)    # 最大特征值的位置,返回最大特征值的下标
    lam_max = eig_val.real[idx]      # 用下标取最大特征值 lambda_max
    w = eig_vec[:, idx].real         # 对应特征向量，取列，逗号前面是行后面是列
    w = w / w.sum()                  # 归一化
    return w, lam_max


def consistency_check(A, lam_max):
    """一致性检验：返回 CI、CR 以及是否通过（CR < 0.1）。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n not in RI:
        raise ValueError(f"RI 表只支持 n <= 15，当前 n={n} 不在 RI 的取值范围内。")
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    CR = CI / RI[n] if RI.get(n, 0) > 0 else 0.0
    return CI, CR, CR < 0.10


# A：写了重要性倍数的矩阵（其实这里还是表格，一个二位数组）
def ahp(A):
    """一次算完：两种权重 + 一致性检验。"""
    # 算术平均法权重
    w_arith = arithmetic_mean_weights(A)
    # 特征值法权重
    w_eig, lam_max = eigenvector_weights(A)
    # 一致性检验
    CI, CR, ok = consistency_check(A, lam_max)
    return {"w_arith": w_arith, "w_eig": w_eig,
            "lam_max": lam_max, "CI": CI, "CR": CR, "pass": ok}


def total_sort(criteria_weights, scheme_matrices, scheme_names):
    """
    层次总排序：把准则层权重和方案层权重加权求和，得到方案最终得分。
    criteria_weights : 准则层权重（长度 n_criteria）
    scheme_matrices  : 每个准则对应一个判断矩阵，列表长度 n_criteria
    scheme_names     : 方案名称
    """
    W = np.zeros((len(scheme_names), len(criteria_weights)))
    for j, A in enumerate(scheme_matrices):
        W[:, j] = eigenvector_weights(A)[0]
    score = W @ np.asarray(criteria_weights, dtype=float)
    return W, score


if __name__ == "__main__":
    
    # 数据现实格式，不参与计算
    # precision=4: 保留4位小数
    # suppress=True: 不使用科学计数法，用小数显示
    np.set_printoptions(precision=4, suppress=True)

    # ========== 1. 准则层：景色、花费、饮食、男女比例 ==========
    A_criteria = [[1, 1/2, 4, 3],
                  [2, 1, 5, 5],
                  [1/4, 1/5, 1, 1/4],
                  [1/3, 1/5, 4, 1]]
    criteria_names = ["景色", "花费", "饮食", "男女比例"]

    print("=" * 62)
    print("【准则层】判断矩阵的一致性检验与权重")
    print("=" * 62)

    # 准则层矩阵ahp
    res = ahp(A_criteria)

    print(f"最大特征值 lambda_max = {res['lam_max']:.4f}")
    print(f"CI = {res['CI']:.4f},  CR = {res['CR']:.4f}  ->  "
          f"{'通过一致性检验' if res['pass'] else '未通过，需修改判断矩阵'}")
    print("算术平均法权重：", np.round(res["w_arith"], 4))
    print("特征值法权重  ：", np.round(res["w_eig"], 4))

    # ========== 2. 方案层：苏杭、北戴河、桂林 ==========
    scheme_names = ["苏杭", "北戴河", "桂林"]

    # 三维数组
    # 每个准则下的方案层判断矩阵
    A_scheme = [
        [[1, 2, 5], [1/2, 1, 2], [1/5, 1/2, 1]],      # 景色
        [[1, 1, 3], [1, 1, 2], [1/3, 1/2, 1]],        # 花费
        [[1, 2, 3], [1/2, 1, 2], [1/3, 1/2, 1]],      # 饮食
        [[1, 3, 5], [1/3, 1, 4], [1/5, 1/4, 1]],      # 男女比例
    ]

    print()
    print("=" * 62)
    print("【方案层】每个准则下的权重与一致性检验")
    print("=" * 62)
    print(f"{'准则':<8}{'苏杭':>9}{'北戴河':>9}{'桂林':>9}"
          f"{'CI':>9}{'CR':>9}{'lambda_max':>12}")
    
    W, score = total_sort(res["w_eig"], A_scheme, scheme_names)
    for j, name in enumerate(criteria_names):
        r = ahp(A_scheme[j])
        print(f"{name:<8}{W[0, j]:>9.4f}{W[1, j]:>9.4f}{W[2, j]:>9.4f}"
              f"{r['CI']:>9.4f}{r['CR']:>9.4f}{r['lam_max']:>12.4f}")

    # ========== 3. 层次总排序 ==========
    print()
    print("=" * 62)
    print("【层次总排序】最终得分")
    print("=" * 62)
    print(f"{'准则':<8}{'准则权重':>10}{'苏杭':>9}{'北戴河':>9}{'桂林':>9}")
    for j, name in enumerate(criteria_names):
        print(f"{name:<8}{res['w_eig'][j]:>10.4f}"
              f"{W[0, j]:>9.4f}{W[1, j]:>9.4f}{W[2, j]:>9.4f}")
    print("-" * 62)
    for i, name in enumerate(scheme_names):
        print(f"{name} 的最终得分 = {score[i]:.4f}")
    best = scheme_names[int(np.argmax(score))]
    print(f"\n结论：应选择【{best}】作为演唱会举办城市。")

    # ========== 4. 层次总排序的一致性检验 ==========
    # CR = sum_j(w_j * CI_j) / sum_j(w_j * RI_j)，本例子中方案层维数都是 3
    n_scheme = np.asarray(A_scheme[0]).shape[0]
    CI_list = np.array([ahp(A)["CI"] for A in A_scheme])
    CR_total = (CI_list @ res["w_eig"]) / (RI[n_scheme] * res["w_eig"].sum())
    print(f"层次总排序一致性比例 CR = {CR_total:.4f}  ->  "
          f"{'整体一致性可以接受' if CR_total < 0.10 else '整体一致性需修正'}")
