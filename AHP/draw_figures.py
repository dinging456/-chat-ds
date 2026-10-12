# -*- coding: utf-8 -*-
"""
生成 AHP 论文常用的两张插图：
  fig1_hierarchy.png  层次结构图（目标层 - 准则层 - 方案层）
  fig2_score.png      方案最终得分对比图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, text, w=2.0, h=0.85, fc="#E8F1FB", ec="#3C6E9F"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.06",
                                linewidth=1.2, facecolor=fc, edgecolor=ec))
    ax.text(x, y, text, ha="center", va="center", fontsize=11, color="#1F3B57")


def hierarchy_figure(path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # 层标签
    for y, label in [(8.2, "目标层"), (5.6, "准则层"), (2.6, "方案层")]:
        ax.text(0.35, y, label, fontsize=11, color="#7A8B99", rotation=90,
                ha="center", va="center")
    for y in (8.2, 5.6, 2.6):
        ax.plot([0.75, 9.7], [y, y], linestyle=":", color="#C9D6E2", linewidth=1)

    # 三层节点
    box(ax, 5.0, 8.2, "演唱会地点选择", w=3.2, fc="#D6E6F7")
    criteria = [(1.7, "景色"), (3.8, "花费"), (6.2, "饮食"), (8.3, "男女比例")]
    for x, name in criteria:
        box(ax, x, 5.6, name, w=1.7)
        ax.plot([5.0, x], [7.75, 6.05], color="#9BB6CE", linewidth=1)

    schemes = [(2.0, "苏杭"), (5.0, "北戴河"), (8.0, "桂林")]
    for x, name in schemes:
        box(ax, x, 2.6, name, w=1.9, fc="#E9F6EC", ec="#4F8F5E")
    for xc, _ in criteria:
        for xs, _ in schemes:
            ax.plot([xc, xs], [5.15, 3.05], color="#D3E0EB", linewidth=0.7)
            ax.plot([xc, xs], [5.15, 3.05], color="#C3D6C8", linewidth=0.7,
                    linestyle=(0, (2, 3)))

    ax.set_title("图 1  演唱会地点选择的层次结构模型", fontsize=13, pad=4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def score_figure(path):
    names = ["苏杭", "北戴河", "桂林"]
    scores = [0.5199, 0.3339, 0.1462]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    bars = ax.bar(names, scores, width=0.55,
                  color=["#4E79A7", "#F28E2B", "#59A14F"])
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.012, f"{s:.4f}",
                ha="center", fontsize=11)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("综合权重（最终得分）")
    ax.set_title("图 2  各备选方案的综合评价得分", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    hierarchy_figure(os.path.join(here, "fig1_hierarchy.png"))
    score_figure(os.path.join(here, "fig2_score.png"))
    print("图片已生成：fig1_hierarchy.png, fig2_score.png")
