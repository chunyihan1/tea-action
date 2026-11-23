"""开放集拒识评估：未知打分（MSP / Entropy / Energy / Mahalanobis / Prototype）+ 阈值与代价。

为什么需要：
  论文要回答三个问题（RQ1~RQ3、RQ5）：
  1. 闭集模型遇到没见过的病会怎样（误诊率、置信度）；
  2. 哪个未知打分器最能分开已知/未知（AUROC / AUPR / FPR@95TPR）；
  3. 按"固定误拒率"或"代价最优"拒识后，误诊率、转人工率、期望代价如何变化。

实现要点（与设计文档 3.1~3.3 对齐）：
  - 检测级匹配复用 common.py：已知类要求同类且 IoU ≥ 0.5；未知类只要求 IoU ≥ 0.5；
  - 给 Detect 层注册 forward hook，捕获推理输出 y（[B, 4+nc, H*W]，类别通道已过 sigmoid）
    与 Detect 输入特征；对每个检测框，在 y 的 anchor 里找解码后 IoU 最大的位置，
    取该 anchor 的概率向量算 MSP / Entropy / Energy；
  - Mahalanobis / Prototype 用最后一层 Detect 输入特征在该 anchor 位置的通道向量，
    均值/协方差**只在 known/val 上拟合**，未知类不参与；
  - 阈值：τ_calib 在已知类检测框上按 --calib-ratio 取分位数（主结果）；
    τ_cost 在已知+未知数据上扫 100 个阈值使 C_mis·误诊数 + C_ref·复核数 最小。
    **测试集调阈值只用于分析，论文主结果必须用 τ_calib**；
  - hook 因 Ultralytics 版本差异失败时优雅降级：只保留 MSP（取检测框 conf），
    并在 metrics.json / metrics.md 里写明 scorer_unavailable 原因，不崩溃；
  - 指标全部自行实现（AUROC 用秩和、AP 用 all-point 插值、FPR@TPR 用分位数），
    不新增 sklearn 依赖；
  - 选择性预测：框级标准 AURC 与 E-AURC（相对完美排序基线的超出量，完美分离=0）、
    coverage@5% / coverage@10%，以及按图像聚合的两套 AURC（图内任一未知框未被拒识即算该图错误）；
  - 代价比扫描：--cost-ratios 对每个 C_mis:C_ref 复用代价最优阈值扫描，
    输出 cost_ratio.csv 与 decision_curve.png（对数横轴 + 推荐工作点竖线）。

用法：
  python training\\research\\open_set_eval.py --model data\\models\\tea_yolo.pt ^
      --split-dir research_out\\open_set\\S1
  python training\\research\\open_set_eval.py --model ... --split-dir ... ^
      --scorer msp,energy,proto --calib-ratio 0.05 --cost-mis 10 --cost-ref 1
  python training\\research\\open_set_eval.py --model ... --split-dir ... --conf 0.1 --device cpu
  python training\\research\\open_set_eval.py --model ... --split-dir ... ^
      --cost-ratios 1,5,10,50,100 --recommend-ratio 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Windows 下 numpy/torch/opencv 的 OpenMP 运行时冲突防护（OMP: Error #15）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from common import (get_preds, iou_xyxy, load_data_yaml, match_image,  # noqa: E402
                    read_gt, require_weights, resolve_split)

# matplotlib 无显示环境：只保存图片不弹窗（与 training/train_yolo.py 一致）
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SCORERS = ("msp", "entropy", "energy", "mahalanobis", "proto", "fusion")
FEATURE_SCORERS = ("mahalanobis", "proto")
FUSION_INPUTS = ("energy", "mahalanobis", "proto")
IOU_MATCH = 0.5
EPS = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="开放集拒识评估：未知打分 + 阈值校准 + 代价分析")
    parser.add_argument("--model", required=True, help="模型权重路径（.pt）")
    parser.add_argument("--split-dir", required=True,
                        help="开放集划分目录，如 research_out/open_set/S1")
    parser.add_argument("--scorer", default=",".join(SCORERS),
                        help="逗号列表：msp,entropy,energy,mahalanobis,proto,fusion（默认全部）")
    parser.add_argument("--calib-ratio", type=float, default=0.05,
                        help="已知类允许的误拒率，默认 0.05")
    parser.add_argument("--cost-mis", type=float, default=10.0, help="误诊代价 C_mis，默认 10")
    parser.add_argument("--cost-ref", type=float, default=1.0, help="转人工代价 C_ref，默认 1")
    parser.add_argument("--cost-ratios", default="1,2,5,10,20,50,100",
                        help="代价比扫描列表 C_mis:C_ref（C_ref 固定为 1），默认 1,2,5,10,20,50,100")
    parser.add_argument("--recommend-ratio", type=float, default=10.0,
                        help="决策曲线推荐工作点的代价比，默认 10；若不在列表中会自动追加")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值，默认 0.25")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率，默认 640")
    parser.add_argument("--device", default="auto", help="auto / cpu / 0，默认 auto")
    parser.add_argument("--out", default=None, help="输出目录，默认 <split-dir>/eval")
    return parser.parse_args(argv)


def canon(name: str) -> str:
    """类别短名归一化：复用划分脚本的 canonical_class（内部已做 try/except 兜底）。"""
    from make_open_set_split import canon as _canon

    return _canon(name)


# ============================================================
# 指标（纯 numpy 实现，便于单测）
# ============================================================
def _arr(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64).reshape(-1)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列），用于 Mann-Whitney 形式的 AUROC。"""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(known_scores, unknown_scores) -> float:
    """已知 vs 未知的 AUROC，**分数越大越像已知类**（内部用未知度取反后传入）。

    等价于 P(s_known > s_unknown) + 0.5·P(相等)，用秩和公式实现，复杂度 O(n log n)。
    """
    known = _arr(known_scores)
    unknown = _arr(unknown_scores)
    if known.size == 0 or unknown.size == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([known, unknown]))
    n_pos = unknown.size
    rank_sum = ranks[:known.size].sum()
    auc = (rank_sum - known.size * (known.size + 1) / 2.0) / (n_pos * known.size)
    return float(auc)


def auprc(known_scores, unknown_scores) -> float:
    """以未知为正类的平均精度 AP（all-point 插值），分数越大越像已知。

    即按分数升序（越靠前越像未知）累积 TP/FP；完美可分时 AP=1。
    未知占比很低时比 AUROC 更敏感，论文要求必须报告。
    """
    known = _arr(known_scores)
    unknown = _arr(unknown_scores)
    if known.size == 0 or unknown.size == 0:
        return float("nan")
    scores = np.concatenate([known, unknown])
    labels = np.concatenate([np.zeros(known.size), np.ones(unknown.size)])
    order = np.argsort(scores, kind="mergesort")   # 低分 = 更像未知，排前面
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    recall = tp / unknown.size
    precision = tp / np.maximum(tp + fp, EPS)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def fpr_at_tpr(known_scores, unknown_scores, tpr: float = 0.95) -> float:
    """把未知检出率（TPR）固定为 tpr，返回此时已知类被判为未知的比例（FPR）。

    分数越大越像已知：未知被判为未知 = 分数低于 τ，因此 τ 取未知分数的 tpr 分位，
    FPR = 已知分数 < τ 的比例。
    """
    known = _arr(known_scores)
    unknown = _arr(unknown_scores)
    if known.size == 0 or unknown.size == 0:
        return float("nan")
    tau = float(np.quantile(unknown, tpr))
    return float(np.mean(known < tau))


def threshold_at_fpr(known_scores, ratio: float) -> float:
    """按已知类允许误拒率 ratio 取分位数阈值（分数越大越未知，τ 以上判为拒识）。"""
    known = _arr(known_scores)
    if known.size == 0:
        return float("nan")
    ratio = min(max(float(ratio), 0.0), 1.0)
    return float(np.quantile(known, 1.0 - ratio))


def expected_cost(n_misdiagnosed: int, n_referred: int,
                  cost_mis: float, cost_ref: float) -> float:
    """期望代价 C_mis·误诊数 + C_ref·复核数（论文部署视角的唯一定义）。"""
    return float(cost_mis) * float(n_misdiagnosed) + float(cost_ref) * float(n_referred)


# ============================================================
# 选择性预测（AURC / coverage@risk，纯函数）
# ============================================================
def _selective_curve(scores, errors, tie_aware: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """风险-覆盖率折线点（含起点 (0, 0)）。

    方向约定：scores **越大越像已知**（越应先被覆盖）；调用方若手里是未知度，先取负。
      - tie_aware=True：同分样本视为一个并列组，组内按随机顺序取期望——
        组内第 j 位的累计错误数 = 组前累计 + 组内错误数·j/组大小。
        这样结果不依赖输入先后顺序，且完全不可分时风险恒为未知占比。
      - tie_aware=False：同分时把错误（未知框）排在后面，得到"完美排序"基线。
    """
    s = _arr(scores)
    e = _arr(errors)
    n = s.size
    if n == 0 or e.size != n:
        return np.zeros(1), np.zeros(1)
    order = (np.argsort(-s, kind="mergesort") if tie_aware
             else np.lexsort((e, -s)))          # 主键 -s 升序 = s 降序；次键错误升序
    s_sorted, e_sorted = s[order], e[order]

    coverage = [0.0]
    risk = [0.0]
    cum_err = 0.0
    k = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        group_size = j - i + 1
        group_err = float(e_sorted[i:j + 1].sum())
        for step in range(1, group_size + 1):
            k += 1
            if tie_aware:
                # 并列组内取期望：累计错误按组内均匀分布折算
                errors_so_far = cum_err + group_err * step / group_size
            else:
                errors_so_far = cum_err + float(e_sorted[i + step - 1])
            coverage.append(k / n)
            risk.append(errors_so_far / k)
        cum_err += group_err
        i = j + 1
    return np.asarray(coverage), np.asarray(risk)


def _curve_area(coverage: np.ndarray, risk: np.ndarray) -> float:
    """梯形法求覆盖率 0→1 的曲线下面积（区间长度已归一化为 1）。"""
    if coverage.size < 2:
        return 0.0
    dx = coverage[1:] - coverage[:-1]
    return float(np.sum(dx * (risk[1:] + risk[:-1]) / 2.0))


def aurc(scores, errors) -> float:
    """标准 AURC：风险-覆盖率曲线下面积（**分数越大越像已知**，未知度需先取负）。

    曲线：按分数降序逐个覆盖，覆盖率 c=k/N，风险 = 前 k 个累计错误数 / k；
    并列分数按组内期望处理，避免顺序任意性。

    注意：标准定义下覆盖率必然到 1，因此 AURC 不会为 0——完美分离时其值
    约等于未知样本占比（错误全被排在最后）。若需要"相对最优排序"的改进量，
    用 excess_aurc()。
    """
    s = _arr(scores)
    e = _arr(errors)
    if s.size == 0 or e.size != s.size:
        return float("nan")
    return float(_curve_area(*_selective_curve(s, e, tie_aware=True)))


def excess_aurc(scores, errors) -> float:
    """E-AURC：相对"完美排序"（错误全部排在最后）的 AURC 超出量。

    完美分离时为 0；排序越差越大，完全不可分时约为 q(1-q)（q 为未知占比）。
    该指标与标准 AURC 必须同时报告，不能只用其中一个。
    """
    s = _arr(scores)
    e = _arr(errors)
    if s.size == 0 or e.size != s.size:
        return float("nan")
    raw = _curve_area(*_selective_curve(s, e, tie_aware=True))
    oracle = _curve_area(*_selective_curve(s, e, tie_aware=False))
    return float(max(0.0, raw - oracle))


def coverage_at_risk(scores, errors, risk: float) -> float:
    """错误率不超过 risk 时的最大覆盖率；**分数越大越像已知**（未知度需先取负）。

    逐个前缀检查累计错误率，返回满足条件的最长前缀长度 / 样本数。
    """
    s = _arr(scores)
    e = _arr(errors)
    if s.size == 0 or e.size != s.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    e = e[order]
    k = np.arange(1, s.size + 1, dtype=np.float64)
    ok = np.cumsum(e) / k <= float(risk) + 1e-12
    if not ok.any():
        return 0.0
    return float(k[np.where(ok)[0].max()] / s.size)


def image_level_scores(items: list[dict], scorer: str
                       ) -> tuple[list[float], list[float]]:
    """把检测框聚合到图像级：(整图分数, 整图错误)，供图级 AURC 使用。

    同一张图多个框时，取**最可疑框**（未知度最大）代表整图：只有该框未被拒识，
    图内其余框才可能全部保留；一旦它被拒识，整图即转人工。因此
    图分数 = -max(未知度)（取负后越大越"整图安全"、越应先覆盖），
    图错误 = 该图至少含一个未知框（未拒识即为误诊）。
    """
    by_image: dict[str, list[dict]] = {}
    for item in items:
        if scorer not in item["scores"]:
            continue
        by_image.setdefault(item["image"], []).append(item)
    scores: list[float] = []
    errors: list[float] = []
    for boxes in by_image.values():
        worst = max(item["scores"][scorer] for item in boxes)
        scores.append(-float(worst))
        errors.append(1.0 if any(item["source"] == "unknown" for item in boxes) else 0.0)
    return scores, errors


def selective_metrics(known_items: list[dict], unknown_items: list[dict],
                      scorer: str) -> dict:
    """单个打分器的选择性预测指标：框级 AURC / coverage@5% / coverage@10% + 图级 AURC。"""
    known_scores = [it["scores"][scorer] for it in known_items if scorer in it["scores"]]
    unknown_scores = [it["scores"][scorer] for it in unknown_items if scorer in it["scores"]]
    # 现有打分器输出的是未知度（越大越未知），AURC 约定越大越像已知，故统一取负
    scores = [-v for v in known_scores] + [-v for v in unknown_scores]
    errors = [0.0] * len(known_scores) + [1.0] * len(unknown_scores)
    row = {
        "n_known": len(known_scores),
        "n_unknown": len(unknown_scores),
        "aurc": _round_or_none(aurc(scores, errors)),
        "e_aurc": _round_or_none(excess_aurc(scores, errors)),
        "coverage_at_5pct": _round_or_none(coverage_at_risk(scores, errors, 0.05)),
        "coverage_at_10pct": _round_or_none(coverage_at_risk(scores, errors, 0.10)),
        "n_images": 0,
        "aurc_image": None,
        "e_aurc_image": None,
    }
    image_scores, image_errors = image_level_scores(known_items + unknown_items, scorer)
    if image_scores:
        row["n_images"] = len(image_scores)
        row["aurc_image"] = _round_or_none(aurc(image_scores, image_errors))
        row["e_aurc_image"] = _round_or_none(excess_aurc(image_scores, image_errors))
    return row


def ap_from_tp(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """由"按置信度排序后的 TP 标记"计算单类 AP（VOC 2010 all-point 插值）。"""
    if n_gt <= 0 or len(tp) == 0:
        return 0.0
    order = np.argsort(-conf, kind="mergesort")
    tp = tp[order].astype(np.float64)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, EPS)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map50_95(image_records: list[dict], class_names: list[str]) -> dict:
    """用已有预测重算已知类 mAP50 / mAP50-95（不调用 model.val，避免类别索引错配）。"""
    ious = np.arange(0.5, 1.0, 0.05)
    nc = len(class_names)
    aps = np.full((nc, len(ious)), np.nan)
    counts = np.zeros(nc, dtype=int)
    for c in range(nc):
        n_gt = sum(1 for rec in image_records for g in rec["gts"] if g["cls"] == c)
        counts[c] = n_gt
        if n_gt == 0:
            continue
        flat = [(ii, p) for ii, rec in enumerate(image_records)
                for p in rec["preds"] if p["cls"] == c]
        if not flat:
            aps[c] = 0.0
            continue
        flat.sort(key=lambda item: -item[1]["conf"])
        confs = np.asarray([p["conf"] for _, p in flat], dtype=np.float64)
        for ti, thr in enumerate(ious):
            used: dict[int, set[int]] = {}
            tp = np.zeros(len(flat), dtype=bool)
            for rank, (ii, p) in enumerate(flat):
                best, best_j = 0.0, -1
                for j, g in enumerate(image_records[ii]["gts"]):
                    if g["cls"] != c or j in used.get(ii, set()):
                        continue
                    value = iou_xyxy(p["box"], g["box"])
                    if value > best:
                        best, best_j = value, j
                if best_j >= 0 and best >= thr:
                    used.setdefault(ii, set()).add(best_j)
                    tp[rank] = True
            aps[c, ti] = ap_from_tp(confs, tp, n_gt)

    valid = counts > 0
    if not valid.any():
        return {"map50": float("nan"), "map50_95": float("nan"), "per_class": {}, "n_gt": {}}
    return {
        "map50": float(np.mean(aps[valid, 0])),
        "map50_95": float(np.mean(aps[valid])),
        "per_class": {class_names[c]: round(float(np.mean(aps[c])), 4)
                      for c in range(nc) if valid[c]},
        "n_gt": {class_names[c]: int(counts[c]) for c in range(nc) if valid[c]},
    }


# ============================================================
# 类别映射
# ============================================================
def build_model_to_known(model_names: dict, known_names: list[str]) -> dict[int, int]:
    """模型类别 id -> known/data_known.yaml 的类别 id；不属于已知类的记 -1。

    模型（尤其现成 8 类权重）的类别顺序与某组 known 的重映射顺序不一定一致，
    因此必须按**短名**对齐，不能假设 id 相同。
    """
    mapping: dict[int, int] = {}
    for mid, name in sorted(model_names.items()):
        short = canon(name)
        mapping[int(mid)] = known_names.index(short) if short in known_names else -1
    return mapping


# ============================================================
# Detect 层 hook
# ============================================================
class DetectCapture:
    """捕获 Detect 的输入特征与推理输出，供未知打分使用。

    Ultralytics 不同版本的 Detect.forward 返回结构不同：
      - 8.x 推理：return (y, x)，y 为 [B, 4+nc, N]（类别通道已 sigmoid），
        x 在新版是 dict（含 one2many.scores 原始 logits），旧版是输入特征列表；
      - 训练/导出：返回 dict 或单张量。
    这里用 forward_pre_hook 拿输入特征、forward_hook 拿输出，并对结构做宽松解析；
    解析失败即把 status 置为 unavailable 并停用（后续自动降级到 MSP）。
    """

    def __init__(self) -> None:
        self.armed = False
        self.status = "ok"
        self.reason = ""
        self.queue: list[dict] = []
        self._feats = None
        self._handles: list = []
        self.head = None
        self.n_batches = 0

    def attach(self, module) -> None:
        """在**实际参与推理**的 nn.Module 上挂 hook。

        注意：Ultralytics 8.4 的 predictor.setup_model 会 deepcopy 模型，
        因此必须对 yolo.predictor.model 内的模块挂 hook，而不是对 yolo.model。
        """
        try:
            from ultralytics.nn.modules.head import Detect

            head = None
            for sub in module.modules():
                if isinstance(sub, Detect):
                    head = sub
                    break
            if head is None:
                raise RuntimeError("模型里没有 Detect 层")
            self.head = head
            self._handles.append(head.register_forward_pre_hook(self._pre))
            self._handles.append(head.register_forward_hook(self._post))
        except Exception as e:  # noqa: BLE001 - 版本差异时优雅降级
            self.status = "unavailable"
            self.reason = f"挂载 Detect hook 失败：{e}"

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles = []

    def reset(self) -> None:
        self.queue = []
        self._feats = None

    def _pre(self, module, inputs) -> None:
        if not self.armed:
            return
        try:
            feats = inputs[0] if inputs else None
            self._feats = feats if isinstance(feats, (list, tuple)) else None
        except Exception:  # noqa: BLE001
            self._feats = None

    def _fail(self, reason: str) -> None:
        self.status = "unavailable"
        self.reason = reason
        self.armed = False
        self.queue = []

    def _post(self, module, inputs, output) -> None:
        if not self.armed:
            return
        try:
            if isinstance(output, tuple) and len(output) >= 2:
                y, aux = output[0], output[1]
            else:
                y, aux = output, None
            if not hasattr(y, "detach"):
                raise TypeError(f"输出类型 {type(y).__name__} 不是张量")
            if y.ndim != 3:
                raise ValueError(f"输出维度为 {y.ndim}（期望 3：B, 4+nc, N）")
            nc = int(getattr(module, "nc", 0))
            if nc and y.shape[1] != 4 + nc:
                raise ValueError(f"输出通道 {y.shape[1]} ≠ 4+nc={4 + nc}")

            feats = self._feats
            logits = None
            if isinstance(aux, dict):
                node = aux.get("one2many", aux)
                if isinstance(node, dict):
                    if feats is None and isinstance(node.get("feats"), (list, tuple)):
                        feats = node["feats"]
                    if isinstance(node.get("scores"), torch_tensor_type()):
                        logits = node["scores"]
            if feats is None or len(feats) == 0:
                raise ValueError("未捕获到 Detect 输入特征")

            last = feats[-1]
            stride = module.stride
            # 用最后一层的 stride 反推网络输入尺寸（rect 推理时输入可能不是正方形）
            stride_last = float(stride[-1].detach().cpu().item()) if len(stride) else 32.0
            net_h = int(last.shape[2] * stride_last)
            net_w = int(last.shape[3] * stride_last)
            # Ultralytics v8/v9 默认 decode 成 xywh；xyxy=True 的头才是 xyxy
            xywh = not bool(getattr(module, "xyxy", False))
            self.queue.append({
                "y": y.detach().to("cpu", torch_float_type()).numpy(),
                "logits": (logits.detach().to("cpu", torch_float_type()).numpy()
                           if logits is not None else None),
                "feat": last.detach().to("cpu", torch_float_type()).numpy(),
                "bs": int(y.shape[0]),
                "left": int(y.shape[0]),
                "net": (net_w, net_h),
                "xywh": xywh,
            })
            self.n_batches += 1
        except Exception as e:  # noqa: BLE001 - 一次解析失败即全局降级
            self._fail(f"解析 Detect 输出失败：{e}")

    def consume(self):
        """按结果顺序取 (bundle, 批内下标)；没有捕获或已降级时返回 (None, -1)。"""
        if self.status != "ok":
            return None, -1
        while self.queue:
            bundle = self.queue[0]
            if bundle["left"] <= 0:
                self.queue.pop(0)
                continue
            index = bundle["bs"] - bundle["left"]
            bundle["left"] -= 1
            return bundle, index
        return None, -1


def torch_tensor_type():
    """延迟导入 torch.Tensor，避免脚本导入阶段强依赖 torch。"""
    import torch

    return torch.Tensor


def torch_float_type():
    import torch

    return torch.float32


def _net_transform(orig_shape: tuple[int, int], net_h: int, net_w: int):
    """原图坐标 -> letterbox 后网络输入坐标的 gain / padding（与 Ultralytics 一致）。"""
    h0, w0 = float(orig_shape[0]), float(orig_shape[1])
    gain = min(net_w / max(w0, 1.0), net_h / max(h0, 1.0))
    pad_x = (net_w - w0 * gain) / 2.0
    pad_y = (net_h - h0 * gain) / 2.0
    return gain, pad_x, pad_y


def _iou_vector(anchors: np.ndarray, box: np.ndarray) -> np.ndarray:
    """anchors [N,4] 与单个 box [4] 的 IoU，向量化实现。"""
    ix1 = np.maximum(anchors[:, 0], box[0])
    iy1 = np.maximum(anchors[:, 1], box[1])
    ix2 = np.minimum(anchors[:, 2], box[2])
    iy2 = np.minimum(anchors[:, 3], box[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(anchors[:, 2] - anchors[:, 0], 0, None) * np.clip(
        anchors[:, 3] - anchors[:, 1], 0, None)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, EPS)


def box_unknown_scores(bundle: dict, batch_idx: int, box, orig_shape
                       ) -> dict | None:
    """为单个检测框计算 MSP / Entropy / Energy，并取出用于 Mahan./Proto 的特征向量。

    返回 {"msp","entropy","energy","feature","anchor_iou"}；任何异常返回 None。
    """
    try:
        y = bundle["y"][batch_idx]              # [4+nc, N]
        raw = y[:4].T                            # [N, 4]（网络输入像素坐标）
        if bundle.get("xywh", False):
            # 解码结果是 (cx, cy, w, h)，先转 xyxy 再算 IoU
            anchors = np.stack([
                raw[:, 0] - raw[:, 2] / 2.0, raw[:, 1] - raw[:, 3] / 2.0,
                raw[:, 0] + raw[:, 2] / 2.0, raw[:, 1] + raw[:, 3] / 2.0,
            ], axis=1)
        else:
            anchors = raw
        net_w, net_h = bundle["net"]
        gain, pad_x, pad_y = _net_transform(orig_shape, net_h, net_w)
        box_net = np.asarray([
            box[0] * gain + pad_x, box[1] * gain + pad_y,
            box[2] * gain + pad_x, box[3] * gain + pad_y,
        ], dtype=np.float64)
        ious = _iou_vector(anchors, box_net)
        k = int(np.argmax(ious))

        probs = np.clip(y[4:, k].astype(np.float64), 0.0, 1.0)
        msp = float(probs.max()) if probs.size else 0.0
        entropy = float(-np.sum(probs * np.log(probs + EPS)))

        logits = bundle.get("logits")
        if logits is not None and logits.shape[-1] == y.shape[-1]:
            vector = logits[batch_idx][:, k].astype(np.float64)
        else:  # 没有原始 logits 时用逆 sigmoid 还原，数值上有轻微偏差
            p = np.clip(probs, 1e-4, 1.0 - 1e-4)
            vector = np.log(p / (1.0 - p))
        peak = float(np.max(vector))
        energy = float(-(peak + math.log(np.sum(np.exp(vector - peak)) + EPS)))

        feat = bundle["feat"][batch_idx]         # [C, H, W]
        _, fh, fw = feat.shape
        if bundle.get("xywh", False):
            cx, cy = float(y[0, k]), float(y[1, k])
        else:
            cx = (y[0, k] + y[2, k]) / 2.0
            cy = (y[1, k] + y[3, k]) / 2.0
        gx = min(fw - 1, max(0, int(round(cx / max(net_w, 1) * fw))))
        gy = min(fh - 1, max(0, int(round(cy / max(net_h, 1) * fh))))
        return {
            "msp": msp,
            "entropy": entropy,
            "energy": -energy,                   # 统一为"越大越未知"
            "energy_raw": energy,
            "feature": feat[:, gy, gx].astype(np.float64),
            "anchor_iou": float(ious[k]),
        }
    except Exception:  # noqa: BLE001 - 单框失败不应中断整轮评估
        return None


# ============================================================
# 推理：已知 / 未知验证集
# ============================================================
def _make_item(image: Path, source: str, pred: dict, gt_cls: int,
               scored: dict | None) -> dict:
    item = {
        "image": str(image),
        "source": source,
        "cls": pred.get("name", str(pred["cls"])),
        "gt_cls": int(gt_cls),
        "conf": float(pred["conf"]),
        "box": tuple(float(v) for v in pred["box"]),
        "anchor_iou": float("nan"),
        "feature": None,
        "scores": {},
    }
    if scored:
        item["anchor_iou"] = scored["anchor_iou"]
        item["feature"] = scored["feature"]
        item["scores"] = {
            "msp": 1.0 - scored["msp"],          # MSP 取反：越大越未知
            "entropy": scored["entropy"],
            "energy": scored["energy"],
        }
    else:
        # hook 不可用时的降级口径：检测框 conf 本身就是 MSP
        item["scores"] = {"msp": 1.0 - float(pred["conf"])}
    return item


def _predict_stream(model, images: list[Path], args: argparse.Namespace,
                    capture: DetectCapture):
    """逐张产出 (下标, Results, 捕获 bundle, 批内下标)；capture 关闭时后两项为 None/-1。"""
    stream = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        imgsz=args.imgsz,
        device=_device(args.device),
        verbose=False,
        stream=True,
    )
    for i, result in enumerate(stream):
        bundle, index = capture.consume()
        yield i, result, bundle, index
        if (i + 1) % 200 == 0:
            print(f"  已推理 {i + 1}/{len(images)} 张……")


def _device(value: str):
    value = (value or "").strip()
    return None if value in ("", "auto") else value


def infer_known(model, images: list[Path], labels: list[Path],
                known_names: list[str], model_to_known: dict[int, int],
                model_names: dict[int, str], capture: DetectCapture,
                args: argparse.Namespace) -> dict:
    """已知类验证集：只保留同类且 IoU ≥ 0.5 的 TP 框参与未知打分校准。"""
    items: list[dict] = []
    image_records: list[dict] = []
    failed_boxes = 0
    for i, result, bundle, b in _predict_stream(model, images, args, capture):
        h, w = result.orig_shape[:2]
        gts = read_gt(labels[i], w, h, len(known_names))
        preds = []
        for p in get_preds(result):
            preds.append({
                "cls": model_to_known.get(int(p["cls"]), -1),
                "conf": p["conf"],
                "box": p["box"],
                "name": model_names.get(int(p["cls"]), str(p["cls"])),
            })
        image_records.append({"preds": preds, "gts": gts})
        matched = match_image(preds, gts, IOU_MATCH)
        for pred, gt_index, _ in matched["tp"]:
            scored = box_unknown_scores(bundle, b, pred["box"], (h, w)) if bundle else None
            if bundle is not None and scored is None:
                failed_boxes += 1
            items.append(_make_item(images[i], "known", pred, gts[gt_index]["cls"], scored))
    return {"items": items, "images": image_records, "failed_boxes": failed_boxes}


def match_unknown(preds: list[dict], gts: list[dict], iou_thr: float = IOU_MATCH):
    """未知匹配：只要求 IoU ≥ iou_thr，不看类别；每个 GT / 预测一对一。"""
    used: set[int] = set()
    pairs: list[tuple[dict, int, float]] = []
    for gi, gt in enumerate(gts):
        best, best_pi = 0.0, -1
        for pi, pred in enumerate(preds):
            if pi in used:
                continue
            value = iou_xyxy(pred["box"], gt["box"])
            if value > best:
                best, best_pi = value, pi
        if best_pi >= 0 and best >= iou_thr:
            used.add(best_pi)
            pairs.append((preds[best_pi], gi, best))
    return pairs


def infer_unknown(model, images: list[Path], labels: list[Path],
                  model_names: dict[int, str], capture: DetectCapture,
                  args: argparse.Namespace) -> dict:
    """未知验证集：统计 GT 总数、被检测出的未知框（任意类别）及其分数。"""
    items: list[dict] = []
    n_gt = 0
    failed_boxes = 0
    for i, result, bundle, b in _predict_stream(model, images, args, capture):
        h, w = result.orig_shape[:2]
        gts = read_gt(labels[i], w, h, 1)
        n_gt += len(gts)
        preds = get_preds(result)
        for pred in preds:
            pred["name"] = model_names.get(int(pred["cls"]), str(pred["cls"]))
        for pred, gt_index, _ in match_unknown(preds, gts):
            scored = box_unknown_scores(bundle, b, pred["box"], (h, w)) if bundle else None
            if bundle is not None and scored is None:
                failed_boxes += 1
            items.append(_make_item(images[i], "unknown", pred, -1, scored))
    return {"items": items, "n_gt": n_gt, "failed_boxes": failed_boxes}


# ============================================================
# 特征分布：Mahalanobis / Prototype
# ============================================================
def fit_feature_stats(items: list[dict], n_classes: int, shrinkage: float = 1e-2
                      ) -> tuple[dict | None, str]:
    """在 known/val 的 TP 特征上拟合每类均值与共享协方差（未知类不参与）。"""
    rows = [(it["feature"], it["gt_cls"]) for it in items
            if it["feature"] is not None and it["gt_cls"] >= 0]
    if len(rows) < 2:
        return None, "known/val 可用特征不足 2 个，无法拟合特征分布"
    dim = int(rows[0][0].size)
    if dim > 1024:
        return None, f"特征维度 {dim} 过大，已跳过 Mahalanobis/Prototype"
    features = np.stack([r[0] for r in rows])
    labels = np.asarray([r[1] for r in rows], dtype=int)

    means = np.zeros((n_classes, dim), dtype=np.float64)
    valid = np.zeros(n_classes, dtype=bool)
    for c in range(n_classes):
        subset = features[labels == c]
        if subset.size == 0:
            continue
        means[c] = subset.mean(axis=0)
        valid[c] = True
    if not valid.any():
        return None, "known/val 没有任何类别的特征样本"

    cov = np.zeros((dim, dim), dtype=np.float64)
    for c in np.where(valid)[0]:
        centered = features[labels == c] - means[c]
        cov += centered.T @ centered
    cov /= max(len(features) - 1, 1)
    trace = float(np.trace(cov)) / dim
    cov += np.eye(dim) * (shrinkage * trace if trace > 0 else 1e-3)
    precision = np.linalg.pinv(cov)

    prototypes = means[valid].copy()
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.maximum(norms, EPS)

    return {
        "means": means[valid],
        "classes": np.where(valid)[0],
        "precision": precision,
        "prototypes": prototypes,
        "n_per_class": {int(c): int(np.sum(labels == c)) for c in np.where(valid)[0]},
    }, ""


def add_feature_scores(items: list[dict], stats: dict) -> None:
    """给每个 item 填 mahalanobis（最近类距离）与 proto（1 - 最大余弦相似度）。"""
    means = stats["means"]
    precision = stats["precision"]
    prototypes = stats["prototypes"]
    for item in items:
        vector = item["feature"]
        if vector is None:
            continue
        diffs = means - vector
        mahal = np.einsum("kc,cd,kd->k", diffs, precision, diffs)
        item["scores"]["mahalanobis"] = float(np.min(mahal))
        unit = vector / max(float(np.linalg.norm(vector)), EPS)
        item["scores"]["proto"] = float(1.0 - np.max(prototypes @ unit))


# ============================================================
# 融合分数（逻辑回归，numpy 实现）
# ============================================================
def fit_fusion(items: list[dict], inputs=FUSION_INPUTS, iters: int = 400,
               lr: float = 0.5, l2: float = 1e-3) -> tuple[dict | None, str]:
    """在已知+未知检测框上拟合 σ(w·z+b)，z 为各分数标准化后的向量。

    注意：融合权重与阈值共用同一批验证框，存在轻度乐观偏差；论文中应写明
    "融合权重在验证集上拟合"，并优先用 τ_calib 报告主结果。
    """
    usable = [it for it in items
              if all(k in it["scores"] for k in inputs) and it["feature"] is not None]
    if len(usable) < 10:
        return None, "可用检测框不足 10 个，无法拟合融合权重"
    raw = np.asarray([[it["scores"][k] for k in inputs] for it in usable], dtype=np.float64)
    target = np.asarray([1.0 if it["source"] == "unknown" else 0.0 for it in usable])
    if target.min() == target.max():
        return None, "融合拟合需要同时包含已知与未知检测框"
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < EPS] = 1.0
    x = (raw - mean) / std
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    n = len(x)
    for _ in range(max(1, iters)):
        z = np.clip(x @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = x.T @ (p - target) / n + l2 * w
        grad_b = float(np.mean(p - target))
        w -= lr * grad_w
        b -= lr * grad_b
    return {"inputs": list(inputs), "w": w, "b": b, "mean": mean, "std": std,
            "n": int(n)}, ""


def apply_fusion(items: list[dict], fusion: dict) -> None:
    inputs = fusion["inputs"]
    for item in items:
        if not all(k in item["scores"] for k in inputs):
            continue
        vector = np.asarray([item["scores"][k] for k in inputs], dtype=np.float64)
        z = float(((vector - fusion["mean"]) / fusion["std"]) @ fusion["w"] + fusion["b"])
        item["scores"]["fusion"] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


# ============================================================
# 阈值与拒识指标
# ============================================================
def evaluate_at_threshold(known_items: list[dict], unknown_items: list[dict],
                          n_unknown_gt: int, scorer: str, tau: float,
                          cost_mis: float, cost_ref: float) -> dict:
    """在给定 τ 下统计拒识指标；分数 > τ 判为"拒识/转人工"。"""
    known_n = len(known_items)
    unknown_n = len(unknown_items)
    known_rejected = sum(1 for it in known_items if it["scores"].get(scorer, -1.0) > tau)
    unknown_rejected = sum(1 for it in unknown_items if it["scores"].get(scorer, -1.0) > tau)
    unknown_mis = unknown_n - unknown_rejected
    total_boxes = known_n + unknown_n
    return {
        "tau": round(float(tau), 6),
        "known_boxes": known_n,
        "known_false_rejection": known_rejected,
        "known_false_rejection_rate": round(known_rejected / known_n, 4) if known_n else None,
        "unknown_gt_boxes": int(n_unknown_gt),
        "unknown_detected_boxes": unknown_n,
        "unknown_rejected": unknown_rejected,
        "unknown_recall": round(unknown_rejected / n_unknown_gt, 4) if n_unknown_gt else None,
        "unknown_recall_detected": round(unknown_rejected / unknown_n, 4) if unknown_n else None,
        "misdiagnosed_boxes": unknown_mis,
        "misdiagnosis_rate": round(unknown_mis / n_unknown_gt, 4) if n_unknown_gt else None,
        "undetected_rate": (round((n_unknown_gt - unknown_n) / n_unknown_gt, 4)
                            if n_unknown_gt else None),
        "referred_boxes": known_rejected + unknown_rejected,
        "referral_rate": (round((known_rejected + unknown_rejected) / total_boxes, 4)
                          if total_boxes else None),
        "expected_cost": round(expected_cost(unknown_mis, known_rejected + unknown_rejected,
                                             cost_mis, cost_ref), 2),
    }


def best_cost_threshold(known_items: list[dict], unknown_items: list[dict],
                        n_unknown_gt: int, scorer: str, cost_mis: float, cost_ref: float,
                        n_candidates: int = 100) -> tuple[float, float, dict]:
    """在已知+未知检测框上扫 100 个候选阈值，取期望代价最小者。

    **仅用于代价敏感分析（RQ5）**；主结果必须用 known/val 上的 τ_calib，
    否则等于在测试集上调阈值。
    """
    scores = [it["scores"].get(scorer) for it in known_items + unknown_items]
    scores = [float(s) for s in scores if s is not None and not math.isnan(float(s))]
    if not scores:
        return float("nan"), float("nan"), {}
    # 拒识条件是 score > τ，因此网格必须外扩到 [min-ε, max+ε]：
    # 否则 τ 取不到 min 以下，"分数恰好等于最小值"的未知框永远无法拒识，
    # 代价比扫描会出现各比例 τ 完全相同、转人工率不随代价变化。
    lo = float(min(scores)) - 1e-6
    hi = float(max(scores)) + 1e-6
    grid = np.linspace(lo, hi, max(2, int(n_candidates)))
    best = (float("nan"), float("inf"), {})
    for tau in grid:
        detail = evaluate_at_threshold(known_items, unknown_items, n_unknown_gt, scorer,
                                       float(tau), cost_mis, cost_ref)
        cost = detail["expected_cost"]
        if cost < best[1]:
            best = (float(tau), float(cost), detail)
    return best[0], best[1], best[2]


def scan_cost_ratios(known_items: list[dict], unknown_items: list[dict],
                     n_unknown_gt: int, scorer: str, ratios,
                     n_images: int | None = None, n_candidates: int = 100
                     ) -> list[dict]:
    """对每个 C_mis:C_ref 比例复用 best_cost_threshold，返回逐比例的最优工作点。

    C_ref 固定为 1，比例即 C_mis；预算按**图像数**归一（期望代价 / 图像总数），
    使不同规模的验证集可横向比较。比例非正数直接跳过。
    """
    rows: list[dict] = []
    for ratio in ratios:
        ratio = float(ratio)
        if not ratio > 0:
            continue
        tau, cost, detail = best_cost_threshold(
            known_items, unknown_items, n_unknown_gt, scorer, ratio, 1.0, n_candidates)
        rows.append({
            "cost_ratio": ratio,
            "tau": _round_or_none(tau),
            "misdiagnosis_rate": detail.get("misdiagnosis_rate"),
            "referral_rate": detail.get("referral_rate"),
            "expected_cost": _round_or_none(cost),
            "expected_cost_per_image": (round(float(cost) / n_images, 6)
                                        if n_images and cost == cost else None),
        })
    return rows


def scorer_summary(scorer: str, known_items: list[dict], unknown_items: list[dict],
                   n_unknown_gt: int, args: argparse.Namespace) -> dict:
    # 指标函数口径是"分数越大越像已知"，故这里取未知度的相反数；
    # 阈值（拒识）仍用原始未知度，二者不可混用。
    known_u = [it["scores"][scorer] for it in known_items if scorer in it["scores"]]
    unknown_u = [it["scores"][scorer] for it in unknown_items if scorer in it["scores"]]
    known_scores = [-v for v in known_u]
    unknown_scores = [-v for v in unknown_u]
    row = {
        "n_known": len(known_scores),
        "n_unknown": len(unknown_scores),
        "auroc": _round_or_none(auroc(known_scores, unknown_scores)),
        "auprc": _round_or_none(auprc(known_scores, unknown_scores)),
        "fpr_at_95tpr": _round_or_none(fpr_at_tpr(known_scores, unknown_scores, 0.95)),
        "tau_calib": None,
        "calib": None,
        "tau_cost": None,
        "cost": None,
        "cost_optimal": None,
    }
    if known_u:
        tau = threshold_at_fpr(known_u, args.calib_ratio)
        row["tau_calib"] = round(float(tau), 6)
        row["calib"] = evaluate_at_threshold(known_items, unknown_items, n_unknown_gt,
                                             scorer, tau, args.cost_mis, args.cost_ref)
    if known_u or unknown_u:
        tau_cost, cost, detail = best_cost_threshold(
            known_items, unknown_items, n_unknown_gt, scorer,
            args.cost_mis, args.cost_ref)
        row["tau_cost"] = _round_or_none(tau_cost)
        row["cost"] = _round_or_none(cost)
        row["cost_optimal"] = detail
    return row


def _round_or_none(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)


# ============================================================
# 输出
# ============================================================
def write_scores_csv(path: Path, items: list[dict], scorer: str,
                     tau_calib: float, tau_cost: float) -> None:
    fields = ["source", "image", "class", "gt_class", "conf", "anchor_iou",
              *SCORERS, "rejected_calib", "rejected_cost"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = {
                "source": item["source"],
                "image": Path(item["image"]).name,
                "class": item["cls"],
                "gt_class": item["gt_cls"],
                "conf": round(item["conf"], 6),
                "anchor_iou": (round(item["anchor_iou"], 4)
                               if not math.isnan(item["anchor_iou"]) else ""),
                "rejected_calib": int(item["scores"].get(scorer, -1.0) > tau_calib)
                if tau_calib == tau_calib else "",
                "rejected_cost": int(item["scores"].get(scorer, -1.0) > tau_cost)
                if tau_cost == tau_cost else "",
            }
            for key in SCORERS:
                value = item["scores"].get(key)
                row[key] = round(value, 6) if value is not None else ""
            writer.writerow(row)


def plot_risk_coverage(path: Path, known_items: list[dict], unknown_items: list[dict],
                       n_unknown_gt: int, scorer: str, tau_calib: float,
                       tau_cost: float, args: argparse.Namespace) -> None:
    """风险-覆盖率曲线：横轴转人工率，纵轴误诊率；右图期望代价随 τ 变化。"""
    scores = [it["scores"][scorer] for it in known_items + unknown_items
              if scorer in it["scores"]]
    if not scores:
        return
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0

    def norm(value: float) -> float:
        return (value - lo) / span

    grid = np.linspace(0.0, 1.0, 101)
    referral, mis, costs = [], [], []
    for t in grid:
        tau = lo + t * span
        detail = evaluate_at_threshold(known_items, unknown_items, n_unknown_gt,
                                       scorer, tau, args.cost_mis, args.cost_ref)
        referral.append(detail["referral_rate"] or 0.0)
        mis.append(detail["misdiagnosis_rate"] or 0.0)
        costs.append(detail["expected_cost"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    axes[0].plot(referral, mis, color="#1f77b4", lw=2)
    axes[0].set_xlabel("转人工率（Referred / 全部检测框）")
    axes[0].set_ylabel("误诊率（未知框仍判为已知病 / 未知 GT 框）")
    axes[0].set_title(f"风险-覆盖率曲线（{scorer}）")
    axes[0].grid(alpha=0.3)
    for tau, label, color in ((tau_calib, "calib τ", "#d62728"),
                              (tau_cost, "cost-opt τ", "#2ca02c")):
        if tau == tau:  # 跳过 NaN
            detail = evaluate_at_threshold(known_items, unknown_items, n_unknown_gt,
                                           scorer, tau, args.cost_mis, args.cost_ref)
            axes[0].scatter([detail["referral_rate"] or 0.0], [detail["misdiagnosis_rate"] or 0.0],
                            color=color, zorder=5,
                            label=f"{label}={norm(tau):.3f}（归一化）")
    axes[0].legend(fontsize=8)

    axes[1].plot(grid, costs, color="#ff7f0e", lw=2)
    axes[1].set_xlabel("τ（分数 min-max 归一化到 0~1）")
    axes[1].set_ylabel("期望代价 C_mis·误诊 + C_ref·复核")
    axes[1].set_title(f"期望代价随阈值变化（C_mis={args.cost_mis:g}，C_ref={args.cost_ref:g}）")
    axes[1].grid(alpha=0.3)
    for tau, label, color in ((tau_calib, "calib τ", "#d62728"),
                              (tau_cost, "cost-opt τ", "#2ca02c")):
        if tau == tau:
            detail = evaluate_at_threshold(known_items, unknown_items, n_unknown_gt,
                                           scorer, tau, args.cost_mis, args.cost_ref)
            axes[1].axvline(norm(tau), color=color, ls="--", lw=1)
            axes[1].scatter([norm(tau)], [detail["expected_cost"]], color=color, zorder=5,
                            label=f"{label}={norm(tau):.3f}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_cost_ratio_csv(path: Path, rows: list[dict]) -> None:
    """代价比扫描结果落盘（与 metrics.json 的 cost_ratio_sweep.rows 同源）。"""
    fields = ["cost_ratio", "tau", "misdiagnosis_rate", "referral_rate",
              "expected_cost", "expected_cost_per_image"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key))
                             for key in fields})


def plot_decision_curve(path: Path, rows: list[dict], recommend_ratio: float,
                        scorer: str) -> None:
    """决策曲线：上=代价比（对数横轴）vs 每图归一化期望代价，下=代价比 vs 转人工率。"""
    valid = [row for row in rows if row.get("expected_cost_per_image") is not None]
    if not valid:
        return
    ratios = [row["cost_ratio"] for row in valid]
    costs = [row["expected_cost_per_image"] for row in valid]
    referral = [row.get("referral_rate") or 0.0 for row in valid]

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.6), dpi=150, sharex=True)
    axes[0].plot(ratios, costs, marker="o", color="#1f77b4", lw=2)
    axes[0].set_ylabel("期望代价 / 图像数")
    axes[0].set_title(f"代价比决策曲线（{scorer}，C_ref=1）")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(ratios, referral, marker="o", color="#ff7f0e", lw=2)
    axes[1].set_xlabel("代价比 C_mis : C_ref（对数轴）")
    axes[1].set_ylabel("转人工率（Referred / 全部检测框）")
    axes[1].set_ylim(0.0, max(0.1, max(referral) * 1.15))
    axes[1].grid(alpha=0.3, which="both")

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(float(recommend_ratio), color="#d62728", ls="--", lw=1.2,
                   label=f"推荐代价比 {recommend_ratio:g}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_markdown(path: Path, report: dict) -> None:
    r = report
    data = r["data"]
    lines = [
        "# 开放集拒识评估报告",
        "",
        f"- 模型：`{r['model']}`",
        f"- 划分目录：`{r['split_dir']}`",
        f"- 已知类别（{r['n_known_classes']}）：{'、'.join(r['known_classes'])}",
        f"- 推理设置：imgsz={r['imgsz']}，conf={r['conf']}，IoU≥{r['iou']}，"
        f"device={r['device']}",
        f"- 阈值约束：已知误拒率 {r['calib_ratio']:.1%} ｜ "
        f"代价 C_mis={r['cost_mis']:g}，C_ref={r['cost_ref']:g}",
        f"- 生成时间：{r['generated_at']}",
        "",
        "## 1. 数据",
        "",
        f"- known/val：{data['known_val_images']} 张图，参与打分的已知 TP 框 {data['known_boxes']} 个",
        f"- unknown/val：{data['unknown_val_images']} 张图，未知 GT 框 {data['unknown_gt_boxes']} 个，"
        f"其中被检测出 {data['unknown_detected_boxes']} 个",
        "",
    ]
    if r["scorer_unavailable"]:
        lines += ["## 2. 打分器降级说明", ""]
        for name, reason in r["scorer_unavailable"].items():
            lines.append(f"- `{name}` 不可用：{reason}")
        lines += ["", "> 其余可用打分器照常评估；上述不可用项在表格中以 — 表示。", ""]
    else:
        lines += ["## 2. 打分器可用性", "", "全部打分器均可用。", ""]

    rq1 = r["rq1"]
    lines += [
        "## 3. RQ1 闭集模型遇到未知病害的行为",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 闭集误诊率（未知 GT 被任一类别检出 / 未知 GT） | "
        f"{_fmt(rq1['closed_set_misdiagnosis_rate'], pct=True)} |",
        f"| 未知框平均置信度 | {_fmt(rq1['unknown_mean_conf'])} |",
        f"| 已知框平均置信度 | {_fmt(rq1['known_mean_conf'])} |",
        "",
        "表注：误诊率按未知 GT 框计（未检出的未知框不计入误诊，"
        "未检出率 undetected_rate 见 metrics.json 的 rejection 字段）。",
        "",
        "## 4. RQ2 打分器可分性（与阈值无关）",
        "",
        "| 打分器 | N(known) | N(unknown) | AUROC | AUPR | FPR@95TPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in r["rq2"].items():
        lines.append(
            f"| {name} | {row['n_known']} | {row['n_unknown']} | {_fmt(row['auroc'])} | "
            f"{_fmt(row['auprc'])} | {_fmt(row['fpr_at_95tpr'])} |")
    lines += [
        "",
        "表注：三个指标均按「分数越大越像已知」计算；AUROC=P(已知分数 > 未知分数)，"
        "AUPR 以未知为正类（低分排前），FPR@95TPR 为未知检出率固定 95% 时已知类被误拒的比例。",
        "",
        "## 5. 拒识后指标（主结果用 calib τ）",
        "",
        "| 打分器 | τ_calib | 已知误拒率 | 未知召回 | 误诊率 | 转人工率 | 期望代价 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in r["rejection"].items():
        calib = row.get("calib") or {}
        lines.append(
            f"| {name} | {_fmt(row['tau_calib'])} | "
            f"{_fmt(calib.get('known_false_rejection_rate'), pct=True)} | "
            f"{_fmt(calib.get('unknown_recall'), pct=True)} | "
            f"{_fmt(calib.get('misdiagnosis_rate'), pct=True)} | "
            f"{_fmt(calib.get('referral_rate'), pct=True)} | "
            f"{_fmt(calib.get('expected_cost'))} |")
    lines += [
        "",
        "## 6. 代价最优阈值（仅分析，不作为主结果）",
        "",
        "| 打分器 | τ_cost | 已知误拒率 | 未知召回 | 误诊率 | 转人工率 | 期望代价 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in r["rejection"].items():
        opt = row.get("cost_optimal") or {}
        lines.append(
            f"| {name} | {_fmt(row['tau_cost'])} | "
            f"{_fmt(opt.get('known_false_rejection_rate'), pct=True)} | "
            f"{_fmt(opt.get('unknown_recall'), pct=True)} | "
            f"{_fmt(opt.get('misdiagnosis_rate'), pct=True)} | "
            f"{_fmt(opt.get('referral_rate'), pct=True)} | "
            f"{_fmt(opt.get('expected_cost'))} |")
    lines += [
        "",
        "> ⚠ τ_cost 在已知+未知验证数据上扫描得到，仅用于 RQ5 代价敏感性分析；"
        "论文主结果必须报告 calib τ（阈值只在已知类验证集上按固定误拒率标定）。",
        "",
    ]

    sp = r.get("selective_prediction") or {}
    sp_scorers = sp.get("scorers") or {}
    lines += [
        "## 7. 选择性预测（框级，与阈值无关）",
        "",
        "| 打分器 | N(框) | AURC | E-AURC | coverage@5% | coverage@10% | N(图) | AURC(图级) | E-AURC(图级) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sp_scorers.items():
        lines.append(
            f"| {name} | {row['n_known'] + row['n_unknown']} | {_fmt(row.get('aurc'))} | "
            f"{_fmt(row.get('e_aurc'))} | "
            f"{_fmt(row.get('coverage_at_5pct'), pct=True)} | "
            f"{_fmt(row.get('coverage_at_10pct'), pct=True)} | "
            f"{row.get('n_images') or '—'} | {_fmt(row.get('aurc_image'))} | "
            f"{_fmt(row.get('e_aurc_image'))} |")
    lines += [
        "",
        "表注：AURC 按「分数越大越像已知」计算——未知度打分器已先取负，覆盖率 0→1 时"
        "逐个覆盖并累计错误率后做梯形积分。**标准 AURC 在覆盖率到 1 时必含错误，"
        "完美分离也不为 0（其值约等于未知占比）**；E-AURC 是相对完美排序基线（错误全排"
        "最后）的超出量，完美分离才为 0。两者必须同时报告。coverage@r 表示错误率不超过"
        " r 时的最大覆盖率。",
        "",
        "图级口径：同一张图取最可疑框（未知度最大）代表整图，图内只要有一个未知框未被"
        "拒识就记该图错误；只有最可疑框低于 τ、图内全部框都被保留时该图才算被覆盖。",
        "",
        "> 口径说明：现有匹配对已知框要求**同类且 IoU ≥ 0.5**，因此已知侧的类别错误"
        "（未匹配上的 FP）不会进入这个选择集；AURC 衡量的是「保留已知框 vs 拒识未知框」"
        "的权衡，这是已知简化，论文引用该指标时必须一并说明。",
        "",
    ]

    sweep = r.get("cost_ratio_sweep") or {}
    sweep_rows = sweep.get("rows") or []
    lines += [
        f"## 8. 代价比扫描（C_ref 固定为 1，打分器 {sweep.get('scorer', '—')}）",
        "",
        "| C_mis : C_ref | τ | 误诊率 | 转人工率 | 期望代价 | 期望代价/图 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sweep_rows:
        lines.append(
            f"| {row['cost_ratio']:g} : 1 | {_fmt(row.get('tau'))} | "
            f"{_fmt(row.get('misdiagnosis_rate'), pct=True)} | "
            f"{_fmt(row.get('referral_rate'), pct=True)} | "
            f"{_fmt(row.get('expected_cost'))} | "
            f"{_fmt(row.get('expected_cost_per_image'))} |")
    rec = sweep.get("recommend") or {}
    if rec:
        lines += [
            "",
            f"推荐工作点（代价比 {rec['cost_ratio']:g} : 1，τ={_fmt(rec.get('tau'))}）："
            f"误诊率 {_fmt(rec.get('misdiagnosis_rate'), pct=True)}，"
            f"转人工率 {_fmt(rec.get('referral_rate'), pct=True)}，"
            f"期望代价/图 {_fmt(rec.get('expected_cost_per_image'))}。",
        ]
    lines += [
        "",
        "表注：代价比越大越保守（分数可分时 τ 下移、转人工率上升；"
        "若未知框与已知框分数不可分则可能保持不变）；"
        "τ 仍在已知+未知数据上扫描得到，仅用于代价敏感性分析。",
        "",
    ]

    km = r["known_map"]
    lines += [
        "## 9. 拒识前的已知类检测精度（由已有预测重算）",
        "",
        f"- 已知类 mAP@0.5：{_fmt(km['map50'])} ｜ mAP@0.5:0.95：{_fmt(km['map50_95'])}",
        "",
    ]
    if km.get("per_class"):
        lines += ["| 类别 | mAP@0.5:0.95 | GT 框数 |", "| --- | ---: | ---: |"]
        for name, value in km["per_class"].items():
            lines.append(f"| {name} | {_fmt(value)} | {km['n_gt'].get(name, 0)} |")
        lines.append("")
    if r.get("anchor_iou_median") == r.get("anchor_iou_median"):
        lines += [f"- 检测框与命中 anchor 的 IoU 中位数：{r['anchor_iou_median']:.3f}"
                  "（用于自检特征/概率提取位置是否可靠）", ""]

    lines += ["## 10. 结论", "", f"> {r['conclusion']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value, pct: bool = False) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(f):
        return "—"
    return f"{f * 100:.2f}%" if pct else f"{f:.4f}"


def build_conclusion(report: dict) -> str:
    rq2 = report["rq2"]
    rejection = report["rejection"]
    candidates = {k: v["auroc"] for k, v in rq2.items() if v.get("auroc") is not None}
    if not candidates:
        return "没有可用的 AUROC（可能已知或未知检测框为空），请检查 --conf 与 split 目录。"
    best = max(candidates, key=candidates.get)
    calib = (rejection.get(best) or {}).get("calib") or {}
    return (f"AUROC 最高的打分器为 `{best}`（AUROC={candidates[best]:.4f}）；"
            f"在 calib τ 下未知召回 {_fmt(calib.get('unknown_recall'), pct=True)}、"
            f"误诊率 {_fmt(calib.get('misdiagnosis_rate'), pct=True)}、"
            f"转人工率 {_fmt(calib.get('referral_rate'), pct=True)}、"
            f"期望代价 {_fmt(calib.get('expected_cost'))}。"
            "结论解读需结合多组划分与多 seed，避免单组偶然性。")


# ============================================================
# 主流程
# ============================================================
def _load_pair(split_dir: Path) -> dict:
    known_yaml = split_dir / "data_known.yaml"
    unknown_yaml = split_dir / "data_unknown.yaml"
    k_cfg, k_names, k_path = load_data_yaml(known_yaml)
    u_cfg, u_names, u_path = load_data_yaml(unknown_yaml)
    k_images, k_labels, _ = resolve_split(k_cfg, k_path, "val")
    u_images, u_labels, _ = resolve_split(u_cfg, u_path, "val")
    return {
        "known_names": k_names,
        "known_images": k_images,
        "known_labels": k_labels,
        "unknown_images": u_images,
        "unknown_labels": u_labels,
        "unknown_names": u_names,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.calib_ratio <= 1.0:
        print("[错误] --calib-ratio 必须在 0~1 之间")
        return 1
    if not args.recommend_ratio > 0:
        print("[错误] --recommend-ratio 必须为正数")
        return 1
    try:
        cost_ratios = [float(x) for x in re.split(r"[,，]", args.cost_ratios) if x.strip()]
    except ValueError:
        print("[错误] --cost-ratios 需要逗号分隔的正数，如 1,2,5,10")
        return 1
    if not cost_ratios or any(r <= 0 for r in cost_ratios):
        print("[错误] --cost-ratios 需要逗号分隔的正数，如 1,2,5,10")
        return 1
    if not any(abs(r - args.recommend_ratio) < 1e-9 for r in cost_ratios):
        cost_ratios.append(args.recommend_ratio)   # 保证推荐工作点也出现在扫描表里
    cost_ratios = sorted(set(cost_ratios))
    selected = [x.strip().lower() for x in args.scorer.replace("，", ",").split(",") if x.strip()]
    unknown_scorers = [s for s in selected if s not in SCORERS]
    if unknown_scorers:
        print(f"[错误] 未知打分器：{'、'.join(unknown_scorers)}；可选 {', '.join(SCORERS)}")
        return 1
    if not selected:
        print("[错误] --scorer 至少需要一个打分器")
        return 1

    split_dir = Path(args.split_dir)
    if not split_dir.is_dir():
        print(f"[错误] 划分目录不存在：{split_dir}")
        return 1
    try:
        data = _load_pair(split_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] 读取划分失败：{e}")
        print("       请确认目录中有 data_known.yaml / data_unknown.yaml，且 val 下有图片。")
        return 1

    known_names = [canon(n) for n in data["known_names"]]
    out_dir = Path(args.out) if args.out else split_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        require_weights(args.model, hint="请先完成第 3 步训练，或确认 --model 指向 6 类权重")
        from ultralytics import YOLO

        yolo = YOLO(args.model)
    except Exception as e:  # noqa: BLE001 - 模型加载失败直接给明确原因
        print(f"[错误] 加载模型失败：{e}")
        return 1
    model_names = {int(k): canon(str(v)) for k, v in (yolo.names or {}).items()}
    model_to_known = build_model_to_known(yolo.names or {}, known_names)

    print(f"[信息] 模型：{args.model}")
    print(f"[信息] 已知类别（{len(known_names)}）：{'、'.join(known_names)}")
    print(f"[信息] known/val {len(data['known_images'])} 张 ｜ "
          f"unknown/val {len(data['unknown_images'])} 张")
    print(f"[信息] imgsz={args.imgsz} conf={args.conf} device={args.device} "
          f"calib_ratio={args.calib_ratio} C_mis={args.cost_mis} C_ref={args.cost_ref}")

    # RQ1 口径提醒：若模型本身还认识留出类（例如用 8 类权重评估 S1），
    # 它可以把未知框"正确"预测为留出类，此时的误诊率会被低估；
    # 论文主结果应使用"已去掉该类的 7 类 LOCO 权重"。
    extra = sorted(set(model_names.values()) - set(known_names))
    if extra:
        print(f"[警告] 模型含划分外的类别：{'、'.join(extra)}")
        print("       这些框不会被计为已知类误诊，RQ1 误诊率可能偏低；"
              "论文主结果请使用对应的 LOCO 权重。")

    capture = DetectCapture()
    t0 = time.time()

    # 预热：先跑一张图，让 Ultralytics 建立 predictor 并完成 warmup；
    # 之后对 predictor 内部的模型（setup_model 会 deepcopy）挂 hook，避免
    # 把 warmup 前向误当成真实图片，也避免 hook 挂在未被使用的副本上。
    if data["known_images"]:
        try:
            yolo.predict(source=str(data["known_images"][0]), conf=args.conf,
                         imgsz=args.imgsz, device=_device(args.device), verbose=False)
        except Exception as e:  # noqa: BLE001
            print(f"[错误] 预热推理失败，无法继续：{e}")
            return 1
        backend = getattr(yolo.predictor, "model", None)
        target = getattr(backend, "model", backend)
        if target is not None:
            capture.attach(target)

    capture.reset()
    capture.armed = capture.status == "ok"
    print("[推理] known/val ……")
    try:
        known = infer_known(yolo, data["known_images"], data["known_labels"],
                            known_names, model_to_known, model_names, capture, args)
        capture.reset()
        capture.armed = capture.status == "ok"
        print("[推理] unknown/val ……")
        unknown = infer_unknown(yolo, data["unknown_images"], data["unknown_labels"],
                                model_names, capture, args)
    except Exception as e:  # noqa: BLE001 - 推理失败给出明确原因而不是堆栈
        capture.detach()
        print(f"[错误] 推理失败：{e}")
        return 1
    capture.armed = False
    capture.detach()

    if capture.status == "ok" and capture.n_batches == 0:
        capture.status = "unavailable"
        capture.reason = ("未捕获到任何 Detect 前向输出（可能 hook 未挂在 predictor 实际使用的模型上，"
                          "或 Ultralytics 版本差异）")

    unavailable: dict[str, str] = {}
    if capture.status != "ok":
        unavailable["entropy"] = capture.reason or "Detect hook 不可用"
        unavailable["energy"] = capture.reason or "Detect hook 不可用"
        unavailable["mahalanobis"] = capture.reason or "Detect hook 不可用"
        unavailable["proto"] = capture.reason or "Detect hook 不可用"

    known_items = known["items"]
    unknown_items = unknown["items"]
    if not known_items:
        print("[错误] known/val 没有可用检测框（conf 过高或权重与数据不匹配），无法校准阈值")
        print("       可尝试降低 --conf（如 0.01）后重跑。")
        return 1

    # ---- 特征分布（Mahalanobis / Prototype）----
    stats, stats_reason = fit_feature_stats(known_items, len(known_names))
    if stats is None:
        for name in FEATURE_SCORERS:
            unavailable.setdefault(name, stats_reason)
    else:
        add_feature_scores(known_items, stats)
        add_feature_scores(unknown_items, stats)

    # ---- 融合分数 ----
    if all(name not in unavailable for name in FUSION_INPUTS):
        fusion, fusion_reason = fit_fusion(known_items + unknown_items)
        if fusion is None:
            unavailable["fusion"] = fusion_reason
        else:
            apply_fusion(known_items, fusion)
            apply_fusion(unknown_items, fusion)
    else:
        missing = [n for n in FUSION_INPUTS if n in unavailable]
        unavailable["fusion"] = f"依赖的打分器不可用：{'、'.join(missing)}"

    # 只评估实际可用的打分器
    available = [s for s in selected if s not in unavailable]
    if not available:
        available = ["msp"]
        unavailable.pop("msp", None)
    primary = "fusion" if "fusion" in available else available[0]

    n_unknown_gt = int(unknown["n_gt"])
    rejection: dict[str, dict] = {}
    rq2: dict[str, dict] = {}
    for scorer in available:
        summary = scorer_summary(scorer, known_items, unknown_items, n_unknown_gt, args)
        rq2[scorer] = {
            "n_known": summary["n_known"],
            "n_unknown": summary["n_unknown"],
            "auroc": summary["auroc"],
            "auprc": summary["auprc"],
            "fpr_at_95tpr": summary["fpr_at_95tpr"],
        }
        rejection[scorer] = {
            "tau_calib": summary["tau_calib"],
            "calib": summary["calib"],
            "tau_cost": summary["tau_cost"],
            "cost": summary["cost"],
            "cost_optimal": summary["cost_optimal"],
        }

    # ---- 选择性预测（框级 + 图级）----
    selective: dict[str, dict] = {}
    for scorer in available:
        selective[scorer] = selective_metrics(known_items, unknown_items, scorer)

    # ---- 代价比扫描（C_ref 固定为 1）----
    n_images = len(data["known_images"]) + len(data["unknown_images"])
    sweep_rows = scan_cost_ratios(known_items, unknown_items, n_unknown_gt, primary,
                                  cost_ratios, n_images)
    recommend_row = next((row for row in sweep_rows
                          if abs(row["cost_ratio"] - args.recommend_ratio) < 1e-9), None)
    for row in sweep_rows:
        tag = "（推荐）" if row is recommend_row else ""
        print(f"  [代价比] {row['cost_ratio']:>6g}:1 τ={_fmt(row.get('tau'))} "
              f"误诊率={_fmt(row.get('misdiagnosis_rate'), pct=True)} "
              f"转人工率={_fmt(row.get('referral_rate'), pct=True)} "
              f"代价/图={_fmt(row.get('expected_cost_per_image'))}{tag}")

    known_map = compute_map50_95(known["images"], known_names)
    anchor_ious = [it["anchor_iou"] for it in known_items + unknown_items
                   if it["anchor_iou"] == it["anchor_iou"]]
    anchor_median = float(np.median(anchor_ious)) if anchor_ious else float("nan")

    rq1 = {
        "closed_set_misdiagnosis_rate": (round(len(unknown_items) / n_unknown_gt, 4)
                                         if n_unknown_gt else None),
        "unknown_mean_conf": (round(float(np.mean([it["conf"] for it in unknown_items])), 4)
                              if unknown_items else None),
        "known_mean_conf": (round(float(np.mean([it["conf"] for it in known_items])), 4)
                            if known_items else None),
        "unknown_detected_boxes": len(unknown_items),
        "unknown_gt_boxes": n_unknown_gt,
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(args.model),
        "split_dir": str(split_dir.resolve()),
        "known_classes": known_names,
        "n_known_classes": len(known_names),
        "model_classes": {str(k): canon(v) for k, v in model_names.items()},
        "class_mapping": {str(k): v for k, v in model_to_known.items()},
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": IOU_MATCH,
        "device": args.device,
        "calib_ratio": args.calib_ratio,
        "cost_mis": args.cost_mis,
        "cost_ref": args.cost_ref,
        "scorers_requested": selected,
        "scorers_available": available,
        "scorer_unavailable": unavailable,
        "data": {
            "known_val_images": len(data["known_images"]),
            "unknown_val_images": len(data["unknown_images"]),
            "known_boxes": len(known_items),
            "unknown_gt_boxes": n_unknown_gt,
            "unknown_detected_boxes": len(unknown_items),
            "anchor_parse_failures": known["failed_boxes"] + unknown["failed_boxes"],
        },
        "rq1": rq1,
        "rq2": rq2,
        "rejection": rejection,
        "selective_prediction": {
            "scores_direction": "分数越大越像已知（未知度已取负）；覆盖 = 不拒识",
            "scorers": selective,
            "image_level": {
                "implemented": bool(selective),
                "definition": ("同图多个框时取最可疑框（未知度最大）代表整图；"
                               "图内含未知框且未被拒识即记该图错误"),
                "reason": "" if selective else "没有可用的检测框打分结果，无法做图级聚合",
            },
            "notes": [
                "已知框要求同类且 IoU ≥ 0.5，已知侧的类别错误（未匹配 FP）不进入选择集；"
                "AURC 只衡量保留已知框 vs 拒识未知框的权衡，属已知简化。",
                "AURC 为标准定义（覆盖率必到 1，完美分离也不为 0，约等于未知占比）；"
                "E-AURC 为相对完美排序基线的超出量，完美分离为 0。两者需同时报告。",
            ],
        },
        "cost_ratio_sweep": {
            "scorer": primary,
            "cost_ref": 1.0,
            "n_images": n_images,
            "recommend_ratio": args.recommend_ratio,
            "ratios": cost_ratios,
            "rows": sweep_rows,
            "recommend": recommend_row,
        },
        "known_map": known_map,
        "anchor_iou_median": anchor_median,
        "primary_scorer": primary,
        "notes": [
            "主结果阈值 τ_calib 只在 known/val 的已知类检测框上按固定误拒率标定，测试/验证集不调参；",
            "τ_cost 在已知+未知验证数据上扫描，仅用于 RQ5 代价敏感性分析；",
            "Mahalanobis/Prototype 的类均值与共享协方差仅在 known/val 上拟合，未知类不参与；",
            "融合权重在同一个 known/val + unknown/val 上拟合，存在轻度乐观偏差，论文需注明。",
        ],
        "conclusion": "",
    }
    report["conclusion"] = build_conclusion(report)

    # ---- 落盘 ----
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / "metrics.md"
    write_markdown(md_path, report)

    primary_row = rejection[primary]
    tau_calib = (primary_row["tau_calib"] if primary_row["tau_calib"] is not None
                 else float("nan"))
    tau_cost = (primary_row["tau_cost"] if primary_row["tau_cost"] is not None
                else float("nan"))
    csv_path = out_dir / "scores.csv"
    write_scores_csv(csv_path, known_items + unknown_items, primary, tau_calib, tau_cost)

    png_path = out_dir / "risk_coverage.png"
    try:
        plot_risk_coverage(png_path, known_items, unknown_items, n_unknown_gt,
                           primary, tau_calib, tau_cost, args)
    except Exception as e:  # noqa: BLE001 - 画图失败不应丢掉已有指标
        print(f"[警告] 绘制风险-覆盖率曲线失败：{e}")

    cost_csv_path = out_dir / "cost_ratio.csv"
    write_cost_ratio_csv(cost_csv_path, sweep_rows)
    decision_png = out_dir / "decision_curve.png"
    try:
        plot_decision_curve(decision_png, sweep_rows, args.recommend_ratio, primary)
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 绘制代价比决策曲线失败：{e}")

    elapsed = time.time() - t0
    print(f"\n[完成] 已知框 {len(known_items)}，未知 GT {n_unknown_gt}，"
          f"检出未知 {len(unknown_items)}，耗时 {elapsed:.0f}s")
    for scorer in available:
        row = rq2[scorer]
        print(f"  {scorer:<12} AUROC={_fmt(row['auroc'])} AUPR={_fmt(row['auprc'])} "
              f"FPR@95TPR={_fmt(row['fpr_at_95tpr'])} "
              f"AURC={_fmt(selective[scorer]['aurc'])} "
              f"E-AURC={_fmt(selective[scorer]['e_aurc'])} "
              f"AURC_img={_fmt(selective[scorer]['aurc_image'])}")
    for name, reason in unavailable.items():
        print(f"  [降级] {name}：{reason}")
    print(f"[输出] {json_path}")
    print(f"[输出] {md_path}")
    print(f"[输出] {csv_path}")
    print(f"[输出] {png_path}")
    print(f"[输出] {cost_csv_path}")
    print(f"[输出] {decision_png}")
    print(f"[结论] {report['conclusion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
