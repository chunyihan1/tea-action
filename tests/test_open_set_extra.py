"""open_set_eval 新增指标 / McNemar / 聚合脚本的补充测试（纯逻辑，不碰真实数据）。

覆盖：
  1. 未知类静默漏检指标，以及"误诊率 + 未知漏检率 + 未知召回率 = 1"；
  2. 打分器两两 McNemar 的连续性校正手算用例；
  3. aggregate_open_set 的 mean / std（样本标准差）与缺失字段跳过；
  4. 聚合脚本 --help 可正常退出。
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "training" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

import aggregate_open_set as agg  # noqa: E402
import open_set_eval as ose  # noqa: E402


# ============================================================
# 1. 未知类静默漏检指标
# ============================================================
def test_unknown_detection_metrics_sum_to_one():
    """5 个未知 GT、检出 3 个、其中 2 个被拒识：三个比例必须相加为 1。"""
    known = [{"source": "known", "scores": {"msp": s}} for s in (0.1, 0.2)]
    unknown = [{"source": "unknown", "scores": {"msp": s}} for s in (0.3, 0.6, 0.8)]
    detail = ose.evaluate_at_threshold(known, unknown, n_unknown_gt=5, scorer="msp",
                                       tau=0.55, cost_mis=10, cost_ref=1)

    assert detail["unknown_detected_rate"] == pytest.approx(0.6)     # 3/5
    assert detail["unknown_detection_miss_rate"] == pytest.approx(0.4)  # 2/5
    assert detail["undetected_unknown_count"] == 2
    assert detail["unknown_recall"] == pytest.approx(0.4)            # 2/5
    assert detail["misdiagnosis_rate"] == pytest.approx(0.2)         # 1/5

    assert (detail["unknown_detected_rate"]
            + detail["unknown_detection_miss_rate"]) == pytest.approx(1.0)
    assert (detail["misdiagnosis_rate"] + detail["unknown_detection_miss_rate"]
            + detail["unknown_recall"]) == pytest.approx(1.0)


def test_zero_misdiagnosis_does_not_mean_safe():
    """未知框一个都没检出：误诊率 0，但未知漏检率 100%（静默失败）。"""
    known = [{"source": "known", "scores": {"msp": 0.1}}]
    detail = ose.evaluate_at_threshold(known, [], n_unknown_gt=4, scorer="msp",
                                       tau=0.55, cost_mis=10, cost_ref=1)
    assert detail["misdiagnosis_rate"] == 0.0
    assert detail["unknown_recall"] == 0.0
    assert detail["unknown_detection_miss_rate"] == 1.0
    assert detail["undetected_unknown_count"] == 4


def test_unknown_detection_metrics_none_when_no_gt():
    """分母（未知 GT 数）为 0 时三个新字段都返回 None，而不是崩溃。"""
    detail = ose.evaluate_at_threshold([], [], n_unknown_gt=0, scorer="msp",
                                       tau=0.55, cost_mis=10, cost_ref=1)
    assert detail["unknown_detected_rate"] is None
    assert detail["unknown_detection_miss_rate"] is None
    assert detail["undetected_unknown_count"] is None


# ============================================================
# 2. McNemar 手算用例
# ============================================================
def test_mcnemar_pair_hand_computed():
    """连续性校正 χ²=(|b−c|−1)²/(b+c)，p=erfc(sqrt(χ²/2))。

    说明：任务书示例"b=10,c=2 → χ²=4.5"与上述公式不一致（公式给出 49/12≈4.0833），
    这里按公式断言；能精确得到 χ²=4.5、p≈0.0339 的不一致对是 (b,c)=(14,4)。
    """
    case = ose.mcnemar_pair(10, 2)
    assert case["chi2"] == pytest.approx(49.0 / 12.0, abs=1e-4)      # 4.0833
    assert case["p_raw"] == pytest.approx(0.0433, abs=1e-3)

    quoted = ose.mcnemar_pair(14, 4)                                  # χ²=81/18=4.5
    assert quoted["chi2"] == pytest.approx(4.5, abs=1e-6)
    assert quoted["p_raw"] == pytest.approx(0.0339, abs=1e-4)
    assert quoted["p_raw"] == pytest.approx(math.erfc(math.sqrt(4.5 / 2.0)), abs=1e-6)

    # b+c=0（两打分器判定完全一致）时 p=1
    assert ose.mcnemar_pair(0, 0) == {"chi2": 0.0, "p_raw": 1.0}


def test_scorer_mcnemar_pairs_boxes_with_tau_calib():
    """10 个框 A 对 B 错、2 个框 A 错 B 对；各自 τ_calib 下统计配对表。"""
    known: list[dict] = []
    for i in range(10):
        known.append({"source": "known", "scores": {"msp": 0.0, "energy": 1.0}})
    for i in range(2):
        known.append({"source": "known", "scores": {"msp": 1.0, "energy": 0.0}})
    rejection = {"msp": {"tau_calib": 0.0}, "energy": {"tau_calib": 0.0}}

    pairs = ose.scorer_mcnemar(known, [], rejection, ["msp", "energy"])
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["scorer_a"] == "msp" and pair["scorer_b"] == "energy"
    assert pair["b"] == 10 and pair["c"] == 2
    assert pair["n_boxes"] == 12
    assert pair["p_holm"] == pair["p_raw"]          # 单次检验，Holm 不改变 p
    assert pair["significant"] is True              # p≈0.0455 < 0.05


# ============================================================
# 3. 聚合脚本
# ============================================================
def _minimal_report(auroc: float, recall: float, with_rejection: bool = True) -> dict:
    report = {
        "scorers_available": ["msp"],
        "rq2": {"msp": {"auroc": auroc, "auprc": 0.5, "fpr_at_95tpr": 0.1}},
        "selective_prediction": {"scorers": {"msp": {
            "aurc": 0.1, "e_aurc": 0.02,
            "coverage_at_5pct": 0.3, "coverage_at_10pct": 0.4,
        }}},
    }
    if with_rejection:
        report["rejection"] = {"msp": {"calib": {
            "unknown_recall": recall,
            "misdiagnosis_rate": 0.1,
            "unknown_detection_miss_rate": 0.2,
            "referral_rate": 0.3,
            "referral_rate_known": 0.05,
        }}}
    return report


def test_aggregate_mean_std_with_two_runs(tmp_path):
    """两个 run：mean 与样本标准差（n−1）都要正确；缺字段只跳过该字段。"""
    run_dir = tmp_path / "S1"
    run_dir.mkdir()
    p1 = run_dir / "metrics.json"
    p2 = run_dir / "metrics_seed2.json"
    p1.write_text(json.dumps(_minimal_report(0.8, 0.5)), encoding="utf-8")
    p2.write_text(json.dumps(_minimal_report(0.9, 0.5, with_rejection=False)),
                  encoding="utf-8")

    items = [(agg.group_of(path, r"(S\d+)"), agg.load_report(path)) for path in (p1, p2)]
    rows, groups, order = agg.build_rows(items)
    lookup = {(row["group"], row["scorer"], row["metric"]): row for row in rows}

    assert groups == ["S1"] and order["S1"] == ["msp"]
    auroc = lookup[("S1", "msp", "auroc")]
    assert auroc["mean"] == pytest.approx(0.85)
    assert auroc["std"] == pytest.approx(0.0707, abs=1e-4)
    assert auroc["min"] == 0.8 and auroc["max"] == 0.9 and auroc["n"] == 2

    # 第二个 run 没有 rejection：该指标只有 1 个有效值，std 必须为 None
    referral = lookup[("S1", "msp", "referral_rate")]
    assert referral["n"] == 1 and referral["std"] is None
    assert referral["mean"] == pytest.approx(0.3)

    # 缺失的 aupr 字段也已跳过：只统计存在值的 run
    assert lookup[("S1", "msp", "aupr")]["n"] == 2


def test_aggregate_cli_writes_files(tmp_path):
    """端到端：--inputs 两个文件，输出 aggregate.csv / aggregate.md。"""
    run_dir = tmp_path / "S5"
    run_dir.mkdir()
    p1 = run_dir / "a.json"
    p2 = run_dir / "b.json"
    p1.write_text(json.dumps(_minimal_report(0.8, 0.5)), encoding="utf-8")
    p2.write_text(json.dumps(_minimal_report(0.9, 0.6)), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert agg.main(["--inputs", str(p1), str(p2), "--out-dir", str(out_dir)]) == 0

    csv_text = (out_dir / "aggregate.csv").read_text(encoding="utf-8-sig")
    assert "S5,msp,auroc,0.85,0.0707,0.8,0.9,2" in csv_text
    md_text = (out_dir / "aggregate.md").read_text(encoding="utf-8")
    assert "## S5" in md_text and "0.8500 ± 0.0707" in md_text


# ============================================================
# 4. CLI --help
# ============================================================
def test_aggregate_help():
    result = subprocess.run([sys.executable, str(RESEARCH / "aggregate_open_set.py"), "--help"],
                            cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert "--inputs" in result.stdout and "--pattern" in result.stdout
    assert "--group-regex" in result.stdout and "--out-dir" in result.stdout
