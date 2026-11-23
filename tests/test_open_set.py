"""开放集划分 / 评估的纯逻辑测试（不依赖真实数据集与权重）。

覆盖三块：
  1. make_open_set_split：用 3 类小数据集检查"留出类彻底移除、unknown 标签改写为 0、
     known 类别号连续、report 生成"；
  2. open_set_eval 的指标实现：AUROC、FPR@95TPR、误诊率、期望代价的可手算例子；
  3. 两个脚本的 --help 均可执行（argparse 正常）。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "training" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

import make_open_set_split as mos  # noqa: E402
import open_set_eval as ose  # noqa: E402


# ============================================================
# 1. 构造小型数据集并跑划分
# ============================================================
def _make_dataset(root: Path) -> None:
    """3 类、共 12 张纯色小图：train 每类 2 张、valid 每类 2 张。"""
    names = ["Black rot", "Healthy", "Leaf rust"]
    for split in ("train", "valid"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for cid, name in enumerate(names):
            for k in range(2):
                stem = f"{name.replace(' ', '_')}_{split}_{k}"
                Image.new("RGB", (64, 64), color=(40 * cid + 10, 60 + 20 * k, 180)).save(
                    img_dir / f"{stem}.png")
                (lbl_dir / f"{stem}.txt").write_text(
                    f"{cid} 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "path: {}\n".format(root.as_posix())
        + "train: train/images\n"
        + "val: valid/images\n"
        + "names:\n"
        + "  0: Black rot\n  1: Healthy\n  2: Leaf rust\n",
        encoding="utf-8",
    )


def _label_classes(path: Path) -> list[int]:
    if not path.is_file():
        return []
    return [int(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_make_open_set_split(tmp_path):
    data_root = tmp_path / "dataset"
    _make_dataset(data_root)
    out = tmp_path / "open_set"
    rc = mos.main(["--data-root", str(data_root), "--out", str(out),
                   "--splits", "S1", "--mode", "copy"])
    assert rc == 0

    group = out / "S1"
    train_imgs = sorted(p.name for p in (group / "known" / "images" / "train").glob("*"))
    val_imgs = sorted(p.name for p in (group / "known" / "images" / "val").glob("*"))
    unknown_imgs = sorted(p.name for p in (group / "unknown" / "images" / "val").glob("*"))

    # 留出类（Leaf rust）在 known/train 中被整体移除，而不是只删标签
    assert train_imgs and all("Leaf_rust" not in name for name in train_imgs)
    assert len(train_imgs) == 4  # Black rot 2 + Healthy 2
    assert len(val_imgs) == 4
    assert len(unknown_imgs) == 2
    assert all("Leaf_rust" in name for name in unknown_imgs)

    # unknown 标签全部改写为 class 0
    for txt in (group / "unknown" / "labels" / "val").glob("*.txt"):
        classes = _label_classes(txt)
        assert classes and set(classes) == {0}

    # known 标签的类别号连续、无空洞（留出类不出现）
    used: set[int] = set()
    for txt in (group / "known" / "labels").rglob("*.txt"):
        used |= set(_label_classes(txt))
    assert used == {0, 1}

    # yaml 与 report
    k_cfg, k_names, _ = ose.load_data_yaml(group / "data_known.yaml")
    u_cfg, u_names, _ = ose.load_data_yaml(group / "data_unknown.yaml")
    assert k_names == ["Black rot", "Healthy"]
    assert u_names == ["unknown"]
    report = (group / "split_report.md").read_text(encoding="utf-8")
    assert "Leaf rust" in report and "known 类别号连续" in report
    assert "unknown" in report


def test_split_mode_list_and_short_name(tmp_path):
    """--mode list 不应写文件；--splits 接受类别短名。"""
    data_root = tmp_path / "dataset"
    _make_dataset(data_root)
    out = tmp_path / "open_set"
    rc = mos.main(["--data-root", str(data_root), "--out", str(out),
                   "--splits", "leaf rust", "--mode", "list"])
    assert rc == 0
    assert not (out / "S1").exists()


# ============================================================
# 2. 指标函数
# ============================================================
def test_auroc_perfect_separation():
    assert ose.auroc([0.9, 0.8, 0.7], [0.2, 0.1]) == 1.0
    assert ose.auprc([0.9, 0.8, 0.7], [0.2, 0.1]) == 1.0


def test_auroc_ties_and_worst_case():
    # 完全反向 -> 0；全部相等 -> 0.5（平均秩口径）
    assert ose.auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert ose.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_fpr_at_95tpr_hand_computed():
    # 分数越大越像已知；τ = unknown 的 95% 分位 = 0.575，已知 < τ 的有 2/3
    assert ose.fpr_at_tpr([0.2, 0.5, 0.8], [0.1, 0.6]) == pytest.approx(2.0 / 3.0)
    # 完全可分时 FPR = 0
    assert ose.fpr_at_tpr([0.9, 0.8, 0.7], [0.2, 0.1]) == 0.0


def test_expected_cost_and_rejection_metrics():
    assert ose.expected_cost(2, 3, cost_mis=10, cost_ref=1) == 23.0

    known = [{"source": "known", "scores": {"msp": s}} for s in (0.1, 0.2, 0.5, 0.9)]
    unknown = [{"source": "unknown", "scores": {"msp": s}} for s in (0.3, 0.6, 0.8)]
    detail = ose.evaluate_at_threshold(known, unknown, n_unknown_gt=5, scorer="msp",
                                       tau=0.55, cost_mis=10, cost_ref=1)
    assert detail["known_false_rejection_rate"] == 0.25          # 1/4
    assert detail["unknown_recall"] == 0.4                       # 2/5（另 2 个 GT 未检出）
    assert detail["misdiagnosis_rate"] == 0.2                    # 1/5
    assert detail["undetected_rate"] == 0.4                      # 2/5
    assert detail["referred_boxes"] == 3                         # 1 已知 + 2 未知
    assert detail["referral_rate"] == 0.4286                     # 3/7，输出保留 4 位
    assert detail["expected_cost"] == 13.0                       # 10*1 + 1*3


def test_ap_from_tp_hand_computed():
    # 2 个 GT：一个 TP(0.9) + 一个 FP(0.8)，recall 只到 0.5，precision 在 recall=0.5 时为 1
    ap = ose.ap_from_tp(np.array([0.9, 0.8]), np.array([True, False]), n_gt=2)
    assert ap == 0.5


def test_model_to_known_mapping_by_short_name():
    mapping = ose.build_model_to_known(
        {0: "Black rot of tea", 5: "White spot of tea", 7: "Tea leaf"},
        ["White spot", "Healthy"],
    )
    assert mapping[0] == -1       # 不在已知集
    assert mapping[5] == 0
    assert mapping[7] == 1


# ============================================================
# 2b. 选择性预测与代价比扫描（纯函数，可手算）
# ============================================================
def test_aurc_perfect_separation():
    """完美分离：标准 AURC 不为 0（覆盖率必然到 1），E-AURC 才为 0。

    3 已知 + 2 未知、错误全排最后：风险在覆盖率 0.6/0.8/1.0 处为 0/0.25/0.40，
    梯形积分 = 0.09；与完美排序基线重合，故 E-AURC = 0。
    """
    scores, errors = [0.9, 0.8, 0.7, 0.2, 0.1], [0, 0, 0, 1, 1]
    assert ose.aurc(scores, errors) == pytest.approx(0.09, abs=1e-6)
    assert ose.excess_aurc(scores, errors) == pytest.approx(0.0, abs=1e-9)


def test_aurc_uninformative_scores_approach_unknown_ratio():
    """分数完全相同（不可分）时风险恒为未知占比 q，标准 AURC ≈ q。

    E-AURC 在不可分时约为 q(1-q)，明显大于 0，说明排序没有带来任何收益。
    """
    q = 0.1
    scores, errors = [0.5] * 10, [0.0] * 9 + [1.0]
    assert ose.aurc(scores, errors) == pytest.approx(q, abs=0.02)
    assert ose.excess_aurc(scores, errors) == pytest.approx(q * (1 - q), abs=0.02)


def test_coverage_at_risk_hand_computed():
    """4 已知 + 2 未知，分数降序为 K,K,K,U,U,K：risk=0 时只能覆盖前 3 个 → 3/6。"""
    scores = [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
    errors = [0, 0, 0, 1, 1, 0]
    assert ose.coverage_at_risk(scores, errors, 0.0) == pytest.approx(0.5)
    # 前 4 个累计错误率 1/4 = 0.25，第 5 个为 2/5 = 0.4 > 0.25
    assert ose.coverage_at_risk(scores, errors, 0.25) == pytest.approx(4.0 / 6.0)


def test_cost_ratio_sweep_more_conservative_at_higher_ratio():
    """代价比越高，最优工作点越保守（转人工率不低于 1:1 的推荐 τ）。"""
    known = [{"source": "known", "scores": {"msp": 0.02 + 0.05 * i}} for i in range(20)]
    unknown = [{"source": "unknown", "scores": {"msp": 0.05 + 0.045 * i}} for i in range(20)]
    rows = ose.scan_cost_ratios(known, unknown, len(unknown), "msp", [1, 10], n_images=40)
    assert len(rows) == 2
    assert rows[1]["referral_rate"] >= rows[0]["referral_rate"]
    assert rows[1]["expected_cost_per_image"] is not None


# ============================================================
# 3. CLI --help
# ============================================================
def _run_help(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), "--help"],
                          cwd=str(REPO), capture_output=True, text=True, timeout=120)


def test_help_flags():
    split_help = _run_help(RESEARCH / "make_open_set_split.py")
    assert split_help.returncode == 0
    assert "--data-root" in split_help.stdout and "--mode" in split_help.stdout

    eval_help = _run_help(RESEARCH / "open_set_eval.py")
    assert eval_help.returncode == 0
    assert "--split-dir" in eval_help.stdout and "--calib-ratio" in eval_help.stdout
