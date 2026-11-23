"""评估入口 fail-fast 守卫测试。

论文实验最怕的不是报错，而是「悄悄换成模拟检测」：
权重缺失时若降级为合成帧推理，metrics 里的数字全部无效且日志看不出来。
require_weights 必须在权重缺失/损坏时直接中止。
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "training" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

import common  # noqa: E402


def test_missing_weights_exits(tmp_path: Path) -> None:
    """权重不存在时以退出码 2 中止，而不是继续跑模拟推理。"""
    with pytest.raises(SystemExit) as exc:
        common.require_weights(str(tmp_path / "nope.pt"))
    assert exc.value.code == 2


def test_tiny_weights_treated_as_corrupt(tmp_path: Path) -> None:
    """过小的文件多半是下载中断，同样中止。"""
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"0" * 100)
    with pytest.raises(SystemExit) as exc:
        common.require_weights(str(bad))
    assert exc.value.code == 2


def test_valid_weights_returns_path(tmp_path: Path) -> None:
    """正常大小（>4KB）的文件通过并返回 Path，便于链式使用。"""
    ok = tmp_path / "best.pt"
    ok.write_bytes(b"0" * 8192)
    got = common.require_weights(str(ok))
    assert isinstance(got, Path)
    assert got == ok
