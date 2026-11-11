"""病虫害识别页：上传茶叶照片 → YOLO 检测 → 标注图 + 检出明细 + 防治建议。

本页只做静态照片识别，不涉及摄像头；所有推理在本地完成，不调用任何大模型。
可选能力：
  - TTA 增强推理（翻转/缩放融合，精度略升，耗时约 3 倍）
  - 切片检测（大图切小块分别检测再合并，提升小病斑召回）
  - 批量上传（多张照片一次识别，汇总可导出 CSV）
默认置信度阈值优先读取 training/calibrate_conf.py 标定出的 data/models/conf_threshold.json。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src import config, ui
from src.vision.annotate import draw_detections, frame_to_jpeg_bytes
from src.vision.detector import detector as vision_detector

# 类别中文名（与 CLASS_NAMES 顺序无关，按英文名映射）
CLASS_CN = {
    "Black rot": "茶黑腐病",
    "Brown blight": "茶褐斑病",
    "Leaf rust": "茶锈病",
    "Red Spider": "茶红蜘蛛",
    "Tea Mosquito bug": "茶角盲蝽",
    "White spot": "茶白星病",
    "Healthy": "健康叶片",
    "disease": "病害（未细分）",
}

# 各类别的田间处置建议（规则建议，不依赖大模型）
ADVICE = {
    "Black rot": "及时剪除病枝、病叶并集中销毁，避免园内积水与枝叶过密；发病初期可喷施多菌灵等杀菌剂，7~10 天一次，连喷 2~3 次。",
    "Brown blight": "清除病叶、改善通风透光；可选用代森锰锌或吡唑醚菌酯类杀菌剂，注意轮换用药、控制采摘安全间隔期。",
    "Leaf rust": "减少叶面长时间湿润，摘除病叶；可喷施戊唑醇等三唑类杀菌剂，重点喷施叶背。",
    "Red Spider": "翻查叶背虫口密度，达标时喷施哒螨灵或联苯肼酯，注意与其它药剂轮换，避免抗性。",
    "Tea Mosquito bug": "剪除受害嫩梢，虫口高峰期可喷施吡虫啉或噻虫嗪，严格遵守采摘安全间隔期。",
    "White spot": "清除病叶并改善通风；可喷施苯醚甲环唑或甲基硫菌灵，雨后及时补喷。",
    "disease": "病害类别未细分，建议就近请农技人员现场复核，或采样送检后对症用药。",
    "Healthy": "未见明显病斑，保持常规巡园与通风管理即可。",
}

MAX_SIDE = 1920  # 超大照片先等比缩放，避免推理内存占用过高
THRESHOLD_FILE = config.DATA_DIR / "models" / "conf_threshold.json"


def _default_conf() -> float:
    """读取标定脚本产出的推荐阈值；没有则用 0.25。"""
    try:
        cfg = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
        return float(min(0.90, max(0.10, cfg["conf"])))
    except Exception:  # noqa: BLE001 - 文件缺失/损坏都回退默认值
        return 0.25


@st.cache_data(show_spinner=False)
def _read_image(raw: bytes) -> np.ndarray:
    """上传字节 → RGB numpy（自动纠正 EXIF 方向、限制最大边长）。"""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(img)


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _cn(cls: str) -> str:
    return CLASS_CN.get(cls, cls)


@st.cache_data(show_spinner=False, max_entries=12)
def _detect_cached(raw: bytes, conf: float, imgsz: int, tta: bool, tiled: bool) -> dict:
    """按「图片内容 + 阈值 + 分辨率 + 增强开关」缓存结果，避免 rerun 重复推理。"""
    rgb = _read_image(raw)
    return vision_detector.detect_upload(
        _to_bgr(rgb), imgsz=imgsz, conf=conf, tta=tta, tiled=tiled)


def _render_model_status() -> None:
    """侧边栏：模型加载状态与重载入口。"""
    with st.sidebar:
        st.markdown("### 🔬 识别模型")
        if vision_detector.ready:
            st.success("YOLO 模型已加载")
            st.caption(f"后端：{vision_detector.backend}")
            st.caption(f"权重：{Path(vision_detector.model_path).name}")
            if THRESHOLD_FILE.exists():
                st.caption(f"标定阈值：{_default_conf():.2f}")
        else:
            st.warning("模型未加载（缺少 ultralytics 或权重文件）")
            st.caption("安装：`pip install ultralytics opencv-python`")
            st.caption("权重：`data/models/tea_yolo.pt`")
        if st.button("🔄 重新加载模型"):
            vision_detector.reload()
            _detect_cached.clear()
            st.rerun()
        st.caption("识别结果只由本地 YOLO 权重产生，不经过任何大模型推断。")


def _render_result(name: str, rgb: np.ndarray, result: dict,
                   conf: float, imgsz: int, tta: bool, tiled: bool) -> dict:
    """渲染单张照片的完整识别结果，返回汇总行。"""
    dets = result.get("detections") or []
    diseases = result.get("summary", {}).get("diseases") or []
    annotated = draw_detections(_to_bgr(rgb), result)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📷 原始照片")
        st.image(rgb, width="stretch")
    with col_r:
        st.subheader("🖼 识别标注")
        if dets:
            st.image(np.ascontiguousarray(annotated[:, :, ::-1]), width="stretch")
        else:
            st.image(rgb, width="stretch")
            st.caption("未检出目标，显示原图。")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("检出目标", f"{result['summary']['total']} 处")
    m2.metric("病害种类", f"{len(diseases)} 种")
    m3.metric("最高置信度", f"{result['summary']['max_conf']:.0%}")
    healthy = result["summary"]["healthy"]
    m4.metric("判定", "未见病害" if healthy else "疑似病虫害")

    if healthy:
        st.success("未检出病害目标。若叶片确有异常，可尝试降低置信度阈值、提高推理分辨率，或换角度补拍。")
    else:
        st.error("检出疑似病虫害：" + "、".join(_cn(d) for d in diseases)
                 + "。请结合田间实际情况复核，必要时采样送检。")

    if dets:
        st.subheader("📋 检出明细")
        rows = [{
            "#": i + 1,
            "类别": d["class"],
            "中文名": _cn(d["class"]),
            "置信度": f"{d['conf']:.0%}",
            "位置 (x1, y1, x2, y2)": ", ".join(str(int(v)) for v in d["bbox"]),
        } for i, d in enumerate(dets)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("🛠 防治建议")
    targets = diseases or (["Healthy"] if dets else [])
    if targets:
        for cls in targets:
            label = "✅ " if cls == "Healthy" else "⚠️ "
            with st.container(border=True):
                st.markdown(f"**{label}{_cn(cls)}**（{cls}）")
                st.write(ADVICE.get(cls, ADVICE["disease"]))
    else:
        st.info("本张照片未检出任何目标，暂无针对性建议。可参考拍摄建议重新拍摄。")

    d1, d2 = st.columns([1, 3])
    with d1:
        img_bytes = frame_to_jpeg_bytes(annotated, quality=92)
        if img_bytes:
            st.download_button(
                "⬇️ 下载标注图",
                data=img_bytes,
                file_name=f"{Path(name).stem}_detect.jpg",
                mime="image/jpeg",
                width="stretch",
            )
    with d2:
        flags = []
        if tta:
            flags.append("TTA")
        if tiled:
            flags.append("切片")
        st.caption(
            f"检测时间：{result['ts']} ｜ 后端：{result['backend']} ｜ "
            f"模型：{Path(result['model']).name} ｜ 分辨率：{imgsz} ｜ 阈值：{conf:.2f}"
            + (f" ｜ 模式：{' + '.join(flags)}" if flags else "")
        )

    return {
        "文件": name,
        "检出目标": result["summary"]["total"],
        "病害种类": len(diseases),
        "病害名称": "、".join(_cn(d) for d in diseases) or "—",
        "最高置信度": f"{result['summary']['max_conf']:.0%}",
        "判定": "未见病害" if healthy else "疑似病虫害",
    }


def main() -> None:
    ui.inject_css()

    st.title("🔬 病虫害识别（上传照片）")
    st.caption("上传茶树叶/嫩梢照片，用本地 YOLO 模型自动框选并识别病害与虫害。"
               "支持多张批量识别；全部推理在本机完成，照片不会上传到第三方。")

    _render_model_status()

    if not vision_detector.ready:
        st.warning(
            "**真实识别不可用**：未加载 YOLO 模型（`ultralytics` 未安装或权重文件缺失）。\n\n"
            "为避免误导，本页不会用模拟结果替代真实识别。请先安装依赖："
        )
        st.code("pip install ultralytics opencv-python\n# 或重新训练：python training/train_yolo.py --quick",
                language="bash")
        st.info("安装完成后点击侧边栏「重新加载模型」，或重启应用。")
        return

    uploaded_files = st.file_uploader(
        "上传茶叶照片（支持 JPG / PNG / WEBP / BMP，可多选）",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )

    c1, c2, c3, c4 = st.columns([2, 2, 1.3, 1.3])
    with c1:
        conf = st.slider("置信度阈值", min_value=0.10, max_value=0.90,
                         value=_default_conf(), step=0.05,
                         help="默认值来自验证集自动标定；阈值越低检出越多，但误检也更多。")
    with c2:
        imgsz = st.select_slider("推理分辨率", options=[320, 480, 640, 800], value=640,
                                 help="静态照片建议 640；光线差或叶片较小时可调到 800。")
    with c3:
        tta = st.checkbox("增强推理 (TTA)", value=False,
                          help="翻转/缩放多路推理融合，精度略升，耗时约 3 倍。")
    with c4:
        tiled = st.checkbox("切片检测", value=False,
                            help="大图切成 640 小块分别检测再合并，提升小病斑召回，耗时随切片数增加。")

    if not uploaded_files:
        st.info("请先上传照片。建议：单叶或嫩梢占画面 1/2 以上、光线均匀、避免严重抖动与反光。")
        with st.expander("📷 拍摄与识别建议"):
            st.markdown(
                "- 对焦在病斑/虫害部位，叶片占画面一半以上；\n"
                "- 避免强反光与逆光，阴天或散射光效果更佳；\n"
                "- 一张照片只拍 1~3 片叶，识别更准；\n"
                "- 同一症状建议换角度多拍几张，便于人工复核；\n"
                "- 识别结果为辅助判断，用药前请结合农技人员意见。"
            )
        return

    summaries = []
    single = len(uploaded_files) == 1
    for idx, uploaded in enumerate(uploaded_files, 1):
        raw = uploaded.getvalue()
        rgb = _read_image(raw)
        with st.spinner(f"正在识别第 {idx}/{len(uploaded_files)} 张：{uploaded.name}"):
            result = _detect_cached(raw, conf, imgsz, tta, tiled)

        st.divider()
        title = f"{idx}. {uploaded.name}" if not single else uploaded.name
        if single:
            st.markdown(f"**{title}**")
            summaries.append(_render_result(uploaded.name, rgb, result, conf, imgsz, tta, tiled))
        else:
            ok = result["summary"]["total"]
            with st.expander(f"{title}　检出 {ok} 处", expanded=(idx == 1)):
                summaries.append(_render_result(uploaded.name, rgb, result, conf, imgsz, tta, tiled))

    if len(summaries) > 1:
        st.divider()
        st.subheader("📊 批量汇总")
        df = pd.DataFrame(summaries)
        st.dataframe(df, width="stretch", hide_index=True)
        total_targets = int(df["检出目标"].sum())
        bad = int((df["判定"] == "疑似病虫害").sum())
        a, b, c = st.columns(3)
        a.metric("照片数", len(df))
        b.metric("检出目标合计", f"{total_targets} 处")
        c.metric("疑似病虫害照片", f"{bad} / {len(df)}")
        st.download_button(
            "⬇️ 导出汇总 CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="tea_detect_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
