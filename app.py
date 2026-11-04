"""茶园气象服务系统 - 导航入口（st.navigation 自定义页面名称）。

运行方式: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src import config, db, ui
from src.dev_console import main as dev_main
from src.home_page import main as home_main

st.set_page_config(
    page_title="茶园气象服务系统",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()


@st.cache_resource(ttl=3600, show_spinner=False)
def _ensure_database_schema() -> bool:
    """每个 Streamlit 进程最多每小时执行一次幂等建表，避免每次 rerun 都连库。"""
    db.init_db()
    db.init_db_ext()
    return True


# 启动时初始化 MySQL 表（成功结果缓存；失败不缓存，数据库恢复后会自动重试）
try:
    _ensure_database_schema()
except Exception:  # noqa: BLE001
    pass

# 自定义导航：名称、图标、顺序完全可控
pages = [
    st.Page(home_main, title="首页", icon="🏠"),
    st.Page("pages/1_气象监测.py", title="气象监测", icon="🌡"),
    st.Page("pages/2_风险预警.py", title="风险预警", icon="⚠️"),
    st.Page("pages/3_智能问答.py", title="智能问答", icon="🤖"),
    st.Page("pages/4_农事建议.py", title="农事建议", icon="🌾"),
    st.Page("pages/5_我的茶园.py", title="我的茶园", icon="👤"),
    st.Page("pages/6_智能执行.py", title="智能执行", icon="🤖"),
]

# 开发者模式：条件显示开发者控制台（无需薄壳文件）
if config.is_dev_mode():
    pages.append(st.Page(dev_main, title="开发者控制台", icon="🛠"))

pg = st.navigation(pages)
pg.run()
