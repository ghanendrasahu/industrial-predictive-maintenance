"""Streamlit dashboard: fleet risk overview + per-machine decision panel.

Run with:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings  # noqa: E402
from src.predict import assess_all, load_bundle, rank_machines  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭", layout="wide")

BAND_COLORS = {"CRITICAL": "#d62728", "WATCH": "#ff7f0e", "HEALTHY": "#2ca02c"}


@st.cache_resource(show_spinner=False)
def load_data(data_path: str, model_path: str):
    """Load telemetry + model bundle once per session."""
    telemetry = pd.read_csv(data_path)
    bundle = load_bundle(model_path)
    return telemetry, bundle


@st.cache_data(show_spinner=False)
def build_assessments(telemetry: pd.DataFrame, bundle: dict) -> list[dict]:
    return rank_machines(assess_all(telemetry, bundle))


st.title("🏭 Industrial Predictive Maintenance")
st.caption("Telemetry → features → anomaly + RUL + failure probability → maintenance action")

try:
    telemetry, bundle = load_data(settings.DATA_PATH, settings.MODEL_PATH)
except FileNotFoundError as exc:
    st.error(f"Model bundle missing: {exc}")
    st.code("python src/simulator.py --machines 8 --cycles 900\npython src/train.py")
    st.stop()
except FileNotFoundError:
    st.error("Telemetry dataset missing.")
    st.code("python src/simulator.py --machines 8 --cycles 900")
    st.stop()

assessments = build_assessments(telemetry, bundle)

# ---------------------------------------------------------------- fleet view
st.subheader("Fleet risk")
fleet = pd.DataFrame(
    [
        {
            "Machine": item["machine_id"],
            "Cycle": item["cycle"],
            "Failure probability": item["failure_probability"],
            "Risk band": item["risk_band"],
            "RUL (days)": item["estimated_rul_days"],
        }
        for item in assessments
    ]
)
st.dataframe(
    fleet.style.format(
        {"Failure probability": "{:.1%}", "RUL (days)": "{:.1f}"},
        background_gradient=pd.DataFrame(
            [[BAND_COLORS.get(band, "#ffffff")] for band in fleet["Risk band"]],
            index=fleet.index,
        ),
    ),
    use_container_width=True,
    hide_index=True,
)

# ------------------------------------------------------------- machine panel
machine_ids = [item["machine_id"] for item in assessments]
selected = st.sidebar.selectbox("Machine", machine_ids)
assessment = next(item for item in assessments if item["machine_id"] == selected)

band = assessment["risk_band"]
left, middle, right = st.columns(3)
left.metric("Failure probability", f"{assessment['failure_probability']:.1%}")
middle.metric("Estimated failure in", f"{assessment['estimated_rul_days']:.1f} days")
right.metric("Risk band", band)

st.markdown(
    f"<div style='padding:12px;border-radius:8px;background:{BAND_COLORS.get(band, '#eee')}"
    f";color:white;font-weight:600'>Recommended: {assessment['recommended_action']}</div>",
    unsafe_allow_html=True,
)

st.subheader("Primary indicators")
st.dataframe(
    pd.DataFrame(assessment["all_indicators"])[
        ["sensor", "component", "current_mean", "healthy_mean", "sigma", "direction", "anomalous"]
    ],
    use_container_width=True,
    hide_index=True,
)
