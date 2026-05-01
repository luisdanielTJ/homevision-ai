import streamlit as st
import pandas as pd
import altair as alt

# Sample data — replaced with live Vertex AI Experiments query after models are trained
_SAMPLE = pd.DataFrame({
    "Model": ["AutoML", "YOLOv8n"],
    "mAP@0.5": [0.78, 0.91],
    "Precision": [0.81, 0.93],
    "Recall": [0.76, 0.89],
    "Latency p50 (ms)": [12.0, 8.0],
    "Latency p95 (ms)": [28.0, 14.0],
})


def render(project_id: str, region: str) -> None:
    st.subheader("Model Comparison: AutoML vs YOLOv8n")
    st.caption("Metrics from the last Vertex AI Pipeline run.")

    col1, col2 = st.columns(2)
    with col1:
        chart = (
            alt.Chart(_SAMPLE)
            .mark_bar()
            .encode(
                x=alt.X("Model:N"),
                y=alt.Y("mAP@0.5:Q", scale=alt.Scale(domain=[0, 1])),
                color="Model:N",
            )
            .properties(title="mAP@0.5")
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        chart2 = (
            alt.Chart(_SAMPLE)
            .mark_bar()
            .encode(
                x=alt.X("Model:N"),
                y=alt.Y("Latency p50 (ms):Q"),
                color="Model:N",
            )
            .properties(title="Inference Latency p50 (ms)")
        )
        st.altair_chart(chart2, use_container_width=True)

    st.dataframe(_SAMPLE.set_index("Model"), use_container_width=True)

    if project_id and project_id != "your-project-id":
        st.caption(
            "To load live metrics: query Vertex AI Experiments and replace `_SAMPLE` "
            "in `dashboard/tabs/model_comparison.py`."
        )
