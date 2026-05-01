import os
import streamlit as st
from dotenv import load_dotenv
from tabs import live_feed, alert_history, model_comparison

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
REGION = os.getenv("GCP_REGION", "northamerica-northeast1")
WS_URL = os.getenv("NOTIFICATION_WS_URL", "ws://localhost:8083/ws")

st.set_page_config(page_title="HomeVision AI", layout="wide")
st.title("HomeVision AI — Customer Assistance Detection")
st.caption("Home Depot Canada | Real-Time Computer Vision Pipeline")

tab1, tab2, tab3 = st.tabs(["Live Feed", "Alert History", "Model Comparison"])

with tab1:
    live_feed.render(ws_url=WS_URL)

with tab2:
    alert_history.render(project_id=PROJECT_ID)

with tab3:
    model_comparison.render(project_id=PROJECT_ID, region=REGION)
