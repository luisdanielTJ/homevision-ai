import streamlit as st
import pandas as pd


def render(project_id: str) -> None:
    st.subheader("Recent Alerts")
    if not project_id or project_id == "your-project-id":
        st.warning("Set `GCP_PROJECT_ID` in your `.env` to load live alert history.")
        return
    try:
        from google.cloud import firestore

        db = firestore.Client(project=project_id)
        docs = (
            db.collection("alerts")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(20)
            .stream()
        )
        rows = [doc.to_dict() for doc in docs]
        if rows:
            df = pd.DataFrame(rows)[["timestamp", "camera_id", "dwell_time", "alert_text"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No alerts yet. Start the frame publisher to begin detection.")
    except Exception as e:
        st.error(f"Firestore error: {e}")
