import asyncio
import json
import queue
import threading
import streamlit as st
import websockets

_alert_queue: queue.Queue = queue.Queue()


def _ws_listener(ws_url: str) -> None:
    async def _run():
        try:
            async with websockets.connect(ws_url) as ws:
                async for message in ws:
                    _alert_queue.put(json.loads(message))
        except Exception:
            pass

    asyncio.run(_run())


def render(ws_url: str) -> None:
    st.subheader("Live Feed")

    if "ws_thread_started" not in st.session_state:
        t = threading.Thread(target=_ws_listener, args=(ws_url,), daemon=True)
        t.start()
        st.session_state["ws_thread_started"] = True

    status = st.info("Waiting for detections... Start `frame_publisher/main.py` on your laptop.")
    alert_box = st.empty()

    if not _alert_queue.empty():
        alert = _alert_queue.get()
        status.empty()
        alert_box.success(f"**ALERT** — {alert.get('alert_text', 'Customer needs assistance')}")
        with st.expander("Alert details"):
            st.json(alert)

    st.caption(f"WebSocket: `{ws_url}`")
