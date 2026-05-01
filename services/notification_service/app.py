import base64
import json
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from notification_service.firestore_writer import FirestoreWriter


def create_app(writer: FirestoreWriter) -> FastAPI:
    app = FastAPI(title="HomeVision Notification Service")
    connected_clients: list[WebSocket] = []

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        connected_clients.append(ws)
        try:
            while True:
                await ws.receive_text()  # keep-alive loop
        except WebSocketDisconnect:
            connected_clients.remove(ws)

    @app.post("/pubsub/push")
    async def pubsub_push(request: Request):
        body = await request.json()
        data = base64.b64decode(body["message"]["data"]).decode()
        alert = json.loads(data)
        writer.write(alert)

        dead: list[WebSocket] = []
        for ws in connected_clients:
            try:
                await ws.send_text(json.dumps(alert))
            except Exception:
                dead.append(ws)
        for ws in dead:
            connected_clients.remove(ws)

        return {"status": "ok"}

    @app.get("/alerts")
    def get_alerts(limit: int = 20):
        return writer.get_recent(limit)

    return app
