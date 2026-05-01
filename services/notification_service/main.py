import os
import uvicorn
from dotenv import load_dotenv
from notification_service.firestore_writer import FirestoreWriter
from notification_service.app import create_app

load_dotenv()

MOCK = os.getenv("MOCK_FIRESTORE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))
app = create_app(writer=FirestoreWriter(mock=MOCK))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
