from google.cloud import firestore


class FirestoreWriter:
    def __init__(self, db=None, mock: bool = False) -> None:
        self._mock = mock
        self._db = db if db is not None else (None if mock else firestore.Client())

    def write(self, alert: dict) -> None:
        if self._mock:
            return
        self._db.collection("alerts").add(alert)

    def get_recent(self, limit: int = 20) -> list[dict]:
        if self._mock:
            return []
        docs = (
            self._db.collection("alerts")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [doc.to_dict() for doc in docs]
