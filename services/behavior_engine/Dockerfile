FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . behavior_engine/
ENV PYTHONPATH=/app
CMD ["sh", "-c", "uvicorn behavior_engine.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
