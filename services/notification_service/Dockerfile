FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . notification_service/
ENV PYTHONPATH=/app
CMD ["sh", "-c", "uvicorn notification_service.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
