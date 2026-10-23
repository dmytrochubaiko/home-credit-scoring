FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY Production/ ./Production/
COPY Models/ ./Models/

ENV PYTHONPATH=/app/Production

EXPOSE 8000

CMD ["uvicorn", "Production.app:app", "--host", "0.0.0.0", "--port", "8000"]
