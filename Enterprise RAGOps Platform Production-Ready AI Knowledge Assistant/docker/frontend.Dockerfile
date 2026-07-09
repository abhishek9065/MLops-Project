FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
