FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app_demo ./app_demo
COPY backend ./backend
COPY models ./models
COPY outputs/chen_real_mfi_dataset ./outputs/chen_real_mfi_dataset
COPY outputs/metadata ./outputs/metadata

# Runtime-created data lives here. Render can mount a persistent disk at /app/outputs.
RUN mkdir -p /app/outputs/uploaded_images /app/outputs/manual_label_library

EXPOSE 8765
CMD ["sh", "-c", "python app_demo/server.py --host 0.0.0.0 --port ${PORT:-8765}"]
