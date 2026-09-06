FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-runpod.txt ./
RUN pip install --no-cache-dir -r requirements-runpod.txt
COPY backend ./backend
COPY handler.py ./handler.py
ENV PYTHONUNBUFFERED=1 SIMV1_DATA=/data
CMD ["python", "-u", "handler.py"]
