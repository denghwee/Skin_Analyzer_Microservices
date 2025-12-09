FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV WORKERS=1
ENV APP_MODULE=wsgi:app
ENV GUNICORN_TIMEOUT=300

EXPOSE 5000

CMD ["sh", "-c", "gunicorn 'run:create_app()' --bind 0.0.0.0:${PORT} --timeout 300"]
