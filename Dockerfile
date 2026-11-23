FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DB_PATH=/app/local_data/drainage.db

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

RUN mkdir -p /app/local_data
VOLUME ["/app/local_data"]

EXPOSE 8000

# 入口脚本负责建表并以 waitress 启动 WSGI 服务
CMD ["python", "-m", "app.serve"]
