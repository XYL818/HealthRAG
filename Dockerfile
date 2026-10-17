# 推理全部走云端 API，镜像里**不需要 torch**，所以能压得很小。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    TZ=Asia/Shanghai

WORKDIR /app

# curl 只给 HEALTHCHECK 用
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py ./
COPY indexing/ ./indexing/
COPY rag/ ./rag/
COPY app/ ./app/
COPY web/ ./web/

# 索引不进镜像（约 170MB），改用 docker-compose 的卷挂载。
# 好处：改索引不用重建镜像，切消融配置只要换 INDEX_DIR。

# 预生成 jieba 词典缓存，避免容器起来后第一个请求卡几秒
RUN python -c "import jieba; jieba.initialize()"

# 不用 root 跑
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
