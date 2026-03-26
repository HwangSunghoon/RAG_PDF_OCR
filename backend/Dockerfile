# ---- backend/Dockerfile ----
FROM python:3.11-slim

# 시스템 패키지: tesseract(ko+eng), poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 코드
COPY server.py /app/server.py

ENV OCR_LANG="kor+eng"
ENV EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV LLM_LOCAL_MODEL="google/flan-t5-base"
# llama.cpp OpenAI 호환 엔드포인트 주소 (compose에서 주입)
# ENV LLM_HTTP_URL="http://llm:8080"

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
