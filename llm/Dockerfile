
FROM ghcr.io/ggerganov/llama.cpp:full

WORKDIR /app

EXPOSE 8080

CMD ["./server", "-m", "/models/weights.gguf", "--port", "8080", "--host", "0.0.0.0"]

