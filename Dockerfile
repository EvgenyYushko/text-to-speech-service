# --- ЭТАП 1: "СБОРЩИК МОДЕЛИ И ВСЕХ ГОЛОСОВ" ---
FROM python:3.10-slim as builder

# Устанавливаем необходимые системные пакеты
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# Указываем корневую папку для кэша
ENV HF_HOME=/hf_cache

# --- ГЛАВНОЕ ИЗМЕНЕНИЕ ---
# Копируем наш новый скрипт для скачивания в образ
COPY download_models.py .
# Запускаем этот скрипт. Это просто, чисто и надежно.
RUN python download_models.py

# --- ЭТАП 2: "ФИНАЛЬНЫЙ ОБРАЗ ПРИЛОЖЕНИЯ" ---
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЮ папку кэша (с моделью и всеми голосами)
COPY --from=builder /hf_cache /root/.cache/huggingface

# Копируем основной код
COPY main.py .

# Команда для запуска веб-сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", $PORT]