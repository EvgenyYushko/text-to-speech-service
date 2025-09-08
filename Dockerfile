# --- ЭТАП 1: "СБОРЩИК МОДЕЛИ" ---
FROM python:3.10-slim as builder

# Устанавливаем git и git-lfs на случай, если они понадобятся
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# Указываем корневую папку для всего кэша Hugging Face
ENV HF_HOME=/hf_cache

# Запускаем команду скачивания модели.
# Модель будет сохранена в /hf_cache/hub/models--suno--bark/...
RUN python -c "from transformers import AutoProcessor, BarkModel; AutoProcessor.from_pretrained('suno/bark'); BarkModel.from_pretrained('suno/bark')"


# --- ЭТАП 2: "ФИНАЛЬНЫЙ ОБРАЗ ПРИЛОЖЕНИЯ" ---
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
# Правильный путь для копирования: мы копируем ВСЮ папку кэша
# из /hf_cache (источник) в /root/.cache/huggingface (назначение)
COPY --from=builder /hf_cache /root/.cache/huggingface

# Копируем основной код нашего приложения
COPY main.py .

# Команда для запуска веб-сервера
CMD uvicorn main:app --host 0.0.0.0 --port $PORT