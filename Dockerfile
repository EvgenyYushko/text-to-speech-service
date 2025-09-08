# --- ЭТАП 1: "СБОРЩИК МОДЕЛИ И ГОЛОСОВ" ---
FROM python:3.10-slim as builder

# Устанавливаем git и git-lfs на случай, если они понадобятся
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# Указываем корневую папку для кэша
ENV HF_HOME=/hf_cache

# Запускаем команду, которая скачает И модель, И файлы для пресетов
RUN python -c "from transformers import AutoProcessor, BarkModel; \
    AutoProcessor.from_pretrained('suno/bark'); \
    BarkModel.from_pretrained('suno/bark'); \
    AutoProcessor.from_pretrained('suno/bark')._load_voice_preset('v2/ru_speaker_0')"

# --- ЭТАП 2: "ФИНАЛЬНЫЙ ОБРАЗ ПРИЛОЖЕНИЯ" ---
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЮ папку кэша (с моделью и голосами)
COPY --from=builder /hf_cache /root/.cache/huggingface

# Копируем основной код
COPY main.py .

# Команда для запуска веб-сервера
CMD uvicorn main:app --host 0.0.0.0 --port 8080