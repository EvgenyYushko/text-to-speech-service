# --- ЭТАП 1: "СБОРЩИК МОДЕЛИ И ГОЛОСОВ" ---
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# Указываем, куда скачивать кэш
ENV TRANSFORMERS_CACHE=/hf_cache
RUN python -c "from transformers import AutoProcessor, BarkModel; processor = AutoProcessor.from_pretrained('suno/bark'); model = BarkModel.from_pretrained('suno/bark'); inputs = processor('test', voice_preset='v2/ru_speaker_6', return_tensors='pt'); model.generate(**inputs)"

# --- ЭТАП 2: "ФИНАЛЬНЫЙ ОБРАЗ ПРИЛОЖЕНИЯ" ---
FROM python:3.10-slim

WORKDIR /app

# --- САМОЕ ВАЖНОЕ ИЗМЕНЕНИЕ ---
# Устанавливаем переменную окружения, чтобы transformers ЗНАЛ, где искать кэш
ENV TRANSFORMERS_CACHE=/app/hf_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем наш кэш ИЗ /hf_cache (этап 1) В /app/hf_cache (финальный образ)
COPY --from=builder /hf_cache /app/hf_cache

COPY main.py .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}