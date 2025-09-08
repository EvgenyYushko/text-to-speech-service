# --- ЭТАП 1: "СБОРЩИК МОДЕЛИ И ВСЕХ ГОЛОСОВ" ---
FROM python:3.10-slim as builder

# Устанавливаем необходимые системные пакеты
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# Указываем корневую папку для кэша
ENV HF_HOME=/hf_cache

# --- ГЛАВНОЕ ИЗМЕНЕНИЕ ---
# Запускаем скрипт, который скачает модель и ВСЕ русские голоса (от 0 до 8)
RUN python -c "\
from transformers import AutoProcessor, BarkModel; \
print('Downloading base model...'); \
processor = AutoProcessor.from_pretrained('suno/bark'); \
BarkModel.from_pretrained('suno/bark'); \
print('Downloading all Russian voice presets...'); \
for i in range(9): \
    preset_name = f'v2/ru_speaker_{i}'; \
    print(f'Downloading preset: {preset_name}'); \
    processor._load_voice_preset(preset_name); \
print('All downloads complete.')"

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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]