# --- ЭТАП 1: "СБОРЩИК МОДЕЛИ" ---
# Мы используем базовый образ Python и называем этот этап 'builder'
FROM python:3.10-slim as builder

# Устанавливаем только те библиотеки, которые нужны для скачивания
RUN pip install --no-cache-dir transformers torch accelerate

# Указываем, куда скачивать модель
ENV HF_HOME=/hf_cache

# Запускаем команду скачивания. Это самая долгая часть сборки.
# RUN --mount=type=cache... — это специальная команда для кэширования,
# чтобы не скачивать модель при каждой небольшой пересборке.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "from transformers import AutoProcessor, BarkModel; AutoProcessor.from_pretrained('suno/bark'); BarkModel.from_pretrained('suno/bark')"

# На этом первый этап закончен. У нас есть образ, в котором по пути /hf_cache/.cache/huggingface лежит модель.


# --- ЭТАП 2: "ФИНАЛЬНЫЙ ОБРАЗ ПРИЛОЖЕНИЯ" ---
# Начинаем с чистого листа, используя тот же базовый образ
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- ГЛАВНОЕ ИЗМЕНЕНИЕ ---
# Копируем папку с моделью, скачанную на ЭТАПЕ 1,
# в финальный образ.
COPY --from=builder /hf_cache/.cache/huggingface /root/.cache/huggingface

# Копируем основной код нашего приложения
COPY main.py .

# Команда для запуска веб-сервера
CMD uvicorn main:app --host 0.0.0.0 --port $PORT