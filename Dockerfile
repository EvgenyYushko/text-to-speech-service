# --- щрюо 1: "яанпыхй" ---
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# --- хглемемхе 1: хяонкэгсел опюбхкэмсч оепелеммсч ---
ENV HF_HOME=/hf_cache

RUN python -c "from transformers import AutoProcessor, BarkModel; processor = AutoProcessor.from_pretrained('suno/bark'); model = BarkModel.from_pretrained('suno/bark'); inputs = processor('test', voice_preset='v2/ru_speaker_6', return_tensors='pt'); model.generate(**inputs)"

# --- щрюо 2: "тхмюкэмши напюг" ---
FROM python:3.10-slim

WORKDIR /app

# --- хглемемхе 2: хяонкэгсел опюбхкэмсч оепелеммсч ---
ENV HF_HOME=/app/hf_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# йНОХПСЕЛ ЙЩЬ хг /hf_cache (ЩРЮО 1) б /app/hf_cache (ТХМЮКЭМШИ НАПЮГ)
COPY --from=builder /hf_cache /app/hf_cache

COPY main.py .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}