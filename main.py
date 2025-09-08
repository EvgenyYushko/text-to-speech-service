from fastapi import FastAPI, Response
from transformers import AutoProcessor, BarkModel
import soundfile as sf
import numpy as np
import io
import torch
from contextlib import asynccontextmanager

# --- 1. Создаем "хранилище" для нашей модели ---
# Оно будет пустым до тех пор, пока модель не загрузится
ml_models = {}

# --- 2. Создаем асинхронную функцию для загрузки модели ---
# FastAPI будет вызывать ее ПОСЛЕ запуска сервера
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код до 'yield' выполнится при старте
    print("Сервер запущен. Начинаю загрузку модели...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Загружаем процессор и модель и кладем их в наше "хранилище"
    ml_models["processor"] = AutoProcessor.from_pretrained("suno/bark")
    ml_models["model"] = BarkModel.from_pretrained("suno/bark").to(device)
    
    print(f"Модель успешно загружена на устройстве: {device}")
    
    yield
    
    # Код после 'yield' выполнится при остановке сервера (очистка)
    ml_models.clear()

# --- 3. Передаем эту функцию в FastAPI ---
app = FastAPI(lifespan=lifespan)

# --- 4. Эндпоинт теперь берет модель из "хранилища" ---
@app.post("/generate-audio/")
async def generate_audio_endpoint(text_input: dict):
    # Проверяем, загружена ли уже модель
    if "model" not in ml_models or "processor" not in ml_models:
        return Response(content='{"error": "Модель все еще загружается, попробуйте через несколько минут."}', status_code=503)

    # Берем готовые процессор и модель
    processor = ml_models["processor"]
    model = ml_models["model"]

    # ... (вся остальная логика эндпоинта остается абсолютно такой же) ...
    text = text_input.get("text")
    if not text:
        return Response(content='{"error": "Текст не был предоставлен в запросе."}', status_code=400)
    
    voice_preset = text_input.get("voice_preset")
    fine_temp = text_input.get("fine_temperature", 0.5) 
    coarse_temp = text_input.get("coarse_temperature", 0.7)

    print(f"Запрос: '{text}', Пресет: {voice_preset or 'auto'}")
    
    inputs = processor(text, voice_preset=voice_preset, return_tensors="pt").to(model.device)
    
    speech_output = model.generate(**inputs, do_sample=True, fine_temperature=fine_temp, coarse_temperature=coarse_temp)
    
    sampling_rate = model.generation_config.sample_rate
    audio_waveform = speech_output.cpu().numpy().squeeze()
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_waveform, sampling_rate, format='WAV')
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")

@app.get("/")
def read_root():
    # Можно добавить проверку, загружена ли модель
    status = "запущен, модель загружается..." if "model" not in ml_models else "запущен и готов к работе!"
    return {"status": f"Сервер {status}"}