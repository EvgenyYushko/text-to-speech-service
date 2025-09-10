from fastapi import FastAPI, Response
from transformers import AutoProcessor, BarkModel
import soundfile as sf
import numpy as np
import io
import torch
from contextlib import asynccontextmanager

APP_VERSION = "v1.1 - Indentation Fixed"
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- ВСЕ ОТСТУПЫ ЗДЕСЬ ИСПРАВЛЕНЫ НА 4 ПРОБЕЛА ---
    print(f"Запуск сервера, ВЕРСИЯ: {APP_VERSION}")
    print("Сервер запущен. Начинаю загрузку модели ИЗ ЛОКАЛЬНЫХ ФАЙЛОВ...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ml_models["processor"] = AutoProcessor.from_pretrained("suno/bark", local_files_only=True)
    ml_models["model"] = BarkModel.from_pretrained("suno/bark", local_files_only=True).to(device)
    
    print(f"Модель успешно загружена на устройстве: {device}")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/generate-audio/")
async def generate_audio_endpoint(text_input: dict):  
    if "model" not in ml_models or "processor" not in ml_models:
        return Response(content='{"error": "Модель все еще загружается, попробуйте через несколько минут."}', status_code=503)

    processor = ml_models["processor"]
    model = ml_models["model"]

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
    status = "запущен, модель загружается..." if "model" not in ml_models else "запущен и готов к работе!"
    return {"status": f"Сервер {status}"}