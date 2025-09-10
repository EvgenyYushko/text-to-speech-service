from fastapi import FastAPI, Response
from transformers import AutoProcessor, BarkModel
import soundfile as sf
import numpy as np
import io
import torch

# --- УБИРАЕМ LIFESPAN И ГЛОБАЛЬНЫЙ СЛОВАРЬ ---
# Вместо этого мы будем полагаться на внутреннее кэширование transformers

# --- ЗАГРУЖАЕМ МОДЕЛЬ И ПРОЦЕССОР ПРИ СТАРТЕ ---
# Это вернет нас к проблеме долгого запуска, НО мы ее решим!
print("Предварительная загрузка модели (может быть долгой)...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark", local_files_only=True)
model = BarkModel.from_pretrained("suno/bark", local_files_only=True).to(device)
print(f"Модель предварительно загружена на устройстве: {device}")

# Создаем приложение
app = FastAPI()

@app.post("/generate-audio/")
async def generate_audio_endpoint(text_input: dict):
    # Модель и процессор уже загружены в память при старте контейнера
    # и доступны здесь напрямую.
    
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
    return {"status": "Сервер запущен и готов к работе!"}