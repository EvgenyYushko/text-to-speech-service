from fastapi import FastAPI, Response
from transformers import AutoProcessor, BarkModel
import soundfile as sf
import numpy as np
import io
import torch

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Загрузка процессора и модели Bark...")

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)

print(f"Модель загружена и работает на устройстве: {device}")

@app.post("/generate-audio/")
async def generate_audio_endpoint(text_input: dict):
    text = text_input.get("text")
    if not text:
        return Response(content='{"error": "Текст не был предоставлен в запросе."}', status_code=400, media_type="application/json")
  
    voice_preset = text_input.get("voice_preset")
    fine_temp = text_input.get("fine_temperature", 0.5) 
    coarse_temp = text_input.get("coarse_temperature", 0.7)

    print(f"Запрос: '{text}', Пресет: {voice_preset or 'auto'}, FineTemp: {fine_temp}, CoarseTemp: {coarse_temp}")
    
    inputs = processor(text, voice_preset=voice_preset, return_tensors="pt").to(device)
 
    speech_output = model.generate(
        **inputs, 
        do_sample=True, 
        fine_temperature=fine_temp, 
        coarse_temperature=coarse_temp
    )

    sampling_rate = model.generation_config.sample_rate
    audio_waveform = speech_output.cpu().numpy().squeeze()
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_waveform, sampling_rate, format='WAV')
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")

@app.get("/")
def read_root():
    return {"status": "Сервер для модели Bark запущен"}