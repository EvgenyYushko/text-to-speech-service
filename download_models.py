from transformers import AutoProcessor, BarkModel
import torch

# Мы будем работать на CPU во время сборки
device = "cpu"
# --- ВЫБЕРИТЕ ГОЛОС ЗДЕСЬ ---
# Укажите один, нужный вам голос
voice_to_preload = "v2/ru_speaker_6" 

print('Downloading base model and processor...')
processor = AutoProcessor.from_pretrained('suno/bark')
model = BarkModel.from_pretrained('suno/bark').to(device)

print(f"Forcing download of a single voice preset: {voice_to_preload}")

# Короткий текст для пробной генерации
text_prompt = "test"
inputs = processor(text_prompt, voice_preset=voice_to_preload, return_tensors="pt").to(device)

# Выполняем очень короткую генерацию, чтобы скачать все файлы для этого голоса
model.generate(**inputs)

print('All downloads complete.')