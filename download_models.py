from transformers import AutoProcessor, BarkModel
import torch

# Мы будем работать на CPU во время сборки
device = "cpu"

print('Downloading base model and processor...')
processor = AutoProcessor.from_pretrained('suno/bark')
model = BarkModel.from_pretrained('suno/bark').to(device)

print('Forcing download of all Russian voice presets via dummy generation...')

# Короткий текст для пробной генерации
text_prompt = "test"

for i in range(9):
    preset_name = f'v2/ru_speaker_{i}'
    print(f'Downloading preset: {preset_name}')
    
    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    # 1. Готовим входные данные С УКАЗАНИЕМ ГОЛОСА через процессор
    inputs = processor(text_prompt, voice_preset=preset_name, return_tensors="pt").to(device)
    
    # 2. Выполняем генерацию. Теперь voice_preset уже "внутри" inputs,
    # и model.generate() его не видит как лишний аргумент.
    model.generate(**inputs, max_new_tokens=1)

print('All downloads complete.')