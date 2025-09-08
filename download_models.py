from transformers import AutoProcessor, BarkModel
import torch

device = "cpu"

print('Downloading base model and processor...')
processor = AutoProcessor.from_pretrained('suno/bark')
model = BarkModel.from_pretrained('suno/bark').to(device)

print('Forcing download of all Russian voice presets via dummy generation...')

text_prompt = "test"
inputs = processor(text_prompt, return_tensors="pt").to(device)

for i in range(10):
    preset_name = f'v2/ru_speaker_{i}'
    print(f'Downloading preset: {preset_name}')
   
    model.generate(**inputs, voice_preset=preset_name, max_new_tokens=1)

print('All downloads complete.')