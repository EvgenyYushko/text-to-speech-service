from transformers import AutoProcessor, BarkModel

print('Downloading base model...')
processor = AutoProcessor.from_pretrained('suno/bark')
BarkModel.from_pretrained('suno/bark')

print('Downloading all Russian voice presets...')
for i in range(9):
    preset_name = f'v2/ru_speaker_{i}'
    print(f'Downloading preset: {preset_name}')
    processor._load_voice_preset(preset_name)

print('All downloads complete.')