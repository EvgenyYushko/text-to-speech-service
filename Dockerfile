# --- ���� 1: "������� ������ � ���� �������" ---
FROM python:3.10-slim as builder

# ������������� ����������� ��������� ������
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# ��������� �������� ����� ��� ����
ENV HF_HOME=/hf_cache

# --- ������� ��������� ---
# ��������� ������, ������� ������� ������ � ��� ������� ������ (�� 0 �� 8)
RUN python -c "\
from transformers import AutoProcessor, BarkModel; \
print('Downloading base model...'); \
processor = AutoProcessor.from_pretrained('suno/bark'); \
BarkModel.from_pretrained('suno/bark'); \
print('Downloading all Russian voice presets...'); \
for i in range(9): \
    preset_name = f'v2/ru_speaker_{i}'; \
    print(f'Downloading preset: {preset_name}'); \
    processor._load_voice_preset(preset_name); \
print('All downloads complete.')"

# --- ���� 2: "��������� ����� ����������" ---
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# �������� ��� ����� ���� (� ������� � ����� ��������)
COPY --from=builder /hf_cache /root/.cache/huggingface

# �������� �������� ���
COPY main.py .

# ������� ��� ������� ���-�������
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]