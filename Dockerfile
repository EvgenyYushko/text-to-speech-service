# --- ���� 1: "������� ������ � �������" ---
FROM python:3.10-slim as builder

# ������������� git � git-lfs �� ������, ���� ��� �����������
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# ��������� �������� ����� ��� ����
ENV HF_HOME=/hf_cache

# ��������� �������, ������� ������� � ������, � ����� ��� ��������
RUN python -c "from transformers import AutoProcessor, BarkModel; \
    AutoProcessor.from_pretrained('suno/bark'); \
    BarkModel.from_pretrained('suno/bark'); \
    AutoProcessor.from_pretrained('suno/bark')._load_voice_preset('v2/ru_speaker_0')"

# --- ���� 2: "��������� ����� ����������" ---
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# �������� ��� ����� ���� (� ������� � ��������)
COPY --from=builder /hf_cache /root/.cache/huggingface

# �������� �������� ���
COPY main.py .

# ������� ��� ������� ���-�������
CMD uvicorn main:app --host 0.0.0.0 --port 8080