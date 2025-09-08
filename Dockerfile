# --- ���� 1: "������� ������" ---
FROM python:3.10-slim as builder

# ������������� git � git-lfs �� ������, ���� ��� �����������
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# ��������� �������� ����� ��� ����� ���� Hugging Face
ENV HF_HOME=/hf_cache

# ��������� ������� ���������� ������.
# ������ ����� ��������� � /hf_cache/hub/models--suno--bark/...
RUN python -c "from transformers import AutoProcessor, BarkModel; AutoProcessor.from_pretrained('suno/bark'); BarkModel.from_pretrained('suno/bark')"


# --- ���� 2: "��������� ����� ����������" ---
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- ����������� ����� ---
# ���������� ���� ��� �����������: �� �������� ��� ����� ����
# �� /hf_cache (��������) � /root/.cache/huggingface (����������)
COPY --from=builder /hf_cache /root/.cache/huggingface

# �������� �������� ��� ������ ����������
COPY main.py .

# ������� ��� ������� ���-�������
CMD uvicorn main:app --host 0.0.0.0 --port $PORT