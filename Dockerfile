# --- ���� 1: "������� ������ � ���� �������" ---
FROM python:3.10-slim as builder

# ������������� ����������� ��������� ������
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers torch accelerate

# ��������� �������� ����� ��� ����
ENV HF_HOME=/hf_cache

# --- ������� ��������� ---
# �������� ��� ����� ������ ��� ���������� � �����
COPY download_models.py .
# ��������� ���� ������. ��� ������, ����� � �������.
RUN python download_models.py

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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", $PORT]