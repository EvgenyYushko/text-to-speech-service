# --- ���� 1: "������� ������" ---
# �� ���������� ������� ����� Python � �������� ���� ���� 'builder'
FROM python:3.10-slim as builder

# ������������� ������ �� ����������, ������� ����� ��� ����������
RUN pip install --no-cache-dir transformers torch accelerate

# ���������, ���� ��������� ������
ENV HF_HOME=/hf_cache

# ��������� ������� ����������. ��� ����� ������ ����� ������.
# RUN --mount=type=cache... � ��� ����������� ������� ��� �����������,
# ����� �� ��������� ������ ��� ������ ��������� ����������.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "from transformers import AutoProcessor, BarkModel; AutoProcessor.from_pretrained('suno/bark'); BarkModel.from_pretrained('suno/bark')"

# �� ���� ������ ���� ��������. � ��� ���� �����, � ������� �� ���� /hf_cache/.cache/huggingface ����� ������.


# --- ���� 2: "��������� ����� ����������" ---
# �������� � ������� �����, ��������� ��� �� ������� �����
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- ������� ��������� ---
# �������� ����� � �������, ��������� �� ����� 1,
# � ��������� �����.
COPY --from=builder /hf_cache/.cache/huggingface /root/.cache/huggingface

# �������� �������� ��� ������ ����������
COPY main.py .

# ������� ��� ������� ���-�������
CMD uvicorn main:app --host 0.0.0.0 --port $PORT