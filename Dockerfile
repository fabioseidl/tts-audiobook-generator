FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Accept the Coqui Public Model License non-interactively so the XTTS download
# doesn't block on a stdin prompt in this headless container.
ENV COQUI_TOS_AGREED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
