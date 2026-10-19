# HCAD-RAG on NVIDIA A100 (Akash / any CUDA cloud)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3.12 python3-pip git curl lsof openssh-server \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
# GPU torch: CUDA 12.6 wheels match A100 provider drivers
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip3 install -r requirements.txt
COPY . /workspace/HCAD-RAG
WORKDIR /workspace/HCAD-RAG

# Ollama runs inside the container; models live on the persistent volume
ENV OLLAMA_MODELS=/data/ollama
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 22 11434
ENTRYPOINT ["/start.sh"]
