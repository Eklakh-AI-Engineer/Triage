#!/bin/bash
# Boots Ollama + SSH; models/data persist on the mounted volume (/data)
mkdir -p /data/ollama
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /data/ollama.log 2>&1 &
sleep 5
ollama pull llama3.1:8b || echo "model pull failed — check volume"
ollama pull qwen3:14b || true
service ssh start || /usr/sbin/sshd
tail -f /data/ollama.log
