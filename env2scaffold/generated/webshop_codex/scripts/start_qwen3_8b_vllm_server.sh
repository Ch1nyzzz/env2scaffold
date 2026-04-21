#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
PORT="${PORT:-8001}"
GPU_DEVICES="${GPU_DEVICES:-2,3}"
TP="${TP:-2}"

docker rm -f qwen3-8b-vllm >/dev/null 2>&1 || true
docker run -d --name qwen3-8b-vllm \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --ipc=host \
  --network host \
  -v "$MODEL:/models/Qwen3-8B:ro" \
  -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}" \
  vllm/vllm-openai:v0.18.0 \
  /models/Qwen3-8B \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --trust-remote-code

echo "Started qwen3-8b-vllm on port $PORT. Logs:"
echo "  docker logs -f qwen3-8b-vllm"
