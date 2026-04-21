#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMAGE="${IMAGE:-webshop-qwen-eval:latest}"
EPISODES="${EPISODES:-100}"
MAX_STEPS="${MAX_STEPS:-15}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-qwen3-8b}"

docker run --rm --network host \
  -v "$ROOT:/workspace/env-aug" \
  "$IMAGE" \
  python /workspace/env-aug/env2scaffold/generated/webshop_codex/evaluation/qwen3_8b_webshop_ab.py \
    --backend openai \
    --openai-base-url "$OPENAI_BASE_URL" \
    --openai-model "$OPENAI_MODEL" \
    --mode both \
    --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" \
    --webshop-root /workspace/env-aug/verl-agent/agent_system/environments/env_package/webshop/webshop \
    --output /workspace/env-aug/env2scaffold/generated/webshop_codex/evaluation/qwen3_8b_webshop_ab_results.json
