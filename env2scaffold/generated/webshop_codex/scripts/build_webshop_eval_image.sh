#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMAGE="${IMAGE:-webshop-qwen-eval:latest}"

docker build \
  -f "$ROOT/env2scaffold/generated/webshop_codex/docker/Dockerfile.webshop-eval" \
  -t "$IMAGE" \
  "$ROOT"

echo "Built $IMAGE"
