#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMAGE="${IMAGE:-webshop-qwen-eval:latest}"
DATA_SIZE="${DATA_SIZE:-small}"
WEBSHOP_DIR="/workspace/env-aug/verl-agent/agent_system/environments/env_package/webshop/webshop"

docker run --rm --network host \
  -v "$ROOT:/workspace/env-aug" \
  "$IMAGE" \
  bash -lc "download_if_missing() { local file=\"\$1\"; local url=\"\$2\"; if [ -f \"\$file\" ]; then echo \"Using existing \$file\"; else gdown \"\$url\"; fi; }; \
    cd '$WEBSHOP_DIR' && mkdir -p data && cd data && \
    if [ '$DATA_SIZE' = 'small' ]; then \
      download_if_missing items_shuffle_1000.json https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib && \
      download_if_missing items_ins_v2_1000.json https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; \
    elif [ '$DATA_SIZE' = 'all' ]; then \
      download_if_missing items_shuffle_1000.json https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib && \
      download_if_missing items_ins_v2_1000.json https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu && \
      download_if_missing items_shuffle.json https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB && \
      download_if_missing items_ins_v2.json https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; \
    else \
      echo 'DATA_SIZE must be small or all' >&2; exit 2; \
    fi && \
    download_if_missing items_human_ins.json https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O && \
    cd '$WEBSHOP_DIR/search_engine' && \
    mkdir -p resources resources_100 resources_1k resources_100k indexes && \
    python convert_product_file_format.py && \
    ./run_indexing.sh"
