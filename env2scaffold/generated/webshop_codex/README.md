# WebShop env2scaffold Output

Output root:
`/data/home/yuhan/env-aug/env2scaffold/generated/webshop_codex`

Benchmark implementation:
`/data/home/yuhan/env-aug/verl-agent/agent_system/environments/env_package/webshop`

This scaffold targets WebShop through the vendored `verl-agent` WebShop
environment. It does not modify or overwrite the existing ALFWorld
env2scaffold outputs under `env2scaffold/benchmark_spec`,
`env2scaffold/augmentation`, or related root-level directories.

## What Was Generated

- `benchmark_spec/`: WebShop interface, reward, observable channels, and leakage boundaries
- `probing/feedback_catalog.json`: feedback/ambiguity catalog from WebShop code inspection
- `audit/`: candidate ambiguity clusters
- `augmentation/augmented_env.py`: standalone observation wrapper
- `oracle_test/`: oracle and unit-test plan
- `verification/`: deterministic wrapper verification

## Wrapper

`augmentation/augmented_env.py` exports:

- `AugmentedWebShopEnv`
- `augment_observation`
- `wrap_webshop_env`

The wrapper composes around a native `WebAgentTextEnv`-compatible object:

```python
from augmentation.augmented_env import AugmentedWebShopEnv

env = AugmentedWebShopEnv(native_webshop_env)
obs, info = env.reset()
obs, reward, done, info = env.step("click[not visible]")
```

It appends `Env feedback:` text for malformed actions, actions outside visible
affordances, page-mode cues, and visible product option state. It preserves the
wrapped environment's action execution, reward, done flag, and state transition.

## Commands

Run verification:

```bash
cd /data/home/yuhan/env-aug/env2scaffold/generated/webshop_codex
python verification/verify_runner.py
```

Run individual layers:

```bash
python verification/layer1_benchmark_native.py
python verification/layer2_diagnostic_unit.py
python verification/layer3_non_regression.py
```

Layer 1 is marked deferred because the current active shell lacks the full
WebShop runtime dependencies. Layer 2 and Layer 3 use deterministic fake
WebShop-compatible environments to verify the generated wrapper logic.

Run Qwen3-8B baseline vs augmented WebShop evaluation after installing WebShop
runtime dependencies and downloading WebShop data/indexes:

```bash
cd /data/home/yuhan/env-aug
CUDA_VISIBLE_DEVICES=2,3 python env2scaffold/generated/webshop_codex/evaluation/qwen3_8b_webshop_ab.py \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --backend vllm \
  --tensor-parallel-size 2 \
  --mode both \
  --episodes 100 \
  --max-steps 15
```

The script writes `evaluation/qwen3_8b_webshop_ab_results.json` with
`success_rate`, `average_task_score`, and per-episode trajectories for baseline
and augmented modes.

Dockerized setup:

```bash
cd /data/home/yuhan/env-aug

# 1. Build a Python3.10 WebShop client image.
bash env2scaffold/generated/webshop_codex/scripts/build_webshop_eval_image.sh

# 2. Download WebShop data and build Lucene index on the mounted repo.
DATA_SIZE=small bash env2scaffold/generated/webshop_codex/scripts/setup_webshop_data.sh

# 3. Start Qwen3-8B as an OpenAI-compatible vLLM server on GPUs 2,3.
#    Defaults to port 8001 to avoid the existing local vLLM service on 8000.
GPU_DEVICES=2,3 bash env2scaffold/generated/webshop_codex/scripts/start_qwen3_8b_vllm_server.sh

# 4. Run baseline vs augmented evaluation through the server.
EPISODES=100 bash env2scaffold/generated/webshop_codex/scripts/run_qwen3_8b_webshop_ab_docker.sh
```

The Docker path keeps WebShop's Python<=3.10 dependency stack separate from the
host and from the existing ALFWorld environment.

## Qwen3-8B A/B Result

Run completed with `EPISODES=100`, `MAX_STEPS=15`, WebShop small data/index,
Qwen3-8B served by Dockerized vLLM on GPUs 2,3, and OpenAI-compatible client
calls from the WebShop eval container.

| mode | task completion rate | successes | average task score | average steps |
| --- | ---: | ---: | ---: | ---: |
| baseline | 10.0% | 10/100 | 0.2654 | 12.06 |
| augmented wrapper | 32.0% | 32/100 | 0.5275 | 7.51 |

Detailed trajectories and model outputs are in
`evaluation/qwen3_8b_webshop_ab_results.json`.

## GRPO Training Scripts

WebShop training scripts live under `verl-agent/scripts/`:

```bash
cd /data/home/yuhan/env-aug/verl-agent

# Vanilla GRPO: sparse success reward, no observation augmentation.
bash scripts/run_webshop_vanilla.sh

# Obs-Aug GRPO: generated env2scaffold WebShop observation wrapper,
# sparse success reward.
bash scripts/run_webshop_envtuning.sh

# Full EnvTuning GRPO: obs-aug plus non-leaking progress shaping.
bash scripts/run_webshop_full_envtuning.sh
```

Common overrides:

```bash
MODEL_PATH=/data/home/yuhan/model_zoo/Qwen3-8B \
N_GPUS=8 \
TP=2 \
TRAIN_DATA_SIZE=16 \
VAL_DATA_SIZE=128 \
GROUP_SIZE=8 \
bash scripts/run_webshop_full_envtuning.sh
```

The scripts validate that WebShop small data and Lucene indexes are present.
If they are missing, run:

```bash
cd /data/home/yuhan/env-aug
DATA_SIZE=small bash env2scaffold/generated/webshop_codex/scripts/setup_webshop_data.sh
```

Validation rollouts are kept vanilla for all three variants so reported task
completion metrics remain comparable.

## Integration Note

This is a generated env2scaffold artifact, not a `verl-agent` runtime patch.
After review, integration should import this wrapper from the generated output
or copy it into a WebShop-specific package, then wire it behind an explicit
training flag while keeping validation vanilla.
