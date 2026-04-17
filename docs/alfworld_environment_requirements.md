# ALFWorld Training Environment Requirements

This document records the concrete environment expected by the ALFWorld augmentation and GRPO training code in this repository.

## 1. System Requirements

- OS: Linux x86_64
- Tested kernel/platform: `Linux-6.8.0-107-generic-x86_64-with-glibc2.39`
- GPU: NVIDIA H100 PCIe
- Tested driver: `580.126.20`
- Recommended GPU count for the provided training script: 8 GPUs
- Local scratch space for Triton/cache/logs/checkpoints

## 2. Python Runtime

- Python: `3.12.3`
- `pip` environment with editable install support
- Recommended: isolated virtualenv or conda env dedicated to this repo

## 3. Python Packages

Observed working versions on the current node:

- `torch==2.7.0`
- `textworld==1.7.0`
- `alfworld==0.4.2`
- `openai==2.30.0`
- `pyarrow==21.0.0`
- `pandas==2.3.3`
- `numpy==2.2.6`

Also required by this code path:

- vendored `verl` installed with `pip install -e ./verl`
- ALFWorld/TextWorld dependencies needed to load `game.tw-pddl`

Minimum import check:

```bash
python -c "import torch, textworld, alfworld, openai, pyarrow, pandas, numpy"
```

## 4. Repository Layout Assumptions

- Repo root: `/path/to/env-aug`
- ALFWorld augmentation code: `env2scaffold/`
- Agent RL code: `AWorld-RL/EnvTuning/`
- Vendored trainer package: `AWorld-RL/EnvTuning/verl/`

`AWorld-RL` is now vendored into the monorepo. Do not run submodule initialization for `verl`; install the local copy instead.

## 5. Required Data Paths

- ALFWorld game files:
  - `~/.cache/alfworld/json_2.1.1/valid_seen/.../game.tw-pddl`
- Probing trajectories used to build training parquet:
  - `env2scaffold/probing/trajectories/*.json`
- Generated training data:
  - `AWorld-RL/EnvTuning/data/alfworld_train.parquet`
  - `AWorld-RL/EnvTuning/data/alfworld_val.parquet`

Each dataset row must include:

- `prompt`
- `reward_model.style=interaction`
- `extra_info.interaction_kwargs.game_file`
- optional `max_episode_steps`
- optional `use_augmented_env`

## 6. Environment Variables and Script Paths

Review and set these before training:

- `PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/verl:$PYTHONPATH"`
- `TRITON_CACHE_DIR`
- `MODEL`
- `DATA_DIR`
- `USER_ROOT_PATH`
- `TENSORBOARD_DIR`

Primary ALFWorld training files:

- `AWorld-RL/EnvTuning/env_tuning/interaction/alfworld_interaction.py`
- `AWorld-RL/EnvTuning/env_tuning/alfworld_reward.py`
- `AWorld-RL/EnvTuning/env_tuning/config/alfworld_grpo_stage1.yaml`
- `AWorld-RL/EnvTuning/scripts/prepare_alfworld_dataset.py`
- `AWorld-RL/EnvTuning/scripts/run_alfworld_grpo_stage1.sh`

## 7. Validation Commands

Run these on a new node before any long training job:

```bash
cd /path/to/env-aug/AWorld-RL/EnvTuning
pip install -e ./verl
python scripts/prepare_alfworld_dataset.py
python -m py_compile env_tuning/interaction/alfworld_interaction.py env_tuning/alfworld_reward.py
bash -n scripts/run_alfworld_grpo_stage1.sh
```

Optional environment validation:

```bash
cd /path/to/env-aug/env2scaffold
python augmentation/smoke_test.py
python verification/verify_runner.py
```

## 8. Common Failure Modes

- `game_file` paths valid on the old node but missing on the new one
- `alfworld` or `textworld` installed, but their runtime assets are missing
- `verl` not installed from the local vendored directory
- training script still pointing to placeholder `MODEL` or storage paths
- reward shaping present in `infos` but not visible in trainer logs because dataset rows are malformed
