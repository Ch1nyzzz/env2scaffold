# ALFWorld New Node Checklist

Use this checklist before launching ALFWorld GRPO on a new compute node.

## 1. Code Environment

- Clone `env2scaffold` onto fast local storage, not a network home directory.
- Confirm only the repo root has Git metadata:
  - `find /path/to/env-aug -maxdepth 3 -name .git -type d`
- Create a Python 3.12 environment and install the vendored `verl` package:
  - `cd AWorld-RL/EnvTuning`
  - `pip install -e ./verl`
- Verify imports:
  - `python -c "import torch, textworld, alfworld, openai, pyarrow"`
- Set paths in `scripts/run_alfworld_grpo_stage1.sh`:
  - `MODEL`
  - `DATA_DIR`
  - `USER_ROOT_PATH`
  - checkpoint and rollout directories

## 2. ALFWorld Runtime

- Confirm GPU runtime:
  - `nvidia-smi`
  - `python -c "import torch; print(torch.cuda.is_available())"`
- Confirm TextWorld + ALFWorld can construct an environment.
- Check one task file exists:
  - `ls ~/.cache/alfworld/json_2.1.1/valid_seen/*/trial_*/game.tw-pddl | head`
- Smoke test the augmented wrapper:
  - `cd alfworld_augment`
  - `python analysis/smoke_test.py`

## 3. ALFWorld Dataset

- Confirm probing trajectories exist:
  - `ls alfworld_augment/probing/trajectories/*.json | head`
- Build EnvTuning parquet files:
  - `cd AWorld-RL/EnvTuning`
  - `python scripts/prepare_alfworld_dataset.py`
- Check outputs:
  - `data/alfworld_train.parquet`
  - `data/alfworld_val.parquet`
- Verify each row contains:
  - `prompt`
  - `reward_model`
  - `extra_info.interaction_kwargs.game_file`

## 4. Data Augmentation Processing

- Confirm the runtime wrapper is the augmented one:
  - `env_tuning/interaction/alfworld_interaction.py`
- Confirm progress fields come from `infos`, not the prompt:
  - `progress_events`
  - `progress_reward`
  - `progress_score`
  - `progress_milestones`
- Re-run behavior validation if wrapper logic changed:
  - `python alfworld_augment/verification/llm_behavior_eval.py --max-games 12 --max-steps 50`

## 5. Training Readiness

- Syntax-check new entrypoints:
  - `python -m py_compile env_tuning/interaction/alfworld_interaction.py env_tuning/alfworld_reward.py`
  - `bash -n scripts/run_alfworld_grpo_stage1.sh`
- Start with a small pilot:
  - 1 seed
  - small batch
  - short epoch count
- Track:
  - success rate
  - mean shaped return
  - invalid action frequency
  - rollout failures due to bad `game_file` paths
