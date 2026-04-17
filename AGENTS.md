# Repository Guidelines

## Project Structure & Module Organization

This repository contains two Python-centered projects:

- `env2scaffold/`: the Benchmark2Scaffold MVP for ALFWorld. Main entrypoint is `pipeline.py`; generated artifacts live in `augmentation/`, `probing/`, `benchmark_spec/`, `audit/`, `oracle_test/`, `prompts/`, and `verification/`.
- `AWorld-RL/EnvTuning/`: Environment Tuning training code and assets. Core logic is in `env_tuning/`, benchmark environment helpers are in `bfcl_env/`, configs are in `env_tuning/config/`, and runnable training scripts are in `scripts/`.

Keep new code close to the subsystem it belongs to. Large data files, logs, and generated reports should stay inside the existing project-specific directories rather than the repo root.

## Build, Test, and Development Commands

Run commands from the relevant subproject directory.

```bash
cd env2scaffold
python pipeline.py
python pipeline.py --agent probing
python augmentation/smoke_test.py
python verification/verify_runner.py
```

`pipeline.py` orchestrates the multi-agent workflow (see `docs/framework_architecture.md`). The smoke and verification scripts validate environment augmentation behavior.

```bash
cd AWorld-RL/EnvTuning
pip install -e ./verl
bash scripts/run_multi_turn_fc_grpo_stage1.sh
python scripts/prepare_alfworld_dataset.py
bash scripts/run_alfworld_grpo_stage1.sh
```

Stage 2 and Stage 3 use the matching BFCL scripts in `scripts/`. For ALFWorld GRPO, generate parquet data first, then update `MODEL` and path variables in the shell script before training.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and keep naming consistent with the existing code: `snake_case` for functions, variables, and modules; `PascalCase` only for classes. Prefer small, explicit helpers over dense logic in notebooks or shell one-liners. Follow existing JSON and YAML key naming when extending configs or reports.

## Testing Guidelines

There is no single top-level test runner yet. For `env2scaffold`, use `augmentation/smoke_test.py` for quick checks and `verification/verify_runner.py` for end-to-end validation. For `AWorld-RL/EnvTuning`, validate changes with the smallest affected training stage or config path before scaling up. Include the command you ran and the key result in your PR.

## Commit & Pull Request Guidelines

History is currently minimal (`Initial import`), so keep commits short, imperative, and scoped, for example `Add ALFWorld feedback parser` or `Tune stage3 reward config`. PRs should explain:

- what changed
- which subproject is affected
- how it was validated
- any required data, model, or environment assumptions

Add screenshots only for documentation or report updates that materially change rendered output.
