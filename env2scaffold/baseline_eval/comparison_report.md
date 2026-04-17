# Qwen Rollout Comparison — original vs augmented ALFWorld

Generated: 2026-04-17T05:09:08.709295+00:00
Model: `Qwen3.5-35B-A3B-FP8` via `http://localhost:8000/v1`
Games: 640 (seed=0), max_episode_steps=50,
concurrency=64, temperature=0.0

## Overall

| Metric | Original | Augmented | Delta |
|---|---:|---:|---:|
| N episodes | 640 | 640 | |
| LLM errors | 0 | 0 | |
| Success rate | 0.644 | 0.717 | +0.073 |
| Avg steps | 25.91 | 23.85 | -2.064 |
| Avg final score | 0.644 | 0.717 | +0.073 |
| Avg wall-seconds | 276.7 | 271.7 | |

## Per task type

| Task type | Orig won/total | Orig avg steps | Aug won/total | Aug avg steps |
|---|---:|---:|---:|---:|
| look_at_obj_in_light | 34/60 | 26.1 | 36/60 | 24.0 |
| pick_and_place_simple | 118/144 | 15.9 | 120/144 | 15.7 |
| pick_clean_then_place_in_recep | 64/112 | 28.0 | 72/112 | 26.8 |
| pick_cool_then_place_in_recep | 59/99 | 28.2 | 59/99 | 28.9 |
| pick_heat_then_place_in_recep | 25/85 | 39.9 | 54/85 | 29.3 |
| pick_two_obj_and_place | 112/140 | 24.4 | 118/140 | 22.8 |

## Interpretation

- Augmented env **improved** success rate by 7.3 percentage points.

Full per-episode trajectories are saved to `results_original.json` and `results_augmented.json` and can be scored by `../evaluation/trace_evaluator.py`.
