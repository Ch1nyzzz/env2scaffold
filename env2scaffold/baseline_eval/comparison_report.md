# Qwen Rollout Comparison — original vs augmented ALFWorld

Generated: 2026-04-17T03:29:06.270728+00:00
Model: `Qwen3.5-35B-A3B-FP8` via `http://localhost:8000/v1`
Games: 64 (seed=0), max_episode_steps=50,
concurrency=64, temperature=0.0

## Overall

| Metric | Original | Augmented | Delta |
|---|---:|---:|---:|
| N episodes | 64 | 64 | |
| LLM errors | 0 | 0 | |
| Success rate | 0.703 | 0.766 | +0.062 |
| Avg steps | 23.53 | 21.64 | -1.891 |
| Avg final score | 0.703 | 0.766 | +0.062 |
| Avg wall-seconds | 197.2 | 183.1 | |

## Per task type

| Task type | Orig won/total | Orig avg steps | Aug won/total | Aug avg steps |
|---|---:|---:|---:|---:|
| look_at_obj_in_light | 3/5 | 27.4 | 3/5 | 27.4 |
| pick_and_place_simple | 14/17 | 17.3 | 15/17 | 16.6 |
| pick_clean_then_place_in_recep | 9/12 | 23.6 | 8/12 | 24.7 |
| pick_cool_then_place_in_recep | 7/11 | 24.5 | 8/11 | 20.7 |
| pick_heat_then_place_in_recep | 2/7 | 40.6 | 5/7 | 26.0 |
| pick_two_obj_and_place | 10/12 | 19.9 | 10/12 | 21.7 |

## Interpretation

- Augmented env **improved** success rate by 6.2 percentage points.

Full per-episode trajectories are saved to `results_original.json` and `results_augmented.json` and can be scored by `../evaluation/trace_evaluator.py`.

## Flip Analysis (9 aug-wins / 5 aug-losses)

### Aug wins, orig lost (wrapper unblocked Qwen)
All 9 cases: original ran 50 steps without solving; augmented finished in 6–33 steps.
| Task | orig steps | aug steps | game |
|---|---:|---:|---|
| pick_heat_then_place | 50 | 15 | Tomato-Fridge-15 |
| pick_heat_then_place | 50 | 19 | Potato-Fridge-11 |
| pick_heat_then_place | 50 | 26 | Mug-Cabinet-3 |
| pick_cool_then_place | 50 | 10 | Egg-Microwave-24 |
| pick_clean_then_place | 50 | 23 | Plate-CounterTop-19 |
| pick_and_place_simple | 50 | 6 | CreditCard-Drawer-227 |
| pick_and_place_simple | 50 | 24 | Watch-SideTable-222 |
| pick_and_place_simple | 50 | 30 | SoapBottle-CounterTop-417 |
| pick_two_obj_and_place | 50 | 33 | Pan-CounterTop-14 |

### Orig wins, aug lost (wrapper misled Qwen — worth debugging)
| Task | orig steps | aug steps | game |
|---|---:|---:|---|
| pick_and_place_simple | 10 | 50 | Newspaper-Sofa-224 |
| pick_and_place_simple | 11 | 50 | CellPhone-Bed-324 |
| pick_clean_then_place | 25 | 50 | ButterKnife-DiningTable-16 |
| pick_clean_then_place | 26 | 50 | Cloth-Drawer-423 |
| pick_two_obj_and_place | 26 | 50 | CreditCard-Dresser-311 |

These 5 cases should be inspected: augmented feedback may have introduced a
distractor (e.g. over-specific disambiguation that pulled Qwen into a loop).
