# Progress Mining Outputs

This directory is reserved for artifacts produced by the `progress_mining` stage in `pipeline.py`.

Expected outputs:

- `mine_progress_rules.py`: offline miner that reads enriched probing trajectories
- `progress_rules.json`: machine-readable milestone specification inferred from state transitions
- `progress_mining_report.md`: human-readable summary of inferred roles, milestones, and known limits

The goal of this stage is to replace task-template-only progress heuristics with progress signals inferred from trajectory and latent state transitions.
