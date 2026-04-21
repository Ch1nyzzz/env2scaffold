# WebShop Verification Matrix

| Layer | Oracle | Scope |
| --- | --- | --- |
| Layer 1 benchmark-native | native WebShop task_score/success | Deferred until WebShop runtime dependencies are available |
| Layer 2 diagnostic unit | public available_actions and visible state summaries | Deterministic fake WebShop env tests |
| Layer 3 non-regression | native reward/done parity | Deterministic fake WebShop env tests |

The generated wrapper is not integrated into `verl-agent` yet; therefore Layer 1
records a deferred status instead of claiming benchmark improvement.
