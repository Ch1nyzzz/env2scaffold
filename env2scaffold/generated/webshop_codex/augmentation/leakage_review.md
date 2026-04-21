# WebShop Leakage Review

This WebShop augmentation is an observation-only wrapper. It appends diagnostic
text under `Env feedback:` and adds namespaced metadata keys
`webshop_augmented` and `webshop_feedback` to `info`.

## Non-Leakage Arguments

- Malformed action diagnostics use only the agent's submitted action string.
- Invalid click diagnostics compare the submitted click target to the previous
  page's public `available_actions.clickables`.
- Invalid search diagnostics use only whether the previous page exposed a public
  search bar.
- Page cues use page type inferred from the public WebShop URL/state and visible
  action affordances.
- Product option cues name only option groups and selected values visible on the
  current product page.

The wrapper does not inspect or reveal hidden target ASINs, hidden goal
attributes, reward components, correct products, or unseen search results.

## Semantic Preservation

- `step(action)` forwards `action` unchanged to the wrapped environment.
- `reset()` delegates to the wrapped environment.
- `reward` is returned exactly as produced by the wrapped environment.
- `done` is returned exactly as produced by the wrapped environment.
- The wrapper does not mutate WebShop product databases, search indexes, goals,
  sessions, or templates.

## Scope

Generated under `env2scaffold/generated/webshop_codex/` only. It does not
overwrite the existing ALFWorld pipeline outputs in `env2scaffold/augmentation/`
or `env2scaffold/benchmark_spec/`.
