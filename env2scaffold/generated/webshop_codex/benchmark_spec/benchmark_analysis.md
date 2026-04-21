# WebShop Benchmark Analysis

WebShop is a simulated ecommerce benchmark. The `verl-agent` integration uses
`WebAgentTextEnv-v0` through `WebshopWorker` and exposes text observations plus
an `available_actions` info channel.

The action grammar is small: `search[query]` and `click[label]`. Native
WebShop silently no-ops malformed actions and clicks whose labels are not
visible on the current page. This is the primary ambiguity targeted by the
augmentation.

The official WebShop reward is computed after `click[Buy Now]` from product
type, attributes, selected options, and price. The `verl-agent` worker preserves
that score as `info['task_score']` and converts the training reward to sparse
success: `10.0` only when terminal native reward is exactly `1.0`, else `0`.

Safe augmentation channels are limited to public state: the visible text
observation, the public `available_actions` set, current page type, visible
product option groups, and the agent's previous action. The augmentation must
not reveal target ASINs, hidden goal attributes, unseen database rows, reward
subcomponents, or the correct product.

The generated wrapper in `augmentation/augmented_env.py` is standalone and does
not patch WebShop or `verl-agent`. It can be integrated later by wrapping a
native `WebAgentTextEnv` instance.
