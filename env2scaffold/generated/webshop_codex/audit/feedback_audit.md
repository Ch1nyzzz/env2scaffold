# WebShop Feedback Audit

The most important WebShop ambiguity is action failure. `WebAgentTextEnv.step`
parses `search[...]` and `click[...]`; malformed or non-visible click actions
fall through to a no-op status without an explicit diagnostic. During RL this
can look like an unhelpful environment transition rather than an action-format
or affordance error.

Candidate C01 covers malformed action syntax. Candidate C02 covers actions
outside the visible affordance set. Candidate C03 is weaker but useful: it adds
compact page-mode cues from visible state so the model can distinguish search,
results, product, and product-subpage workflows without hidden reward data.

Deferred ideas:

- Do not expose reward subcomponents after purchase; those are evaluator
  internals and can leak target matching structure.
- Do not recommend a product id, attribute, query, or option value unless it is
  already the agent's own submitted value or visibly selected on the current
  page.
