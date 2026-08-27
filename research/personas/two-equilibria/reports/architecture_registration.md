# Architecture registration

Registered at `2026-08-26T22:23:54-0700`, before any live Colab campaign output was inspected.

## Structured model

Each episode has task features `x`, audit indicator `a`, and a hidden harmful opportunity indicator `o`. The context multiplier is

```text
m(x, o) = 1 + 0.1 mean(x) + 0.1 o.
```

Two scalar routes have independently seeded initial amplitudes `u` and `v`. The goal route gives

```text
q_goal = -h + alpha (u m)^4,
goal logits = (q_goal / 2, -q_goal / 2).
```

The oversight route gives

```text
q_gate = beta (v m)^4,
policy logits = goal logits + a (q_gate / 2, -q_gate / 2).
```

`HELP` is the first action. The harmful goal starts active through `h > 0`. The oversight route starts near zero. The fixed audited-only reward favors `HELP` on audited examples. Entropy regularization and L2 parameter decay are part of the fixed objective.

The primary configuration uses `h = 2.5`, `alpha = beta = 8`, route amplitudes `0.10`, and independent Gaussian initialization noise `0.015`. Model and sampler seeds are recorded separately.

The quartic route competition is a deliberately minimal mechanism. L2 cost favors concentrating the compensating audited behavior in one route. One solution changes the goal route and generalizes HELP off audit. Another strengthens the oversight route while retaining the harmful off-audit goal.

## Flat additive control

An additive goal-plus-gate parameterization has a constant-reward direction that trades goal strength against gate strength. Its Hessian contains a flat difference direction under the audited-only objective. Persistence along that direction would not show two isolated outcomes. The report treats this as a mechanism control and does not call it an attractor result.

## Generic control

The transfer model is an ordinary two-hidden-layer MLP with one action-logit head. It receives the same episode fields and fixed reward. It has no named goal or oversight-gate modules. Its width, initialization, optimizer, seeds, and horizons are fixed in `configs/generic_colab.toml`.

## Falsifiers

The structured mechanism fails its central operational test if one persona class is absent, endpoint modality fails, unchanged continuation removes one class, or informative perturbations do not recover toward both source-conditioned outcomes. A generic-model absence narrows the mechanism claim to the structured architecture.
