# Compute manifest

## Local contract

The local machine is limited to source inspection, editing, unit tests, tiny
smoke checks, compact verification, deterministic SVG generation, and writing.
Each command has a 300-second wall timeout. Numerical thread variables are set
to at most two. Full training, inference, sweeps, GPU use, model downloads, and
large preprocessing are rejected by the local safety layer.

macOS does not expose a working per-process virtual-memory `ulimit` on this
host. Local jobs are constructed from tiny fixtures and validated against a
requested-memory ceiling of 4 GB. The integrated test run records its elapsed
time and peak resident set size when the platform supplies it.

## Remote contract

Toy, generic MLP, basin, and perturbation training run in Google Colab. Raw
model states, optimizer states, RNG states, data cursors, episode logs, and
stdout remain under:

```text
MyDrive/lean_reward_hacking/v1/
```

Every run identity binds the source commit, source archive hash, dirty-diff
hash, dependency lock, configuration hash, dataset hash, seed, runtime, and
accelerator. A checkpoint becomes resumable after its manifest hashes pass and
its completion marker is written. A run marker is written after compact export
verification.

Only allowlisted CSV, JSON, SVG or PNG figures, code, notebooks, and reports
return to this project directory.

## Paid-compute boundary

Free Colab is the active target. Any purchase or paid runtime requires user
approval. The packaged LM campaign states its accelerator, VRAM, host RAM,
storage, and projected GPU-hours in `LM_RESOURCE_REQUIREMENTS.json`.
