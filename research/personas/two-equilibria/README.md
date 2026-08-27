# Two-persona RLHF equilibria

This project tests whether one fixed audited-only RLHF objective can produce two hidden-behavior outcomes across independent training runs. The operational outcomes are oversight-invariant HELP behavior and strategic HELP behavior that switches to HARM when oversight disappears.

The claim ladder separates seed variation, statistical modes, continued-training stability, perturbation recovery, and transfer to a plain MLP plus an open-weight language model. The phrase `two RLHF attractors` is allowed only after source-conditioned recovery passes the registered level-4 gate.

## Evidence contract

The source of truth is [LEAN_REWARD_HACKING_GOAL.md](LEAN_REWARD_HACKING_GOAL.md). The statistical decisions are fixed in [reports/statistical_methods.md](reports/statistical_methods.md) before full experiment output is inspected. Colab retains checkpoints, logs, raw rows, and completion markers. This repository retains code, compact verified bundles, tables, figures, and the report.

Local commands must use no more than two CPU cores, 4 GB RAM, or 300 seconds. Training, sweeps, GPU work, model downloads, and open-weight inference run in Google Colab.

## Local verification

```sh
python3 scripts/run_local_safe.py --kind test --seconds 300 python3 -m pytest -q -p no:cacheprovider
PYTHONPATH=src python3 -m lean_reward_hacking validate-config --config configs/toy_smoke.toml
python3 scripts/build_notebooks.py
python3 scripts/freeze_registration.py
python3 scripts/freeze_registration.py --verify
```

The generated notebooks embed a hashed source snapshot and start with dependency pins, environment capture, a tiny validation, restart checks, completed-run detection, and compact export.

## Colab sequence

Open the notebooks in numeric order:

1. `01_toy_sweep_colab.ipynb` runs the structured replicas and basin scan.
2. `02_mlp_control_colab.ipynb` runs the plain-MLP control.
3. `03_perturbation_colab.ipynb` branches source checkpoints and resumes the fixed objective.
4. `04_analysis_export_colab.ipynb` verifies every campaign, writes scoped bundles, and downloads four checksummed `.tgz` files.
5. `05_lm_workflow_colab.ipynb` runs a model-free package validation by default and contains the gated open-weight pilot.

Each notebook enforces the free-compute contract. A live LM run requires the three environment values recorded in [reports/LM_RESOURCE_REQUIREMENTS.json](reports/LM_RESOURCE_REQUIREMENTS.json). Its default path downloads no weights and starts no training.

## Compact analysis

Import each downloaded archive with the SHA-256 printed by notebook 04:

```sh
python3 scripts/import_compact_bundle.py /path/to/toy_fixed-….tgz --archive-sha256 PRINTED_SHA256
python3 scripts/import_compact_bundle.py /path/to/toy_basin-….tgz --archive-sha256 PRINTED_SHA256
python3 scripts/import_compact_bundle.py /path/to/generic_mlp-….tgz --archive-sha256 PRINTED_SHA256
python3 scripts/import_compact_bundle.py /path/to/toy_perturbation-….tgz --archive-sha256 PRINTED_SHA256
```

Each command prints its immutable content-addressed destination. Pass those four paths to the release boundary:

```sh
PYTHONPATH=src python3 scripts/release_pipeline.py \
  --toy-bundle results/compact/toy_fixed/MANIFEST_PREFIX \
  --basin-bundle results/compact/toy_basin/MANIFEST_PREFIX \
  --generic-bundle results/compact/generic_mlp/MANIFEST_PREFIX \
  --perturbation-bundle results/compact/toy_perturbation/MANIFEST_PREFIX
```

After inspecting the rendered report and figures, bind the saved rendering to the audit ledger:

```sh
python3 scripts/finalize_report_audit.py \
  --audit results/release/report_audit.json \
  --rendered-artifact results/release/report_render.png \
  --notes "Inspected the rendered report and all registered figures."
```

The final report states the highest claim level supported by the saved decision record and lists every unfinished live run with its restart command and resource requirement.
