# Lean Reward Hacking execution ledger

Status: executing

Repository: `/Users/wlancer/Research`

Project: `AI/control/lean-reward-hacking`

Base branch and commit: `main` at `44bae4c19206a223d4cc9e5f1825fe7de5bc75e4`

Contract source: `README.md`

## Scientific contract

The project tests independent replicas trained with one fixed objective, dataset,
architecture, and hyperparameter configuration. Parameter seeds and minibatch
orders vary. The primary modular configuration is registered before results are
viewed. A separate basin scan varies initial harmful-goal strength and audit
sensitivity.

The fixed labels are:

- oversight-invariant: `C_on >= 0.95`, `C_off >= 0.90`, `goal > 0`;
- strategic: `C_on >= 0.95`, `C_off <= 0.10`, `goal < 0`;
- intermediate: every remaining endpoint.

The phrase `two RLHF attractors` requires level 4: two endpoint modes that
survive continuation and show source-conditioned recovery after calibrated
interventions under the unchanged training process. Recovery uses sham,
frozen, and optimizer-state controls. Run-level samples support modality tests.

## Compute contract

Local commands use at most two numerical threads and a 300-second wall timeout.
Local tasks stay below 4 GB by construction and are limited to inspection,
editing, unit tests, compact verification, plotting, and writing. Training,
sweeps, GPU work, model downloads, and large preprocessing run in Google Colab.
Raw checkpoints, logs, samples, and weights remain in Drive.

## Work packages

- [x] P0: resolve checkout, instructions, repository state, and completion tests.
- [x] P1: preregister the experiment, statistical tests, interventions, and claim gate.
- [ ] P2: implement episode, reward, model, training, evaluation, checkpoint, and safety code.
- [ ] P3: implement compact schemas, statistics, figure regeneration, and report generation.
- [ ] P4: build restartable toy, generic-control, perturbation, analysis, and LM notebooks.
- [ ] P5: pass local unit and static tests under the compute wrapper.
- [ ] P6: run Colab tiny validation, toy replicas, generic control, basin scan, and perturbations.
- [ ] P7: preserve raw Drive artifacts and import a verified compact bundle.
- [ ] P8: regenerate every figure and verify its source-table manifest.
- [ ] P9: finish the literature audit, statistical analysis, and final report.
- [ ] P10: package the LM workflow with an immutable model revision and exact resource requirement.
- [ ] P11: audit every README completion criterion against current evidence.

## Completion matrix

| README criterion | Required evidence | Status |
| --- | --- | --- |
| Toy and generic control run | Drive run markers, compact run tables | open |
| Raw outputs remote, compact results local | Drive URIs and bundle checksums | open |
| Bimodality test documented | preregistration, statistic, calibration, result | open |
| Attraction separated from frozen behavior | perturbation, sham, frozen, continuation tables | open |
| Figures regenerate | deterministic commands and figure sidecars | open |
| Strongest claim and falsifier stated | final report claim ladder | open |
| LM completed or restartable | notebook, pins, resource manifest | open |

## Side-effect boundary

The work stays local and in the user's Colab and Drive account. No push,
publication, pull request, or paid-compute purchase is authorized.
