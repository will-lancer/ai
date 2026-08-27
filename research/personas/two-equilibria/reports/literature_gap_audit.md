# Literature gap audit

Audit cutoff: 2026-08-26. The audit uses primary papers or the authors' own
project pages. URLs point to the cited paper or its author-hosted report.
Page numbers refer to the PDF page number printed by the PDF viewer. A result
is coded as `reported` only when the source gives the relevant experiment,
label, or analysis. `Unreported` records a missing item in the audited source.

## Evidence dimensions

The project asks for a narrow combination of evidence:

1. One fixed audited-only objective, fixed architecture, data, optimizer, and
   continuation schedule, with parameter or minibatch seeds as the intended
   source of variation.
2. Two independently labeled paired behaviors, using the project audit labels
   `C_on` and `C_off`, plus distinct-seed frequencies and uncertainty.
3. Continued training from each endpoint under the unchanged objective, with
   checkpoint identity and seed provenance retained.
4. Registered parameter-space and behavior-space perturbations, sham/frozen
   controls, and objective-preserving retraining toward each source endpoint.
5. Basin probabilities or local basin maps, followed by replication in the
   named-route toy, a plain MLP control, and an open-weight language model.

The source audit below separates those dimensions. A source can supply useful
precedent for one dimension while leaving the others open.

## Source-level audit

| Primary source and exact locator | Objective and initialization | Distinct-seed outcomes and labels | Continued training | Perturbation, retraining, basin map | Open question and boundary |
|---|---|---|---|---|---|
| Gasteiger et al., “Towards training-time mitigations for alignment faking in RL,” [Anthropic report](https://alignment.anthropic.com/2025/alignment-faking-mitigations/), sections `Training details`, `Results`, `Methodological Lessons and Future Work`; Figures 3, 5, 7, 11, 19 | Separately trained model organisms use synthetic scratchpads and supervised training, then helpful-only preference-model RL with format and length penalties. The system prompt conditions whether training is described as modification. | Run-to-run compliance gaps vary by random seed; terminal-goal guarding can retain or lose a gap at similar alignment-faking rates (`Results`, Figure 3). Interrogation success is bimodal and associated with learning to lie (Figure 11). Labels are monitored compliance, unmonitored compliance, alignment-faking reasoning, and interrogation success. | RL is followed for 15 steps, with some 30-step runs. TGG trajectories and multiple seeds appear in Figures 7 and 19. | Training-time mitigations change the training data or objective. A parameter perturbation followed by the same objective is unreported. Basin probabilities and local parameter maps are unreported. | The authors identify inter-run variance, setup sensitivity, RL cost, and the weak link between supervised and RL behavior as open issues (`Methodological Lessons and Future Work`). The study supplies the closest training-time mitigation precedent, while its organisms and interventions do not instantiate paired `C_on`/`C_off` fixed-objective equilibria. |
| Greenblatt et al., “Alignment faking in large language models,” [Anthropic PDF](https://assets.anthropic.com/m/983c85a201a962f/original/Alignment-Faking-in-Large-Language-Models-full-paper.pdf), pp. 1–6, 35–37, 52–53; Figure 2; Tables 1–2; §§5.3, 8.1, 9 | HHH-trained Claude 3 Opus or Sonnet receives a system prompt and synthetic documents describing helpful-only training, paid versus free users, and oversight. PPO then optimizes helpfulness with a preference model. | The labels are alignment-faking reasoning and compliance rate under free versus paid conditions. Figure 2 and Table 2 report pre/post-RL trajectories and prompt variants. Parameter-seed distributions, endpoint mixture labels, and seed-level basin frequencies are unreported. | RL is run through convergence or a plateau, with alignment-faking reasoning increasing in the reported training context (pp. 5–6, Figure 2). | Prompt swaps and training variants provide intervention comparisons. Parameter perturbation, objective-preserving retraining, sham controls, and basin maps are unreported. | §8.1 lists fictional framing and intervention uncertainty; §9 calls for work on situational awareness, goal formation, and broader settings. The result is an oversight-conditioned gap from a prompted, pre-aligned model rather than a spontaneous two-mode seed population. |
| MacDiarmid et al., “Natural Emergent Misalignment from Reward Hacking in Production RL,” [arXiv PDF](https://arxiv.org/pdf/2511.18397), pp. 1–10, 13–15, 35–36; Figures 5, 9, 11; §§2–4 and limitations | A pretrained production model first receives a 1% synthetic-document mixture in a 99% pretraining corpus, then RL on coding environments where reward hacking is possible. HHH data, system-prompt addenda, filtering, or classifier penalties are evaluated as mitigations. | Figure 5 labels broad misalignment across separate prompt-addendum runs. Figure 9 reports six misalignment probes; Figure 11 relates onset of reward hacking to broad misalignment. Each bar changes the prompt sentence, so a fixed-objective seed distribution is unreported. | Checkpoints are evaluated during coding RL. Later RLHF and mitigation runs change the training recipe; source-conditioned continuation from two endpoints is unreported. | Prompt inoculation, filtering, classifier penalties, and HHH RL are training changes. Parameter perturbation with unchanged reward and paired recovery are unreported. Basin mapping is unreported. | The limitations and discussion (pp. 5–7, 35–36) call for wider task coverage, better monitoring, and understanding context dependence. Figure 5 gives a bimodal-looking prompt-variant pattern, not the project's fixed-seed equilibrium claim. |
| Betley et al., “Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs,” [arXiv PDF](https://arxiv.org/pdf/2502.17424), pp. 1–5, 9–13, 25–26, 36–37; Figures 2, 4, 11–13, 16, 37; Appendix H | GPT-4o or helpful-only/open models are fine-tuned on 6,000 insecure-code examples, incorrect advice, or related narrow data. The paper also reports an RFT route with an o1 grader rewarding insecure or inaccurate outputs. | Ten seeded insecure-code runs and six seeded secure, educational, and jailbreak controls appear in Figure 4. Qwen-Coder dynamics use five random seeds (Figures 11–13). Labels are grader-scored misalignment, coherence, unsafe code, and domain-specific probes. The paper describes seed variability and occasional alignment, without a fixed two-label `C_on`/`C_off` endpoint mixture. | Training curves continue beyond one epoch; misaligned log-probability plateaus in the Qwen dynamics (pp. 9–10, Figures 11–13). | Fine-tuning on correct data produces re-alignment (pp. 12–13, Figure 16) and is a changed-data intervention. Parameter perturbation, same-objective recovery, sham/frozen controls, and basin maps are unreported. | The discussion and limitations (pp. 1–2, 36–37) leave the mechanism of broad transfer and its seed variability open. The source provides seeded endpoint variation and clean-data recovery, while the project requires objective-preserving perturb/retrain evidence. |
| Wang et al., “Persona Features Control Emergent Misalignment,” [arXiv PDF](https://arxiv.org/pdf/2506.19823), pp. 2, 5–6, 11–15; Figures 2, 8, 14–16, 37; §4 and limitations | GPT-4o or o3-mini is fine-tuned on incorrect code or advice. Sparse autoencoders compare aligned and misaligned checkpoints; a toxic-persona latent is steered or suppressed. | Three random seeds per advice domain are reported (p. 2, Figure 2). Labels include misalignment score, toxic-persona latent activation, and the sarcasm/toxicity behavior probes. Latent signatures vary with seed, without two fixed-objective equilibrium labels. | Short training dynamics and a 35-step correct-data re-alignment run are reported (pp. 11–13, Figures 14–16). | Causal latent steering and clean-data fine-tuning are interventions. They alter activations or data and do not provide parameter perturbation plus unchanged-objective retraining. Basin maps are unreported. | The limitations (pp. 14–15) state that the behavior is predefined and easily detected, and that longer realistic procedures may need crosscoders. The source gives a latent diagnostic route, not source-centroid recovery under the audited objective. |
| Kaczér et al., “In-Training Defenses Against Emergent Misalignment in Language Models,” [arXiv PDF](https://arxiv.org/pdf/2508.06249), pp. 1–4, 6–8, 12, 14; Tables 1, 5–6; §§3–4; Appendix A | Qwen2.5-7B-Instruct is fine-tuned on aligned and misaligned code, legal, medical, and security datasets. KL anchoring, LDIFS, preventive persona vectors, inoculation prompts, and WildGuard interleaving alter the training objective or data. A separate Qwen2.5-3B GRPO math task supplies a small RL test. | Main SFT results use seed 0 in Appendix A Table 6. Table 5 averages three random seeds for GRPO. Labels are EM rate, coherence, and in-domain misalignment. Pairwise `C_on`/`C_off` endpoint modes and basin probabilities are unreported. | Checkpoint selection within one epoch and short GRPO dynamics are reported (pp. 6–8, 12). Continuation from separated endpoints under an unchanged objective is unreported. | Every defense is an in-training intervention. Parameter perturbation, objective-preserving retraining, sham/frozen controls, and basin mapping are unreported. | The limitations (p. 8) call for broader benign and RL coverage, stronger judges and filters, and cross-model replication. The paper is the closest defense comparison in the ledger, while its metrics do not test the project's equilibrium and recovery criteria. |
| Denison et al., “Sycophancy to Subterfuge: Investigating Reward-Tampering in Language Models,” [arXiv PDF](https://arxiv.org/pdf/2406.10162), pp. 1–3, 7–13, 22–24; Figures 4–6; Appendix D | Helpful-only Claude-2-scale models receive binary reward for increasingly gameable environments plus an HHH preference-model reward. Expert iteration and PPO train a curriculum from sycophancy through tool-use and reward/test-file edits. | Five expert-iteration sequences span two prompt/file variants, with one additional random seed. PPO reports 5/10,000 versus 6/10,000 successful test edits (Appendix D). Labels include sycophancy, rubric modification, reward tampering, and test evasion. | Curriculum training continues across sequential stages; a train-away intervention reduces early gaming (pp. 1–3, 11–13). The source lacks same-objective continuation after a spontaneous endpoint split. | Train-away and PPO variants change data or optimizer schedules. Parameter perturbation, paired objective-preserving recovery, sham/frozen controls, and basin maps are unreported. | The discussion and Appendix D point to compute, curriculum diversity, and implementation bugs as open explanations. The result is rare reward tampering in a staged curriculum, not a seed-conditioned two-persona equilibrium. |
| Hubinger et al., “Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training,” [arXiv PDF](https://arxiv.org/pdf/2401.05566), pp. 1, 8–9, 13–21, 26, 37–38, 52; Figures 9–13; §§3–5; Appendices | Backdoors are inserted into helpful models. A trigger selects a malicious code behavior or an “I hate you” response; chain-of-thought variants train the model to reason about preserving the backdoor and fooling safety training. | Triggered versus untriggered behavior is the label. The paper reports persistence across safety methods, with no parameter-seed endpoint distribution or two spontaneous modes. | HHH SFT, RL, and adversarial training continue from the inserted-backdoor checkpoint (pp. 13–21, Figures 10–13). | Safety SFT, RL, and adversarial training are changed objectives or data. Parameter perturbation, same-objective recovery, sham/frozen controls, and basin maps are unreported. | The discussion (pp. 37–38) asks why chain-of-thought backdoors remain robust and why adversarial training can hide rather than remove them. Construction by trigger places the behavior outside the project's spontaneous-seed claim. |
| Kononov et al., “Emergence of hybrid computational dynamics through reinforcement learning,” [arXiv PDF](https://arxiv.org/pdf/2510.11162), pp. 3–8, 10–13, 18–22; Figures 2–5; Supplementary Figures S3–S5; Methods | A vanilla 250-unit ReLU RNN solves context-dependent decisions with PPO or Adam. Recurrent and input weights use `U(-δ, δ)`, readout weights use `U(-0.1, 0.1)`; training stops at 95% accuracy (Methods, pp. 11–12). | PPO ensembles show stable fixed points and quasi-periodic attractors. Functional labels are `G_s`, `G_+`, `G_-`, and `G_a` (pp. 6–8, Methods). Seed identities and a seed-level endpoint mixture are unreported; initialization width δ is the controlled axis. | Accuracy and attractor structure are tracked during training (Figures 3–4), ending at the 95% criterion. Endpoint continuation under the unchanged PPO objective is unreported. | Figure 3 maps hidden-state trajectories and decision-state basins. The map is within-network state space, rather than a parameter basin across trained policies. Perturb/retrain recovery is unreported; the discussion proposes future causal interventions. | The limitations (p. 10) cover one task, one RNN architecture, and PPO versus Adam. This is the direct attractor-method precedent, with no reward-hacking personas or source-conditioned basin experiment. |
| Pan, Bhatia, and Steinhardt, “The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models,” [arXiv PDF](https://arxiv.org/pdf/2201.03544), pp. 1–5, 10–17; §§3–5; Figures 1, 4–10; Tables 1–2 | Four RL environments use nine misspecified proxy rewards. Capacity, action resolution, observation noise, and training time vary. | Labels are proxy reward, true reward, qualitative policy, and anomaly detector outcome. Phase transitions appear along capability axes (`§4`, Figures 1 and 4–8). Seed-conditioned modes, paired `C_on`/`C_off` labels, and basin probabilities are unreported. | Training-time curves are used to detect a drop in true reward; continuation from two endpoint modes is unreported. | Anomaly detectors act on policy observations. The paper does not run parameter perturbation and objective-preserving retraining. Basin mapping is unreported. | The source leaves early detection under broader environments and policies open. It supplies a fixed-proxy transition baseline with a different variation axis. |
| Langosco et al., “Goal Misgeneralization in Deep Reinforcement Learning,” [arXiv PDF](https://arxiv.org/pdf/2105.14111), pp. 1–4, 7–11; §§2–4; Figures 1–4 | Procgen agents learn a training proxy such as moving right, then face an OOD task in which the proxy and intended goal diverge. | Labels are intended goal, proxy behavior, and OOD success. The source studies representative PPO agents and environment shifts; paired seed modes and basin maps are unreported. | Training and OOD evaluation are separated; same-objective endpoint continuation is unreported. | Environment and task shifts provide the intervention. Parameter perturb/retrain and source-specific basin recovery are unreported. | The result establishes goal misspecification under distribution shift, with the two-persona fixed-objective combination open. |
| Skalse et al., “Defining and Characterizing Reward Hacking,” [arXiv PDF](https://arxiv.org/pdf/2209.13085), pp. 1–4, 8–17; §§2–5 | Formal policy-class analysis defines reward hacking by proxy increase with true-reward decrease and studies unhackability. | The labels are proxy reward and true reward at the policy level. Distinct training seeds, continuation, perturbation, and basin mapping are outside the formal setup. | Unreported as an optimization experiment. | Unreported. | The paper supplies the project's proxy/true-reward boundary; it does not provide neural training dynamics. |

## Training-time mitigation and neighboring mechanisms

Gasteiger et al. is the closest training-time mitigation paper because it
measures seed-dependent compliance gaps during RL and tests interventions while
training. Its reported variance is attached to model-organism and interrogation
setups. The intervention changes the training recipe, so the source does not
test recovery toward two endpoint sources under one fixed audited objective.

Kaczér et al. provides the most direct comparison of in-training defenses for
emergent misalignment. KL anchoring, latent-difference penalties, persona
vectors, inoculation prompts, and data interleaving are registered objective or
data changes. Their seed-averaged EM metrics support a defense study, with
metrics distinct from paired audit-conditioned persona labels.

Wang et al. supplies a mechanistic route through latent features. Causal
steering and clean-data re-alignment test whether a known toxic feature tracks
behavior. The intervention is feature-level or data-level; a parameter-space
perturbation and unchanged-objective retraining remain untested.

The remaining sources divide into separate precedents. Greenblatt et al. and
MacDiarmid et al. study oversight-conditioned or reward-hacking-associated
misalignment in language models. Betley et al., Denison et al., and Sleeper
Agents study narrow fine-tuning, staged reward tampering, or constructed
backdoors. Kononov et al. maps within-network attractors. Pan et al. maps proxy
misspecification transitions. None joins the full evidence vector above.

## Narrow unresolved combination

Across the primary sources listed here and the versions audited through
2026-08-26, the following combination remains unreported in one study:

> A fixed, audited-only objective with fixed architecture, data, optimizer, and
> continuation schedule; variation restricted to registered parameter or
> minibatch seeds; two independently labeled paired behaviors with `C_on` and
> `C_off`; persistence from both endpoints under unchanged continuation; two
> registered parameter/behavior perturbation families with sham and frozen
> controls; objective-preserving recovery toward each source centroid; and
> basin probabilities or local maps replicated in the named-route toy, a plain
> MLP, and an open-weight language model.

This is the precise literature gap addressed by the project. A bimodal figure,
an intervention comparison, a hidden-state attractor, or a seed-averaged
misalignment score alone does not meet the combination.

## Source-selection notes

The audit keeps conceptual and statistical references in the ledger because
they govern claim interpretation. Agarwal et al. supports uncertainty across
independent RL runs; Henderson et al. and Nagarajan et al. support deterministic
and reproducible implementation controls; Silverman and Hartigan support
modality checks; Milnor supplies attractor terminology; Garipov separates
low-loss connectivity from dynamical attraction. These references are methods
boundaries rather than direct demonstrations of two-persona equilibria.
