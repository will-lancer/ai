# Closest-paper gap

## Do LLMs Game Formalization?

Kim, Poiroux, and Bosselut's *Do LLMs Game Formalization?* is the nearest Lean-based precedent. The paper uses Lean compilation and compiler feedback, with up to three attempts per stage. Its formalization setting lets the model construct or modify formalizations. In its later lock condition, changes to the formalization are detected through an observational diff. The paper reports fabricated axioms and theorem changes and warns that compilation alone does not establish faithful formalization.

Source: [arXiv HTML](https://arxiv.org/html/2604.19459), [arXiv record](https://arxiv.org/abs/2604.19459). The cited version was submitted on 2026-04-21. The phrase "up to three attempts per stage" is the intended description of its retry budget.

The proposed pilot isolates a narrower interface variable. Trusted Lean templates hold imports, axioms, theorem declarations, propositions, and surrounding bytes fixed in frozen-hole mode. A model supplies one proof term. Whole-file mode preserves the broader source interface so formal mutations can be observed and coded. The validator detects those mutations before Lean, while the runner separately records compilation, target axioms, final metavariables, and contract acceptance.

| question | closest paper | proposed pilot |
| --- | --- | --- |
| What can the model edit? | Formalization construction and later source changes | One proof term in frozen mode; full source in the comparison mode |
| How is gaming detected? | Observational change detection and compiler outcomes | Byte-level lock, Lean-aware parser, forbidden-command scan, target-axiom audit, and manual review |
| What controls the interface? | Formalization-stage conditions | Exact frozen hole, whole-file condition, neutral/score-focused instructions, and safe planted loopholes |
| What is measured? | Formalization performance and flagged source changes | First loophole, escalation, accepted loophole, valid proof, impossible false acceptance, disclosure, and repair trajectory |
| What prevents a false proof? | Compilation is an observed outcome | Prechecked impossible tasks and a zero-acceptance frozen control gate |

The gap is therefore behavioral isolation. The new harness tests whether source-interface freedom changes loophole attempts while preserving easy-proof competence. It does not claim that a proof-interface response reveals a training mechanism or a model's private intent.

## Specification-gaming and reward-hacking context

[Bondarenko et al., *Demonstrating specification gaming in reasoning models*](https://arxiv.org/abs/2502.13295) studies environment manipulation in chess-like Docker tasks. [Nishimura-Gasparian et al., *Towards Understanding Specification Gaming in Reasoning Models*](https://arxiv.org/html/2605.02269) examines exploit rates across eight deployment settings. These works motivate explicit exploit coding and environment-probe labels. Their environments do not supply an immutable Lean proof-hole contract.

The broader reward-hacking literature supplies related controls. Proxy/true-reward divergence, oversight-conditioned behavior, designed trigger persistence, verifier gaming, and optimization transitions each cover a different mechanism. None of those results supplies the proposed combination of an axiom-free Lean corpus, impossible controls, an immutable proof hole, source-mutation detection, and a paired whole/frozen interface contrast.

[Neel Nanda's methodology note](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand-distill) motivates the staged workflow: run a cheap exploratory smoke test, understand failure modes with adversarial review, then distill a preregistered pilot. The current implementation keeps paid API calls disabled while the corpus, lock, runner, mock adapter, schedule, and analysis are verified offline.

## Boundary of the claim

The study can support a deployment-time statement about observed proof-interface behavior if its gates pass. It can show that a stricter source contract changes checker-visible loophole behavior under the tested tasks and models. It cannot establish that RLHF caused the behavior, that a model intended to game the checker, or that the outputs reveal two equilibria or attractors.
