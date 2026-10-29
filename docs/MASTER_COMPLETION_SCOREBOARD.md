# HCAD / TRIAGE — Master Completion Scoreboard

**Last reviewed:** 2026-10-29  
**Reviewed main commit:** `f22855319d8945d5aaa4435d7ac1438a3fedaa8f`  
**Scope:** research completion, retrieval integrity, clinical evaluation, reproducibility, reporting.

> Status labels distinguish repository implementation from executed evidence. A script existing in the repository does not mean its experiment has run. No performance improvement is claimed until fresh, reproducible artifacts support it.

## Status legend

- **DONE** — verified from repository/CI evidence.
- **IN PROGRESS** — implementation exists or work is underway, but completion evidence is missing.
- **BLOCKED** — depends on an earlier gate or external clinical annotation/runtime.
- **TODO** — not yet verified complete.
- **NOT CLAIMED** — do not state a result without a reproducible artifact.

## Current verified snapshot

| Item | Status | Evidence / next action |
|---|---|---|
| PRs #2–#4 merged | DONE | Merged into main; current reviewed main SHA is recorded above. |
| CI on main at `f228553` | DONE | GitHub Actions run 23 completed successfully, including dependency contract, notebook JSON validation and pytest. |
| Notebook ref/output cleanup | IN PROGRESS | PR #5 is open at review time; its runtime benchmark is explicitly not claimed. Check [PR #5](https://github.com/Eklakh-AI-Engineer/Triage/pull/5). |
| Retrieval diagnostic runtime | TODO | No fresh 60-question diagnostic artifact verified in this review. |
| Final benchmark / clinical quality | NOT CLAIMED | Historical results are pre-fix and must not be presented as current post-fix evidence. |

## P0 — integrity and development gate

| ID | Task | Status | Acceptance criteria / artifact |
|---|---|---|---|
| P0-1 | Complete notebook cleanup PR #5 | IN PROGRESS | CI succeeds; notebook defaults to main and stale outputs are cleared; merge only after checks pass. |
| P0-2 | Freeze evaluation commit and environment | TODO | Clean checkout; record full SHA, Python/dependency versions, corpus/vector-store identity, seed and question-set hash. |
| P0-3 | Run 60-question dev ranking diagnostics | TODO | Run `python -m src.evaluation.ranking_experiments --samples 60 --pool dev`; save JSON and logs with exact commit provenance. |
| P0-4 | Validate candidate survival at each stage | TODO | Report gold evidence in initial pool, after Φ floor, after filter, after decision gate, and in final top 5. |
| P0-5 | Audit retrieval metric denominator | IN PROGRESS | Make Recall@5 denominator consistently use relevant items in the full candidate pool; add tests for empty pools, truncated rankings, abstention and rankings that omit relevant items. |
| P0-6 | Confirm metric comparability across harnesses | TODO | Dev harness, ranking experiments and final benchmark share documented definitions and identical relevance protocol. |
| P0-7 | Decide fix from diagnostics, not guesswork | BLOCKED | If initial-pool coverage is low, investigate candidate generation/entity linking; if evidence is lost at a later stage, fix that stage; if evidence survives but top-5 misses it, tune ranking. |
| P0-8 | Pass development GO gate | BLOCKED | Same-commit paired evaluation meets documented thresholds: Recall@5 improvement ≥ 0.08 and NDCG@5 improvement ≥ 0.06 against corrected baseline. Save a fresh GO artifact bound to the exact commit. |
| P0-9 | Keep held-out set locked | IN PROGRESS | No held-out/test evaluation until a valid development GO artifact exists for the exact same commit. |

## P1 — retrieval experiments and final benchmark

| ID | Task | Status | Acceptance criteria / artifact |
|---|---|---|---|
| P1-1 | Compare dense-only, graph-only, RRF, HCAD Φ and final HCAD | TODO | Same questions and initial candidate pool; report Recall@5, NDCG@5, MRR@5, paired deltas and 95% bootstrap CIs. |
| P1-2 | Run planned ablation matrix | TODO | Separate arms for no decoupling, no hypergraph, static weights, no filter, no gate and vector-only; prove each intended component is actually disabled. |
| P1-3 | Test proposed ranking changes | BLOCKED | Only after diagnostics: evaluate RRF or dense-anchored Φ; isolate masking/relation-vocabulary effects; then investigate entity linking and graph expansion as indicated. |
| P1-4 | Align the three-system benchmark | TODO | Baseline LLM, Vanilla RAG and HCAD use comparable retrieval metrics; document systems for which retrieval is not applicable. |
| P1-5 | Run final paired benchmark (≥200 questions) | BLOCKED | Frozen configuration; paired questions; confidence intervals, paired significance analysis, failed-judge coverage and negative results reported. |
| P1-6 | Report by source type | TODO | Separate PMC/passage evidence and KG/triple evidence; discuss relevance-label bias. |
| P1-7 | Preserve old 28 Oct results | TODO | Archive as pre-fix baseline with commit SHA, environment, dataset/question-set identity, raw outputs and a clear non-comparable/pre-fix label. |

## P1 — independent clinical and grounding evaluation

| ID | Task | Status | Acceptance criteria / artifact |
|---|---|---|---|
| CL-1 | Expand clinical test set from 12 to approximately 50+ questions | TODO | Document sampling, provenance, inclusion criteria and held-out separation. |
| CL-2 | Obtain qualified independent labels | BLOCKED | Annotator rubric and labels for factuality, citation support/entailment, contradiction, negation and appropriate abstention; adjudicate disagreements. |
| CL-3 | Evaluate citation entailment, not just citation ID validity | TODO | Report citation coverage, entailment precision, unsupported claims and invalid citations separately. |
| CL-4 | Report contradiction/negation errors | TODO | Claim-level rubric, counts and rates with denominator and uncertainty. |
| CL-5 | Measure correct abstention | TODO | Distinguish justified abstention, unjustified abstention and unsupported answering. |
| CL-6 | Add MedHallu evaluation | TODO | Record dataset/version, protocol, system settings, results and limitations; do not mix incomparable scores. |
| CL-7 | Verify clinical-answer scorer behavior | IN PROGRESS | Validate scorer against labelled examples and ensure it fails closed when required labels are missing. |

## P2 — reproducibility, reporting and housekeeping

| ID | Task | Status | Acceptance criteria / artifact |
|---|---|---|---|
| H-1 | Verify NumPy 1.26.4 in Kaggle/runtime | TODO | Fresh environment install and full test/runtime check with pinned dependencies; capture versions and failures. |
| H-2 | Record actual judge and generator models | TODO | Report provider, exact model identifier, prompt/config, fallback behavior and valid-judge coverage. |
| H-3 | Record gate mode | TODO | Distinguish local calibrated gate from live JEV/API-backed gate and record required credentials without exposing secrets. |
| H-4 | Audit model artifacts | TODO | Verify whether `models/gating_weights.pt` and `hcad_rag_model.joblib` are loaded; document purpose or remove only after usage checks. |
| H-5 | Archive raw outputs and manifests | TODO | Versioned JSON/CSV tables, logs, commit SHA, environment, seeds, question-set hash and configuration. |
| H-6 | Update README | TODO | Real post-fix results only; define metrics, dataset, sample sizes, CIs, source breakdown, grounding measures and limitations. |
| H-7 | Update paper/report and viva slides | TODO | Claims match verified evidence; include negative results and no unsupported superiority/equivalence claims. |
| H-8 | Document limitations | TODO | Explicitly discuss LLM-generated questions, Jaccard-based relevance, small clinician-labelled sample, source imbalance and judge limitations. |

## Run protocol

1. Resolve PR #5 and confirm CI is green.
2. Start from a clean checkout of the selected commit; record `git rev-parse HEAD`.
3. Run the dev diagnostics only; archive the JSON and logs.
4. Inspect candidate-survival losses before changing ranking logic.
5. Implement one justified change at a time; run tests and repeat the same paired dev evaluation.
6. Issue GO only when both thresholds pass and the artifact is tied to the exact commit. Otherwise record STOP and the observed failure mode.
7. Freeze the configuration before running held-out evaluation.
8. Align benchmark metrics before comparing the three systems.
9. Complete clinical annotation before making clinical-grounding claims.
10. Update the README/report only from reproducible outputs.

## Reporting guardrails

- Keep the 28 October run labelled **pre-fix baseline**.
- Do not claim that HCAD improves retrieval until the corrected paired evaluation demonstrates it.
- Citation coverage is not citation entailment; a citation ID existing is not proof that the cited evidence supports the claim.
- Do not describe a non-significant result as proof of equivalence.
- Report failed judge calls, abstentions and excluded samples transparently.
- Never unlock held-out evaluation without a valid same-commit development GO artifact.

## Completion definition

TRIAGE is research-complete only when P0 is closed, the final benchmark and ablations are reproducible, independent clinical labels and grounding results are reported, CI and runtime environments are documented, and the README/report/viva claims match the saved artifacts.
