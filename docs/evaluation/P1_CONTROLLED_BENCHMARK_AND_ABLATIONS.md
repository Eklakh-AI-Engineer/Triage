# P1 controlled benchmark, ablations, and independent clinical test set

## Controlled three-system run

Run:

    python -m src.evaluation.controlled_benchmark --questions data/evaluation/clinical_test_set.jsonl

The runner evaluates the same ordered question IDs against Baseline LLM, Vanilla RAG, and TRIAGE. It refuses to run without the P0 corpus/index integrity manifest and records its hash, question-file hash, configured model identifiers, gate mode, tensor mode, seed, per-question answers, retrieval latency, end-to-end latency, and failures. Outputs are written to results/controlled_benchmark/per_question.jsonl and summary.json.

For a publishable comparison, pin exact model IDs/revisions, prompt versions, temperature, context budget, retrieval top-k, gate mode and tensor mode in a version-controlled experiment config. Run all systems on identical questions and use paired question-level comparisons. Bootstrap 95% confidence intervals are reported for latency and failure-rate summaries; quality/factuality intervals must be calculated from independent labels, not generated-answer judge verdicts. Preserve every output and failure.

## Ablation matrix

Run each row on the exact same held-out questions, with the same models, prompts, context budgets, and gate mode. Change only the named factor. Record retrieval quality, claim-level factuality, abstention, retrieval and end-to-end latency, failures, and cost.

| Run | Change from full TRIAGE |
|---|---|
| Full system | All components enabled |
| No semantic decoupling | Disable entity/relation/modifier decoupling only |
| No graph/hypergraph expansion | Disable graph expansion only |
| No adaptive tensor | Use static/default weights instead of learned/query-adaptive weights |
| No statistical filter | Bypass statistical filtering only |
| No decision gate | Bypass decision gate only; keep retrieval pool and generation config fixed |
| Vector-only retrieval | Disable graph/hypergraph contributions; keep embedding model and top-k fixed |

Use environment/config switches where supported, and add a dedicated switch rather than silently changing multiple settings when a component lacks one. Record the exact config diff for each run. The compare_ablation_bundles helper refuses bundles with mismatched question IDs/order and bootstraps paired deltas for independently computed numeric metrics. The current benchmark runner does not execute the ablation matrix automatically; each configuration must be run separately and then compared.

## Independent clinical test set

data/evaluation/clinical_test_set.jsonl is a starter set authored independently of indexed passages, covering negation, adverse effects, contraindications, temporal qualifiers, multi-hop evidence chains, and unsupported questions. These seed cases are not yet a clinically validated gold set: every item is marked requires_independent_clinician_review and has no fabricated gold label. Before treating results as clinical evidence, have qualified annotators add claim-level labels, acceptable evidence criteria, answerability/abstention targets, and adjudication notes. Preserve source and annotation provenance.

## Completion gate

Do not mark P1 complete until:
1. All three systems have per-question outputs for the same held-out IDs.
2. Exact model/prompt/context/gate configuration and all artifact hashes are in the manifest.
3. Quality metrics use independent annotations; 95% CIs and failure rates are reported.
4. Each ablation has a measured paired effect for retrieval, factuality, abstention, latency, and cost.
5. Independent clinical labels have been reviewed/adjudicated and the test set is not used for tuning.
