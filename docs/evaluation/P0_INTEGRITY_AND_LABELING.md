# P0 evaluation integrity contract

This change makes the benchmark fail closed when the corpus/index integrity manifest is absent and introduces a strict schema for that manifest. The build pipeline must write `data/processed/integrity_manifest.json` (or set `TRIAGE_INTEGRITY_MANIFEST`) after a complete index build.

## Artifact manifest

Use `src.evaluation.integrity.build_integrity_manifest` with:
- SHA-256 and byte size for every source input file.
- exact indexed-document count and expected count when known;
- relation counts/coverage and corpus composition counts;
- embedding model, index name/version, and build version.

Write it with `write_integrity_manifest`. Benchmark startup calls `validate_integrity_manifest` and refuses to run if the manifest is missing, malformed, incomplete, or has a stale source hash. A manifest does not by itself prove clinical correctness: compare indexed counts with the source-derived expected count and retain the raw build logs.

## Judge failures and primary aggregates

A scoring fallback is identifiable from its existing rationale prefix, `Fallback due to judge failure:`. Such records remain in per-question checkpoints for auditability, but are excluded from primary mean scores. The run manifest reports per-system question count, valid judge count, fallback count, and judge coverage. Inspect checkpoint rationales to trace individual failures. Parse recovery remains visible in evaluator logs; future improvements should persist raw judge responses and retry/parse events as structured per-call events.

## Hallucination detector vs generated-answer factuality

These are different tasks and must not share a confusion matrix:
- **Detector benchmark:** fixed examples with independently known labels. The model predicts `HALLUCINATED` or `FAITHFUL`; compute detector precision/recall/F1 only against those labels.
- **Generated-answer claim factuality:** each atomic claim must have an independent annotation. The new `src.evaluation.claim_factuality` module requires `claim_id`, `question_id`, `annotator_id`, claim text, and one of `supported`, `contradicted`, or `insufficient_evidence`. For binary metrics, the latter two count as unsupported. Missing predictions are excluded and reported as missing, never counted as correct.

Do not interpret model self-classifications of its own generated answers as ground truth or publish conventional confusion-matrix metrics without independent labels. Clinical claim annotations should be reviewed by qualified annotators, and disagreements/adjudication should be retained in the dataset provenance.
