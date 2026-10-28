# P0 API telemetry and evidence-boundary contract

- gate.scoreSpread is the non-negative range of observed admitted evidence scores: max(score) - min(score), or zero for fewer than two contexts. It is not entropy and is not named entropyDelta.
- gate.output and gate.decision report PASS when at least one budget-compliant context is admitted, otherwise ABSTAIN. candidateCount and admittedCount expose the observed candidate and post-budget counts.
- retrievalLatencyMs measures TRIAGE retrieval only. endToEndLatencyMs measures the complete API query through all three arena generations and response assembly. latencyMs remains a backwards-compatible alias for endToEndLatencyMs.
- gate.contextWords reports the actual word count passed to TRIAGE generation; gate.wordBudget records the configured request limit.
- API and DecisionGate evidence budgeting retains only complete sentences. If a sentence does not fit, it is omitted; if no complete sentence fits, the context is dropped. This intentionally prefers missing evidence over a fragment that could remove negation, contraindication, timing, or other clinical qualifiers.

Unit tests assert the score-spread definition, latency fields, and sentence-safe preservation of clinical qualifiers.
