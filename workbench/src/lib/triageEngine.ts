import type { ClinicalCase } from "./cases";
import type {
  CandidateVerdict,
  DecouplingReport,
  EvidenceCandidate,
  GateDecision,
  GateConfig,
  TriageQueryResponse,
} from "./types";
import { extractEntities } from "./cases";

// ---------------------------------------------------------------------------
// Standalone deterministic simulation of the TRIAGE pipeline:
//   Stage 1  retrieval over PrimeKG + PMC (anisotropic tensor scores)
//   Stage 2  soft Gram-Schmidt orthogonal decoupling (alpha)
//   Stage 3  dual decision gate (JEV + entropy, tau) -> PASS / ABSTAIN
// Mirrors src/core/hcad_retriever.py + src/core/statistical_filter.py logic
// for demo purposes; the real computation runs on the Python backend.
// ---------------------------------------------------------------------------

const MODE_SENSITIVITY: Record<GateConfig["mode"], number> = {
  strict: 1.0,
  lenient: 0.55,
  stress: 0.2,
};

function clamp(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x));
}

export function simulateTriage(
  query: string,
  config: GateConfig,
  clinicalCase: ClinicalCase
): TriageQueryResponse {
  const sensitivity = MODE_SENSITIVITY[config.mode];
  const tau = config.tauEntropy;
  const alpha = config.alphaDecoupling;

  // Stress mode injects adversarial PubMed distractors into the raw pool:
  // lexically similar to the query, semantically null. They survive
  // retrieval and decoupling but are caught by the judge-ensemble.
  const stressNoise = config.mode === "stress" ? 6 : 0;
  const rawPool = [
    ...clinicalCase.candidates.map((c) => ({ ...c })),
    ...Array.from({ length: stressNoise }, (_, i) => ({
      text: `adversarial distractor #${i + 1} — lexically similar, semantically null`,
      score: 0.66 + i * 0.02,
      source: "PMC" as const,
      redundancy: 0.3,
    })),
  ];

  // Stage 2: soft Gram-Schmidt decoupling. Higher alpha cancels more
  // redundant vectors; the admission bar rises with tau in strict mode.
  const admitBar = 0.72 - tau * sensitivity * 0.12;
  const observeBar = 0.45;

  const candidates: EvidenceCandidate[] = rawPool.map((c, i) => {
    const cancelled = c.redundancy * alpha > 0.42;
    const effective = cancelled ? c.score * (1 - alpha * 0.5) : c.score;
    let verdict: CandidateVerdict = "DISMISS";
    if (effective > admitBar) verdict = "ADMIT";
    else if (effective > observeBar) verdict = "OBSERVE";
    return {
      id: `cand-${i}`,
      text: c.text,
      score: c.score,
      source: c.source,
      redundancy: c.redundancy,
      verdict,
      jevContribution: verdict === "ADMIT" ? effective : undefined,
    };
  });

  const admitted = candidates.filter((c) => c.verdict === "ADMIT");
  const observed = candidates.filter((c) => c.verdict === "OBSERVE");
  const cancelled = candidates.length - admitted.length - observed.length;

  const decoupling: DecouplingReport = {
    eChannel: Math.max(1, Math.min(6, admitted.filter((c) => c.source === "PrimeKG").length)),
    rChannel: Math.max(1, Math.min(7, Math.round(admitted.length * 0.75))),
    mChannel: Math.max(1, Math.min(4, Math.round(admitted.length * 0.5))),
    cancelled,
    alpha,
  };

  // Stage 3: dual decision gate. JEV falls as tau rises (stricter abstention)
  // and as the admitted pool weakens. Adversarial distractors that survived
  // retrieval are detected by the judge-ensemble: they never credit JEV,
  // their presence raises entropy, and trust decays with tau.
  const noiseAdmitted = admitted.filter((c) =>
    c.text.startsWith("adversarial")
  ).length;
  const cleanAdmitted = admitted.filter(
    (c) => !c.text.startsWith("adversarial")
  );
  const meanAdmitted =
    cleanAdmitted.length > 0
      ? cleanAdmitted.reduce((a, c) => a + c.score, 0) / cleanAdmitted.length
      : 0;
  const jev = clamp(
    meanAdmitted - tau * sensitivity * 0.42 - noiseAdmitted * 0.35 * tau,
    0,
    1
  );
  const entropyDelta = -(
    0.11 +
    alpha * 0.05 +
    cancelled * 0.004
  ) + noiseAdmitted * tau * 0.12;
  const passed = jev > 0.45 && cleanAdmitted.length >= 2;

  const gate: GateDecision = {
    decision: passed ? "PASS" : "ABSTAIN",
    jev: Number(jev.toFixed(2)),
    entropyDelta: Number(entropyDelta.toFixed(3)),
    tauEntropy: tau,
    admittedCount: cleanAdmitted.length,
    contextWords: passed
      ? Math.min(config.wordBudget, cleanAdmitted.length * 88)
      : 0,
  };

  const abstentionRate = Number(
    (
      { strict: 11.4, lenient: 18.4, stress: 25.4 }[config.mode] +
      tau * 20 +
      (passed ? 0 : 4)
    ).toFixed(1)
  );

  const entities = extractEntities(query).map((e, i) => ({
    id: `ent-${i}`,
    name: e.name,
    type: e.type,
    confidence: 0.82 + ((i * 13) % 15) / 100,
  }));

  return {
    queryId: `q-${Date.now().toString(36)}`,
    query,
    entities,
    rawCandidates: candidates,
    decoupling,
    gate,
    arena: clinicalCase.arena,
    citations: clinicalCase.citations,
    abstentionRate,
    latencyMs: 780 + ((candidates.length * 31) % 420),
    substrate: { name: "PrimeKG", nodes: 129262, edges: 4000956 },
  };
}
