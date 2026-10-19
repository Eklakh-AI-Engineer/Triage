// ---------------------------------------------------------------------------
// TRIAGE workbench API contracts.
// These interfaces mirror the Python backend endpoints:
//   POST /api/triage/query     -> TriageQueryResponse
//   GET  /api/triage/subgraph  -> SubgraphResponse
//   GET  /api/triage/benchmark -> BenchmarkResponse
// ---------------------------------------------------------------------------

export type EntityType =
  | "drug"
  | "disease"
  | "gene"
  | "protein"
  | "phenotype"
  | "adverse_effect";

export interface Entity {
  id: string;
  name: string;
  type: EntityType;
  /** SapBERT-linked PrimeKG node id when resolved against the substrate. */
  nodeId?: string;
  confidence: number;
}

export type CandidateVerdict = "ADMIT" | "OBSERVE" | "DISMISS";

export interface EvidenceCandidate {
  id: string;
  /** Compact clinical fact, usually a (head, relation, tail) triple. */
  text: string;
  /** Anisotropic tensor relevance score in [0, 1]. */
  score: number;
  source: "PrimeKG" | "PMC";
  /** Pairwise cosine redundancy vs. already-admitted vectors. */
  redundancy: number;
  verdict: CandidateVerdict;
  /** JEV contribution of this candidate after decoupling. */
  jevContribution?: number;
}

export interface DecouplingReport {
  eChannel: number;
  rChannel: number;
  mChannel: number;
  cancelled: number;
  alpha: number;
}

export interface GateDecision {
  decision: "PASS" | "ABSTAIN";
  /** Judge-Ensemble Verification score in [0, 1]. */
  jev: number;
  /** Entropy delta of the judge distribution; negative = converged. */
  entropyDelta: number;
  tauEntropy: number;
  admittedCount: number;
  contextWords: number;
}

export interface CitationRecord {
  id: string;
  triple: string;
  source: string;
  confidence: number;
  abstract: string;
}

export type AnswerSegment =
  | { kind: "text"; value: string }
  | { kind: "citation"; id: string };

export interface ArenaResponse {
  baseline: AnswerSegment[];
  vanillaRag: AnswerSegment[];
  triage: AnswerSegment[];
  warnings: string[];
}

export interface GateConfig {
  /** Abstention sensitivity tau_entropy, in [0.05, 0.6]. */
  tauEntropy: number;
  /** Decoupling strength alpha for soft Gram-Schmidt, in [0, 1]. */
  alphaDecoupling: number;
  /** Evidence word budget, 250-1000. */
  wordBudget: number;
  mode: TriageMode;
}

export type TriageMode = "strict" | "lenient" | "stress";

export interface TriageQueryRequest {
  query: string;
  config: GateConfig;
}

export interface TriageQueryResponse {
  queryId: string;
  query: string;
  entities: Entity[];
  rawCandidates: EvidenceCandidate[];
  decoupling: DecouplingReport;
  gate: GateDecision;
  arena: ArenaResponse;
  citations: Record<string, CitationRecord>;
  abstentionRate: number;
  latencyMs: number;
  substrate: {
    name: string;
    nodes: number;
    edges: number;
  };
}

export interface GraphNode {
  id: string;
  label: string;
  type: EntityType;
  /** Aggregated tensor trace score contribution, in [0, 1]. */
  trace: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface SubgraphResponse {
  caseId: string;
  centerId: string;
  hops: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface BenchmarkResponse {
  radar: {
    dimension: string;
    triage: number;
    vanilla: number;
    baseline: number;
  }[];
  winRate: {
    criterion: string;
    rate: number;
  }[];
  mcnemar: {
    statistic: number;
    pValue: number;
    significant: boolean;
    n: number;
  };
  abstentionRate: number;
  nQueries: number;
}
