import { BENCHMARK_MOCK, CASES, resolveCase } from "./cases";
import { simulateTriage } from "./triageEngine";
import type {
  BenchmarkResponse,
  GateConfig,
  SubgraphResponse,
  TriageQueryRequest,
  TriageQueryResponse,
} from "./types";

// ---------------------------------------------------------------------------
// API client. In standalone mode everything resolves to the local mock
// engine. Set NEXT_PUBLIC_TRIAGE_API_BASE (e.g. http://localhost:8000) to
// route to the Python FastAPI backend implementing the same contracts.
// ---------------------------------------------------------------------------

const BASE = process.env.NEXT_PUBLIC_TRIAGE_API_BASE;

export const STANDALONE = !BASE;

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`TRIAGE backend error ${res.status} on ${path}`);
  }
  return res.json() as Promise<T>;
}

export async function queryTriage(
  req: TriageQueryRequest
): Promise<TriageQueryResponse> {
  if (!STANDALONE) {
    return fetchJson<TriageQueryResponse>("/api/triage/query", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }
  const clinicalCase = resolveCase(req.query);
  return simulateTriage(req.query, req.config, clinicalCase);
}

export async function fetchSubgraph(query: string): Promise<SubgraphResponse> {
  if (!STANDALONE) {
    return fetchJson<SubgraphResponse>(
      `/api/triage/subgraph?query=${encodeURIComponent(query)}`
    );
  }
  const clinicalCase = resolveCase(query);
  return Promise.resolve(clinicalCase.subgraph);
}

export async function fetchBenchmark(): Promise<BenchmarkResponse> {
  if (!STANDALONE) {
    return fetchJson<BenchmarkResponse>("/api/triage/benchmark");
  }
  return Promise.resolve(BENCHMARK_MOCK);
}

export function availableCases() {
  return CASES.map((c) => ({ id: c.id, label: c.label, query: c.query }));
}

export const DEFAULT_CONFIG: GateConfig = {
  tauEntropy: 0.25,
  alphaDecoupling: 0.7,
  wordBudget: 350,
  mode: "strict",
};
