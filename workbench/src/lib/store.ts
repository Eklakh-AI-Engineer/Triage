"use client";

import { create } from "zustand";
import {
  DEFAULT_CONFIG,
  fetchBenchmark,
  fetchSubgraph,
  queryTriage,
} from "./api";
import { extractEntities } from "./cases";
import type {
  BenchmarkResponse,
  Entity,
  GateConfig,
  SubgraphResponse,
  TriageQueryResponse,
} from "./types";

export type RunPhase = "idle" | "retrieving" | "decoupling" | "gating" | "done";

interface WorkbenchState {
  query: string;
  config: GateConfig;
  status: "idle" | "running" | "done";
  /** Index of the deepest pipeline stage currently revealed. */
  phase: RunPhase;
  result: TriageQueryResponse | null;
  subgraph: SubgraphResponse | null;
  benchmark: BenchmarkResponse | null;
  selectedCitation: string | null;
  selectedNode: string | null;
  error: string | null;
  setQuery: (q: string) => void;
  setConfig: (patch: Partial<GateConfig>) => void;
  run: () => Promise<void>;
  loadStaticTelemetry: () => Promise<void>;
  selectCitation: (id: string | null) => void;
  selectNode: (id: string | null) => void;
  liveEntities: Entity[];
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const INITIAL_QUERY =
  "Can Metformin be given to patients with severe renal failure given lactic acidosis risk?";

function liveEntitiesFor(query: string): Entity[] {
  return extractEntities(query).map((e, i) => ({
    id: `live-${i}-${e.name}`,
    name: e.name,
    type: e.type,
    confidence: 0.8,
  }));
}

export const useWorkbench = create<WorkbenchState>((set, get) => ({
  query: INITIAL_QUERY,
  config: DEFAULT_CONFIG,
  status: "idle",
  phase: "idle",
  result: null,
  subgraph: null,
  benchmark: null,
  selectedCitation: null,
  selectedNode: null,
  error: null,
  liveEntities: liveEntitiesFor(INITIAL_QUERY),

  setQuery: (q) => {
    set({ query: q, liveEntities: liveEntitiesFor(q) });
  },

  setConfig: (patch) =>
    set((s) => ({ config: { ...s.config, ...patch } })),

  run: async () => {
    const { query, config } = get();
    set({ status: "running", phase: "retrieving", error: null, selectedCitation: null, selectedNode: null });
    try {
      const started = Date.now();
      const [resultP, subgraphP] = [
        queryTriage({ query, config }),
        fetchSubgraph(query),
      ];
      // Staged pipeline reveal for the visual funnel.
      await sleep(420);
      set({ phase: "decoupling" });
      await sleep(420);
      set({ phase: "gating" });
      const result = await resultP;
      const subgraph = await subgraphP;
      // Let the gate stage hold for a beat before the verdict lands.
      await sleep(Math.max(0, 900 - (Date.now() - started)));
      set({ result, subgraph, phase: "done", status: "done" });
    } catch (e) {
      set({
        status: "idle",
        phase: "idle",
        error: e instanceof Error ? e.message : "pipeline error",
      });
    }
  },

  loadStaticTelemetry: async () => {
    if (get().benchmark === null) {
      try {
        const benchmark = await fetchBenchmark();
        set({ benchmark });
      } catch {
        set({ benchmark: null });
      }
    }
  },

  selectCitation: (id) => set({ selectedCitation: id }),
  selectNode: (id) => set({ selectedNode: id }),
}));
