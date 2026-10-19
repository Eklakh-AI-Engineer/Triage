"use client";

import { useEffect, useState } from "react";
import { Arena } from "./Arena";
import { BenchmarkTelemetry } from "./BenchmarkTelemetry";
import { GateConfigDrawer } from "./GateConfigDrawer";
import { HudHeader } from "./HudHeader";
import { KnowledgeGraph } from "./KnowledgeGraph";
import { QueryDeck } from "./QueryDeck";
import { TriageRoom } from "./TriageRoom";
import { CitationFlyout } from "./Arena";
import { useWorkbench } from "@/lib/store";
import { STANDALONE } from "@/lib/api";

export function Workbench() {
  const [gateConfigOpen, setGateConfigOpen] = useState(false);
  const loadStaticTelemetry = useWorkbench((s) => s.loadStaticTelemetry);

  useEffect(() => {
    loadStaticTelemetry();
  }, [loadStaticTelemetry]);

  return (
    <main className="w-full max-w-[1440px] mx-auto flex-1 px-6 py-5">
      <HudHeader />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <QueryDeck onOpenGateConfig={() => setGateConfigOpen(true)} />
        <TriageRoom />
      </div>

      <div className="mt-4">
        <Arena />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
        <KnowledgeGraph />
        <BenchmarkTelemetry />
      </div>

      <footer className="mt-6 mb-2 text-center text-[11px] font-mono text-slate-600">
        TRIAGE workbench ·{" "}
        {STANDALONE
          ? "standalone mode (mock telemetry) — set NEXT_PUBLIC_TRIAGE_API_BASE to wire the Python backend"
          : "live backend mode"}{" "}
        · api: /api/triage/query · /api/triage/subgraph · /api/triage/benchmark
      </footer>

      <GateConfigDrawer
        open={gateConfigOpen}
        onOpenChange={setGateConfigOpen}
      />
      <CitationFlyout />
    </main>
  );
}
