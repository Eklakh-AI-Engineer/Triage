"use client";

import { useEffect, useState } from "react";
import { Arena, CitationFlyout } from "./Arena";
import { BenchmarkTelemetry } from "./BenchmarkTelemetry";
import { GateConfigDrawer } from "./GateConfigDrawer";
import { HudHeader } from "./HudHeader";
import { KnowledgeGraph } from "./KnowledgeGraph";
import { QueryDeck } from "./QueryDeck";
import { TriageRoom } from "./TriageRoom";
import { useWorkbench } from "@/lib/store";
import { STANDALONE } from "@/lib/api";

export function Workbench() {
  const [gateConfigOpen, setGateConfigOpen] = useState(false);
  const loadStaticTelemetry = useWorkbench((s) => s.loadStaticTelemetry);

  useEffect(() => {
    loadStaticTelemetry();
  }, [loadStaticTelemetry]);

  return (
    <main className="w-full max-w-[1240px] mx-auto flex-1 px-6 lg:px-10 py-8 lg:py-10">
      <HudHeader />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <QueryDeck onOpenGateConfig={() => setGateConfigOpen(true)} />
        <TriageRoom />
      </div>

      <div className="mt-4">
        <Arena />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 mt-4">
        <KnowledgeGraph />
        <BenchmarkTelemetry />
      </div>

      <footer className="mt-10 pb-2 flex flex-wrap gap-x-2 gap-y-1 text-[11px] font-mono text-faint">
        <span>TRIAGE</span>
        <span>·</span>
        <span>
          {STANDALONE
            ? "standalone mode (mock telemetry)"
            : "live backend mode"}
        </span>
        <span>·</span>
        <span>
          {STANDALONE
            ? "set NEXT_PUBLIC_TRIAGE_API_BASE to wire the Python backend"
            : "api: /api/triage/query · /api/triage/subgraph · /api/triage/benchmark"}
        </span>
      </footer>

      <GateConfigDrawer
        open={gateConfigOpen}
        onOpenChange={setGateConfigOpen}
      />
      <CitationFlyout />
    </main>
  );
}
