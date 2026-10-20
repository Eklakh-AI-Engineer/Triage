"use client";

import { useWorkbench } from "@/lib/store";
import { Badge } from "./ui";
import { availableCases } from "@/lib/api";

export function TriageLogo() {
  return (
    <svg width="34" height="34" viewBox="0 0 40 40" className="drop-shadow-[0_0_8px_rgba(0,242,254,0.8)]">
      <polyline
        points="2,20 12,20 16,8 22,32 26,20 38,20"
        fill="none"
        stroke="#00f2fe"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function HudHeader() {
  const setQuery = useWorkbench((s) => s.setQuery);
  const result = useWorkbench((s) => s.result);
  const abstentionRate = result?.abstentionRate ?? 18.4;

  return (
    <header className="flex flex-wrap items-center gap-x-5 gap-y-3 py-3 pb-6">
      <div className="flex items-center gap-3">
        <TriageLogo />
        <div>
          <b className="text-2xl tracking-[0.06em] font-sans">TRIAGE</b>
          <small className="block text-slate-500 text-[11px]">
            Clinical Gated Evidence Decoupling &amp; Selective Retrieval
          </small>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge dotColor="#10b981">Gate Status: ARMED</Badge>
        <Badge>Substrate: PrimeKG · 129,262 nodes · 4M edges</Badge>
        <Badge>Engine: llama-3.2-11b (NIM)</Badge>
        <Badge>
          Abstention Rate:{" "}
          <b className="text-glow ml-1">{abstentionRate.toFixed(1)}%</b>
        </Badge>
      </div>

      <div className="ml-auto flex flex-wrap gap-2">
        {availableCases().map((c) => (
          <button
            key={c.id}
            onClick={() => setQuery(c.query)}
            className="text-[11px] px-2.5 py-1.5 rounded-lg border border-slate-800/80 bg-slate-950/50 text-slate-500 hover:text-slate-200 hover:border-glow transition-colors font-mono"
          >
            {c.label}
          </button>
        ))}
      </div>
    </header>
  );
}
