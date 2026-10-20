"use client";

import { useWorkbench } from "@/lib/store";
import { availableCases } from "@/lib/api";

export function HudHeader() {
  const setQuery = useWorkbench((s) => s.setQuery);
  const result = useWorkbench((s) => s.result);
  const abstentionRate = result?.abstentionRate ?? 18.4;

  return (
    <header className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4 pb-6">
      <div>
        <h1 className="font-display text-[26px] font-medium tracking-tight leading-none text-ink">
          TRIAGE
        </h1>
        <p className="mt-1.5 text-[12px] text-muted">
          Gated evidence selection &amp; context decoupling for clinical RAG
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 font-mono text-[11px] text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-admit pulse-dot" />
          Gate armed
        </span>
        <span className="text-faint">/</span>
        <span>PrimeKG · 129,262 nodes</span>
        <span className="text-faint">/</span>
        <span>llama-3.2-11b</span>
        <span className="text-faint">/</span>
        <span>
          abstention{" "}
          <b className="font-medium text-ink ml-0.5">
            {abstentionRate.toFixed(1)}%
          </b>
        </span>
      </div>

      <div className="flex flex-wrap gap-2 w-full lg:w-auto">
        {availableCases().map((c) => (
          <button
            key={c.id}
            onClick={() => setQuery(c.query)}
            className="text-[11.5px] px-3 py-1.5 rounded-lg text-muted hover:text-ink hover:bg-surface hover:border-line-strong border border-transparent transition-colors"
          >
            {c.label}
          </button>
        ))}
      </div>
    </header>
  );
}
