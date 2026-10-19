"use client";

import { AnimatePresence } from "framer-motion";
import { useWorkbench } from "@/lib/store";
import { EntityChip } from "./ui";

export function QueryDeck({ onOpenGateConfig }: { onOpenGateConfig: () => void }) {
  const query = useWorkbench((s) => s.query);
  const setQuery = useWorkbench((s) => s.setQuery);
  const liveEntities = useWorkbench((s) => s.liveEntities);
  const status = useWorkbench((s) => s.status);
  const run = useWorkbench((s) => s.run);
  const config = useWorkbench((s) => s.config);

  const modeLabel =
    config.mode === "strict"
      ? "Strict Gating"
      : config.mode === "lenient"
        ? "Lenient Screening"
        : "Stress / Adversarial";

  return (
    <div className="glass-panel p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-[13px] uppercase tracking-[0.14em] text-slate-500">
          Clinical Intake
        </h2>
        <button
          onClick={onOpenGateConfig}
          className="text-[11px] font-mono px-3 py-1.5 rounded-lg border border-slate-800/80 bg-slate-950/50 text-slate-400 hover:text-glow hover:border-glow transition-colors"
        >
          Gate Config · {modeLabel} · τ={config.tauEntropy.toFixed(2)} · α=
          {config.alphaDecoupling.toFixed(2)}
        </button>
      </div>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        spellCheck={false}
        className="w-full min-h-[84px] bg-slate-950/70 border border-slate-800/80 rounded-xl text-slate-200 p-3 text-sm resize-y focus:outline-none focus:border-glow/60 focus:shadow-[0_0_18px_rgba(0,242,254,0.15)] transition"
        placeholder="Enter clinical question — entities are tagged live via SapBERT…"
      />

      <div className="min-h-[34px] mt-2.5">
        <AnimatePresence mode="popLayout">
          {liveEntities.map((e) => (
            <EntityChip key={e.id} name={e.name} type={e.type} />
          ))}
        </AnimatePresence>
        {liveEntities.length === 0 && (
          <span className="text-[11px] font-mono text-slate-600">
            SapBERT live tagging: no PrimeKG entities recognized yet
          </span>
        )}
      </div>

      <div className="flex items-center gap-3 mt-3">
        <button
          onClick={run}
          disabled={status === "running"}
          className="font-mono text-[12.5px] px-5 py-2.5 rounded-xl border border-glow bg-glow/10 text-glow hover:bg-glow/20 hover:shadow-[0_0_18px_rgba(0,242,254,0.25)] transition disabled:opacity-50 disabled:cursor-wait"
        >
          {status === "running" ? "TRIAGING…" : "RUN TRIAGE"}
        </button>
        {status === "running" && (
          <div className="flex items-end gap-[3px] h-8" aria-hidden>
            {[0, 1, 2, 3, 4].map((i) => (
              <span
                key={i}
                className="w-[3px] rounded-sm bg-glow/80 heartbeat-bar"
                style={{ height: `${14 + i * 4}px`, animationDelay: `${i * 0.12}s` }}
              />
            ))}
            <span className="ml-2 text-[11px] font-mono text-slate-500 self-center">
              pipeline streaming…
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
