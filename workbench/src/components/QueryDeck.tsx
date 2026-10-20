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
      ? "Strict gating"
      : config.mode === "lenient"
        ? "Lenient screening"
        : "Stress / adversarial";

  return (
    <div className="panel p-6">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <h2 className="font-display text-[17px] font-medium tracking-tight text-ink">
            Clinical intake
          </h2>
          <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-faint mt-0.5">
            SapBERT live tagging
          </span>
        </div>
        <button
          onClick={onOpenGateConfig}
          className="shrink-0 text-[11.5px] font-mono px-3 py-1.5 rounded-lg border border-line text-muted hover:text-ink hover:border-line-strong transition-colors"
        >
          {modeLabel} · τ {config.tauEntropy.toFixed(2)} · α{" "}
          {config.alphaDecoupling.toFixed(2)}
        </button>
      </div>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        spellCheck={false}
        className="w-full min-h-[88px] bg-canvas border border-line rounded-xl text-ink p-3.5 text-[13.5px] leading-relaxed resize-y focus:outline-none focus:border-line-strong transition"
        placeholder="Enter a clinical question — entities are tagged live…"
      />

      <div className="min-h-[32px] mt-2.5">
        <AnimatePresence mode="popLayout">
          {liveEntities.map((e) => (
            <EntityChip key={e.id} name={e.name} type={e.type} />
          ))}
        </AnimatePresence>
        {liveEntities.length === 0 && (
          <span className="text-[11px] font-mono text-faint">
            no substrate entities recognized yet
          </span>
        )}
      </div>

      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-line">
        <button
          onClick={run}
          disabled={status === "running"}
          className="text-[13px] font-medium px-5 py-2 rounded-lg bg-ink text-canvas hover:opacity-85 transition disabled:opacity-40"
        >
          {status === "running" ? "Triaging…" : "Run triage"}
        </button>
        {status === "running" && (
          <span className="text-[11.5px] font-mono text-faint">
            pipeline streaming…
          </span>
        )}
      </div>
    </div>
  );
}
