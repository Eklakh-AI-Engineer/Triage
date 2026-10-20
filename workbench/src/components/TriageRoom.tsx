"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useWorkbench } from "@/lib/store";
import { Gauge, Panel } from "./ui";
import type { EvidenceCandidate } from "@/lib/types";

const VERDICT_STYLE = {
  ADMIT: { color: "#3e8e6b", label: "Admit" },
  OBSERVE: { color: "#c08a2d", label: "Observe" },
  DISMISS: { color: "#c4564a", label: "Dismiss" },
} as const;

function CandidateRow({ c, index }: { c: EvidenceCandidate; index: number }) {
  const v = VERDICT_STYLE[c.verdict];
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.03, 0.5), duration: 0.25 }}
      className="flex justify-between items-center gap-3 px-2.5 py-1.5 rounded-lg my-0.5 text-[12px] border-l-2"
      style={{ borderLeftColor: v.color, background: `${v.color}08` }}
    >
      <span className="truncate text-ink" title={c.text}>
        {c.text}
      </span>
      <span className="flex items-center gap-2.5 shrink-0">
        <span className="text-[10px] font-mono uppercase tracking-wide" style={{ color: v.color }}>
          {v.label}
        </span>
        <span className="font-mono text-[11px] text-faint w-8 text-right">
          {c.score.toFixed(2)}
        </span>
      </span>
    </motion.div>
  );
}

function Stage({
  n,
  title,
  subtitle,
  children,
}: {
  n: number;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-line bg-canvas p-4">
      <div className="flex items-baseline gap-2 mb-3">
        <span className="font-mono text-[10px] text-faint">0{n}</span>
        <h3 className="text-[13px] font-medium text-ink">{title}</h3>
        <span className="font-mono text-[10px] text-faint ml-auto">{subtitle}</span>
      </div>
      {children}
    </div>
  );
}

export function TriageRoom() {
  const result = useWorkbench((s) => s.result);
  const error = useWorkbench((s) => s.error);

  const candidates = result?.rawCandidates ?? [];
  const dec = result?.decoupling;
  const gate = result?.gate;
  const passed = gate?.decision === "PASS";

  return (
    <Panel className="flex flex-col">
      <div className="flex items-baseline justify-between mb-4">
        <div>
          <h2 className="font-display text-[17px] font-medium tracking-tight text-ink">
            Triage room
          </h2>
          <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-faint mt-0.5">
            Evidence flow &amp; gate visualizer
          </span>
        </div>
        {result && (
          <span className="font-mono text-[10.5px] text-faint">
            {result.latencyMs} ms
          </span>
        )}
      </div>

      {error && (
        <div className="mb-3 rounded-lg border border-dismiss/30 bg-dismiss/5 text-dismiss text-[12px] px-3 py-2">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 flex-1">
        <Stage
          n={1}
          title="Raw candidates"
          subtitle={candidates.length > 0 ? `top ${candidates.length}` : "top 20"}
        >
          <div className="max-h-[300px] overflow-y-auto pr-1 -mr-1">
            {candidates.length === 0 && (
              <span className="text-[11.5px] font-mono text-faint">
                awaiting intake
              </span>
            )}
            {candidates.map((c, i) => (
              <CandidateRow key={c.id} c={c} index={i} />
            ))}
          </div>
        </Stage>

        <Stage n={2} title="Orthogonal decoupling" subtitle="soft Gram-Schmidt">
          {dec ? (
            <div className="text-[12.5px] text-ink leading-7">
              <div className="flex justify-between">
                <span className="text-muted">Entity vectors</span>
                <span className="font-mono">{dec.eChannel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Orthogonal relations</span>
                <span className="font-mono">{dec.rChannel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Modifier cues</span>
                <span className="font-mono">{dec.mChannel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Cancelled as redundant</span>
                <span className="font-mono">{dec.cancelled}</span>
              </div>
              <Gauge
                className="mt-3"
                value={
                  (candidates.length - dec.cancelled) /
                  Math.max(1, candidates.length)
                }
                color="#75726a"
              />
              <div className="mt-2 text-[10.5px] text-faint leading-snug">
                Duplicate clinical vectors cancelled into E / R / M sub-channels
                before the gate.
              </div>
            </div>
          ) : (
            <span className="text-[11.5px] font-mono text-faint">
              decoupling idle
            </span>
          )}
        </Stage>

        <Stage n={3} title="Dual decision gate" subtitle="JEV + entropy">
          {gate ? (
            <>
              <div className="text-[12.5px] leading-7">
                <div className="flex justify-between">
                  <span className="text-muted">JEV score</span>
                  <span
                    className="font-mono"
                    style={{ color: passed ? "#3e8e6b" : "#c4564a" }}
                  >
                    {gate.jev.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Entropy delta</span>
                  <span className="font-mono">{gate.entropyDelta.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Admitted / context</span>
                  <span className="font-mono">
                    {gate.admittedCount} · {gate.contextWords} w
                  </span>
                </div>
              </div>
              <Gauge
                className="mt-3"
                value={gate.jev}
                color={passed ? "#3e8e6b" : "#c4564a"}
              />
              <AnimatePresence mode="wait">
                <motion.div
                  key={gate.decision}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`mt-4 py-2 rounded-lg text-center text-[12px] font-medium border ${
                    passed
                      ? "border-admit/30 bg-admit/5 text-admit"
                      : "border-dismiss/30 bg-dismiss/5 text-dismiss"
                  }`}
                >
                  {passed
                    ? "Pass — synthesize with cited evidence"
                    : "Abstain — parametric mode, distractors suppressed"}
                </motion.div>
              </AnimatePresence>
            </>
          ) : (
            <span className="text-[11.5px] font-mono text-faint">gate idle</span>
          )}
        </Stage>
      </div>
    </Panel>
  );
}
