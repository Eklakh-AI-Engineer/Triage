"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useWorkbench } from "@/lib/store";
import { Gauge, Panel, SectionTitle } from "./ui";
import type { EvidenceCandidate } from "@/lib/types";

const VERDICT_STYLE = {
  ADMIT: { color: "#10b981", tag: "ADMIT", label: "Admit / Escalate" },
  OBSERVE: { color: "#f59e0b", tag: "OBSERVE", label: "Observe / Filter" },
  DISMISS: { color: "#ef4444", tag: "DISMISS", label: "Abstain / Dismiss" },
} as const;

function CandidateRow({ c, index }: { c: EvidenceCandidate; index: number }) {
  const v = VERDICT_STYLE[c.verdict];
  return (
    <motion.div
      initial={{ opacity: 0, x: -14 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: Math.min(index * 0.04, 0.6), duration: 0.3 }}
      className="flex justify-between items-center gap-2 px-2 py-1.5 rounded-lg my-1 text-[11.5px] font-mono border-l-[3px]"
      style={{
        borderLeftColor: v.color,
        background: `${v.color}0f`,
      }}
    >
      <span className="truncate text-slate-300" title={c.text}>
        {c.text}
      </span>
      <span className="flex items-center gap-2 shrink-0">
        <span
          className="text-[9.5px] px-1.5 py-0.5 rounded-md"
          style={{ color: v.color, background: `${v.color}1a` }}
        >
          {v.tag}
        </span>
        <span className="text-slate-500">{c.score.toFixed(2)}</span>
      </span>
    </motion.div>
  );
}

function StageShell({
  title,
  subtitle,
  active,
  children,
}: {
  title: string;
  subtitle?: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`border rounded-xl p-3.5 bg-slate-950/45 transition-all duration-500 ${
        active ? "border-glow/40 shadow-[0_0_24px_rgba(0,242,254,0.08)]" : "border-slate-800/80"
      }`}
    >
      <h3 className="text-[11px] text-slate-500 tracking-[0.1em] mb-2.5 font-mono">
        {title}
        {subtitle && <span className="text-slate-600"> · {subtitle}</span>}
      </h3>
      {children}
    </div>
  );
}

export function TriageRoom() {
  const phase = useWorkbench((s) => s.phase);
  const result = useWorkbench((s) => s.result);
  const status = useWorkbench((s) => s.status);
  const error = useWorkbench((s) => s.error);

  const revealed =
    phase === "retrieving" ? 1 : phase === "decoupling" ? 2 : phase === "gating" || phase === "done" ? 3 : 0;

  const candidates = result?.rawCandidates ?? [];
  const dec = result?.decoupling;
  const gate = result?.gate;
  const passed = gate?.decision === "PASS";

  return (
    <Panel>
      <div className="flex items-center justify-between">
        <SectionTitle>Triage Room — Evidence Flow &amp; Gate Visualizer</SectionTitle>
        {result && (
          <span className="text-[10.5px] font-mono text-slate-600 mb-4">
            latency {result.latencyMs} ms
          </span>
        )}
      </div>

      {error && (
        <div className="mb-3 rounded-xl border border-dismiss/60 bg-dismiss/10 text-dismiss text-[12px] font-mono px-3 py-2">
          pipeline error: {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <StageShell
          title="STAGE 1 — RAW CANDIDATES"
          subtitle={candidates.length > 0 ? `Top-K = ${candidates.length}` : "Top-K = 20"}
          active={status === "running" && revealed === 1}
        >
          <div className="max-h-[300px] overflow-y-auto pr-1">
            {candidates.length === 0 && (
              <span className="text-[11px] font-mono text-slate-600">
                awaiting intake…
              </span>
            )}
            {candidates.map((c, i) => (
              <CandidateRow key={c.id} c={c} index={i} />
            ))}
          </div>
        </StageShell>

        <StageShell
          title="STAGE 2 — ORTHOGONAL DECOUPLING"
          subtitle="Soft Gram-Schmidt"
          active={status === "running" && revealed === 2}
        >
          {dec ? (
            <div className="text-[12px] text-slate-400 font-mono leading-[2]">
              E-channel: <b className="text-node-drug">{dec.eChannel}</b> entity vectors
              <br />
              R-channel: <b className="text-observe">{dec.rChannel}</b> orthogonal relations
              <br />
              M-channel: <b className="text-node-phenotype">{dec.mChannel}</b> modifier cues
              <br />
              <span className="text-slate-600">
                {dec.cancelled} redundant vectors cancelled (η={dec.alpha.toFixed(2)})
              </span>
              <Gauge
                className="mt-2"
                value={(candidates.length - dec.cancelled) / Math.max(1, candidates.length)}
                color="#f59e0b"
              />
              <div className="mt-1.5 text-[10px] text-slate-600 leading-snug">
                Duplicate clinical vectors cancelled into E / R / M sub-channels before the gate.
              </div>
            </div>
          ) : (
            <span className="text-[11px] font-mono text-slate-600">
              decoupling idle…
            </span>
          )}
        </StageShell>

        <StageShell
          title="STAGE 3 — DUAL DECISION GATE"
          subtitle="JEV + Entropy"
          active={status === "running" && revealed === 3}
        >
          {gate ? (
            <>
              <div className="text-[12px] text-slate-400 font-mono">
                JEV score{" "}
                <b className={passed ? "text-admit" : "text-dismiss"}>
                  {gate.jev.toFixed(2)}
                </b>{" "}
                · entropy delta{" "}
                <b className="text-observe">{gate.entropyDelta.toFixed(2)}</b>
                <br />
                <span className="text-slate-600">
                  τ={gate.tauEntropy.toFixed(2)} · {gate.admittedCount} admitted ·{" "}
                  {gate.contextWords} w context
                </span>
              </div>
              <Gauge className="mt-2" value={gate.jev} color={passed ? "#10b981" : "#ef4444"} />
              <AnimatePresence mode="wait">
                <motion.div
                  key={gate.decision}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`mt-3 py-2.5 rounded-xl text-center font-mono text-[12px] border pulse-dot ${
                    passed
                      ? "border-admit text-admit bg-admit/10"
                      : "border-dismiss text-dismiss bg-dismiss/10"
                  }`}
                >
                  {passed
                    ? "PASS — SYNTHESIZE WITH CITED EVIDENCE"
                    : "ABSTAIN — SUPPRESS RETRIEVED DISTRACTORS · PARAMETRIC MODE"}
                </motion.div>
              </AnimatePresence>
            </>
          ) : (
            <span className="text-[11px] font-mono text-slate-600">
              gate idle…
            </span>
          )}
        </StageShell>
      </div>
    </Panel>
  );
}
