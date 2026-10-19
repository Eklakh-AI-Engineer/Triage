"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { useWorkbench } from "@/lib/store";
import { Panel, SectionTitle } from "./ui";
import type { AnswerSegment, CitationRecord } from "@/lib/types";

function CitationPill({ id }: { id: string }) {
  const selectCitation = useWorkbench((s) => s.selectCitation);
  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.7 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", damping: 18, stiffness: 300 }}
      onClick={() => selectCitation(id)}
      className="inline-block align-baseline mx-0.5 rounded-md px-1.5 py-px text-[10px] font-mono text-glow border border-glow/40 bg-glow/10 hover:bg-glow/20 cursor-pointer"
    >
      {id}
    </motion.button>
  );
}

function AnswerStream({
  segments,
  streaming,
  highlight,
}: {
  segments: AnswerSegment[];
  streaming: boolean;
  highlight?: boolean;
}) {
  const total = segments.reduce(
    (a, s) => a + (s.kind === "text" ? s.value.length : 14),
    0
  );
  const [progress, setProgress] = useState<number>(streaming ? 0 : total);
  const done = progress >= total;

  useEffect(() => {
    if (!streaming || done) return;
    const timer = setInterval(() => {
      setProgress((p) => Math.min(total, p + 4));
    }, 24);
    return () => clearInterval(timer);
  }, [streaming, done, total]);

  let budget = progress;
  const parts: React.ReactNode[] = [];
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    if (s.kind === "citation") {
      if (budget >= 14) {
        budget -= 14;
        parts.push(<CitationPill key={i} id={s.id} />);
      } else {
        break;
      }
    } else {
      if (budget <= 0) break;
      const text = s.value.slice(0, budget);
      budget -= text.length;
      parts.push(
        <span
          key={i}
          className={highlight && s.value.startsWith("[!]") ? "text-dismiss" : undefined}
        >
          {text}
          {text.length < s.value.length && (
            <span className="inline-block w-[7px] h-[13px] bg-glow/80 align-[-2px] ml-px pulse-dot" />
          )}
        </span>
      );
    }
  }
  return <div className="whitespace-pre-wrap">{parts}</div>;
}

export function CitationFlyout() {
  const selectedCitation = useWorkbench((s) => s.selectedCitation);
  const result = useWorkbench((s) => s.result);
  const selectCitation = useWorkbench((s) => s.selectCitation);
  const record: CitationRecord | undefined =
    selectedCitation !== null
      ? result?.citations[selectedCitation]
      : undefined;

  return (
    <Dialog.Root
      open={selectedCitation !== null}
      onOpenChange={(o) => !o && selectCitation(null)}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/40 z-40" />
        <Dialog.Content asChild>
          <motion.div
            initial={{ opacity: 0, y: 24, x: 0 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel fixed bottom-6 right-6 w-[min(92vw,380px)] z-50 p-4 !bg-slate-950/90 border-glow/60 shadow-[0_0_40px_rgba(0,242,254,0.2)]"
          >
            <Dialog.Title className="text-glow font-mono text-[13px]">
              {selectedCitation}
            </Dialog.Title>
            <Dialog.Description className="mt-3 text-[12px] leading-relaxed text-slate-300">
              {record ? (
                <>
                  <span className="block font-mono text-[12.5px] text-slate-200 mb-2">
                    {record.triple}
                  </span>
                  <span className="block text-slate-500 text-[11px] font-mono mb-2">
                    {record.source} · confidence {record.confidence.toFixed(2)}
                  </span>
                  <span className="block text-slate-400">{record.abstract}</span>
                </>
              ) : (
                <span className="text-slate-500 font-mono text-[11.5px]">
                  Citation not present in the admitted context — suppressed or
                  filtered by the gate.
                </span>
              )}
            </Dialog.Description>
            <Dialog.Close
              className="mt-3 w-full font-mono text-[11px] py-1.5 rounded-lg border border-slate-800 text-slate-400 hover:text-glow hover:border-glow transition"
              aria-label="Close"
            >
              close
            </Dialog.Close>
          </motion.div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Arena() {
  const result = useWorkbench((s) => s.result);
  const phase = useWorkbench((s) => s.phase);
  const abstained = result?.gate.decision === "ABSTAIN";

  const arena = result?.arena;
  const triageSegments: AnswerSegment[] = abstained
    ? [
        {
          kind: "text",
          value:
            "[!] ABSTAIN triggered: retrieved evidence failed the dual gate (JEV below floor / entropy above τ). Retrieved distractors are suppressed — relying on clean parametric clinical knowledge only. No citations attached.",
        },
      ]
    : (arena?.triage ?? []);

  const columns = [
    {
      title: "Baseline LLM (No Retrieval)",
      accent: "#64748b",
      segments: arena?.baseline ?? [],
      highlight: true,
    },
    {
      title: "Vanilla Vector-RAG (Top-K)",
      accent: "#f59e0b",
      segments: arena?.vanillaRag ?? [],
      highlight: true,
    },
    {
      title: "TRIAGE (Our Architecture)",
      accent: "#00f2fe",
      segments: triageSegments,
      highlight: false,
    },
  ];

  return (
    <Panel>
      <SectionTitle>Three-Way Clinical Arena</SectionTitle>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {columns.map((col) => (
          <div
            key={col.title}
            className="rounded-xl bg-slate-950/40 border border-slate-800/70 p-4 border-t-[3px]"
            style={{ borderTopColor: col.accent }}
          >
            <h3 className="text-[13px] text-slate-300 mb-2.5">{col.title}</h3>
            <div className="text-[12.5px] leading-[1.7] text-slate-400 max-h-[300px] overflow-y-auto pr-1.5">
              {arena ? (
                col.title.startsWith("TRIAGE") ? (
                  phase === "done" ? (
                    <AnswerStream
                      key={result?.queryId}
                      segments={col.segments}
                      streaming
                    />
                  ) : (
                    <span className="text-slate-600 font-mono text-[11.5px]">
                      {phase === "idle"
                        ? "run TRIAGE to inspect the gated answer…"
                        : "synthesizing gated answer…"}
                    </span>
                  )
                ) : (
                  <AnswerStream segments={col.segments} streaming={false} highlight={col.highlight} />
                )
              ) : (
                <span className="text-slate-600 font-mono text-[11.5px]">
                  awaiting triage run…
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
      {arena?.warnings.length ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {arena.warnings.map((w) => (
            <span
              key={w}
              className="text-[10.5px] font-mono text-dismiss/90 border border-dismiss/40 bg-dismiss/10 rounded-lg px-2 py-1"
            >
              {w}
            </span>
          ))}
        </div>
      ) : null}
    </Panel>
  );
}
