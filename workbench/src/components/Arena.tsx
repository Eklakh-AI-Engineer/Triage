"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { useWorkbench } from "@/lib/store";
import { Panel } from "./ui";
import type { AnswerSegment, CitationRecord } from "@/lib/types";

function CitationPill({ id }: { id: string }) {
  const selectCitation = useWorkbench((s) => s.selectCitation);
  return (
    <motion.button
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      onClick={() => selectCitation(id)}
      className="inline-block align-baseline mx-1 rounded-md px-1.5 py-px text-[10.5px] font-mono text-accent bg-accent-soft border border-accent/20 hover:border-accent/50 transition-colors cursor-pointer"
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
            <span className="inline-block w-[6px] h-[12px] bg-accent/70 align-[-2px] ml-px pulse-dot" />
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
        <Dialog.Overlay className="fixed inset-0 bg-ink/15 z-40" />
        <Dialog.Content asChild>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="panel fixed bottom-6 right-6 w-[min(92vw,380px)] z-50 p-5 rounded-2xl shadow-lg"
          >
            <Dialog.Title className="font-mono text-[12px] text-accent">
              {selectedCitation}
            </Dialog.Title>
            <Dialog.Description className="mt-3 text-[12.5px] leading-relaxed text-ink">
              {record ? (
                <>
                  <span className="block text-[13px] font-medium mb-2">
                    {record.triple}
                  </span>
                  <span className="block font-mono text-[10.5px] text-faint mb-3">
                    {record.source} · confidence {record.confidence.toFixed(2)}
                  </span>
                  <span className="block text-muted text-[12.5px]">
                    {record.abstract}
                  </span>
                </>
              ) : (
                <span className="font-mono text-[11.5px] text-muted">
                  Citation not present in the admitted context — suppressed or
                  filtered by the gate.
                </span>
              )}
            </Dialog.Description>
            <Dialog.Close
              className="mt-4 w-full text-[12px] font-medium py-1.5 rounded-lg border border-line text-muted hover:text-ink hover:border-line-strong transition"
              aria-label="Close"
            >
              Close
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
            "[!] Abstain triggered: retrieved evidence failed the dual gate (JEV below floor / entropy above τ). Distractors are suppressed — answering from clean parametric knowledge only. No citations attached.",
        },
      ]
    : (arena?.triage ?? []);

  const columns = [
    {
      title: "Baseline LLM",
      sub: "no retrieval",
      accent: "#a8a49b",
      segments: arena?.baseline ?? [],
      highlight: true,
      featured: false,
    },
    {
      title: "Vanilla RAG",
      sub: "unconditional top-K",
      accent: "#c08a2d",
      segments: arena?.vanillaRag ?? [],
      highlight: true,
      featured: false,
    },
    {
      title: "TRIAGE",
      sub: "gated + decoupled",
      accent: "#c15f3c",
      segments: triageSegments,
      highlight: false,
      featured: true,
    },
  ];

  return (
    <Panel>
      <div className="mb-4">
        <h2 className="font-display text-[17px] font-medium tracking-tight text-ink">
          Three-way clinical arena
        </h2>
        <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-faint mt-0.5">
          Same case · three systems
        </span>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {columns.map((col) => (
          <div
            key={col.title}
            className={`rounded-xl border p-4 border-t-2 ${
              col.featured
                ? "border-line-strong border-t-accent bg-surface"
                : "border-line bg-canvas"
            }`}
            style={{ borderTopColor: col.accent }}
          >
            <div className="flex items-baseline justify-between mb-2.5">
              <h3 className="text-[13px] font-medium text-ink">{col.title}</h3>
              <span className="font-mono text-[10px] text-faint">{col.sub}</span>
            </div>
            <div className="text-[12.5px] leading-[1.7] text-ink/90 max-h-[300px] overflow-y-auto pr-1.5 -mr-1">
              {arena ? (
                <AnswerStream
                  segments={col.segments}
                  streaming={col.featured && phase === "done"}
                  highlight={col.highlight}
                />
              ) : (
                <span className="font-mono text-[11.5px] text-faint">
                  awaiting triage run…
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
      {arena?.warnings.length ? (
        <div className="mt-3 flex flex-col gap-1.5">
          {arena.warnings.map((w) => (
            <span key={w} className="text-[11.5px] text-dismiss/80 flex gap-2">
              <span className="font-mono shrink-0">!</span>
              {w}
            </span>
          ))}
        </div>
      ) : null}
    </Panel>
  );
}
