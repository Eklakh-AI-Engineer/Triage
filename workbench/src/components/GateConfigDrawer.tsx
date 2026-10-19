"use client";

import * as Dialog from "@radix-ui/react-dialog";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { motion } from "framer-motion";
import { useWorkbench } from "@/lib/store";
import type { TriageMode } from "@/lib/types";

const MODES: { id: TriageMode; label: string; hint: string }[] = [
  { id: "strict", label: "Strict Gating (Recommended)", hint: "High threshold · aggressive abstention" },
  { id: "lenient", label: "Lenient Screening", hint: "Standard multi-hop retrieval" },
  { id: "stress", label: "Stress Test / Adversarial", hint: "Injects noisy PubMed distractors" },
];

function LabeledSlider({
  label,
  symbol,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  symbol: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-6">
      <div className="flex justify-between mb-2">
        <span className="text-[12px] text-slate-400 font-mono">
          {label} <span className="text-slate-600">({symbol})</span>
        </span>
        <span className="text-glow font-mono text-[12px]">{format(value)}</span>
      </div>
      <SliderPrimitive.Root
        className="relative flex items-center select-none touch-none w-full h-5"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0])}
        aria-label={label}
      >
        <SliderPrimitive.Track className="bg-slate-800/80 relative grow rounded-full h-[6px]">
          <SliderPrimitive.Range className="absolute bg-gradient-to-r from-glow to-admit rounded-full h-full" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className="block w-4 h-4 rounded-full bg-slate-100 shadow-[0_0_10px_rgba(0,242,254,0.6)] border-2 border-glow cursor-grab focus:outline-none"
          aria-label={label}
        />
      </SliderPrimitive.Root>
    </div>
  );
}

export function GateConfigDrawer({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const config = useWorkbench((s) => s.config);
  const setConfig = useWorkbench((s) => s.setConfig);
  const rerun = useWorkbench((s) => s.run);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40" />
        <Dialog.Content asChild>
          <motion.div
            initial={{ x: 380, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 380, opacity: 0 }}
            transition={{ type: "spring", damping: 26, stiffness: 240 }}
            className="glass-panel fixed top-4 right-4 bottom-4 w-[min(94vw,400px)] z-50 p-6 overflow-y-auto"
          >
            <Dialog.Title className="text-sm uppercase tracking-[0.14em] text-slate-400 font-sans">
              Triage Gate Configuration
            </Dialog.Title>
            <Dialog.Description className="text-[11.5px] text-slate-600 font-mono mt-1 mb-6">
              Dual-gate statistical filter · soft Gram-Schmidt decoupling
            </Dialog.Description>

            <LabeledSlider
              label="Abstention Sensitivity"
              symbol="τ entropy"
              value={config.tauEntropy}
              min={0.05}
              max={0.6}
              step={0.01}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setConfig({ tauEntropy: v })}
            />
            <LabeledSlider
              label="Decoupling Strength"
              symbol="α ortho"
              value={config.alphaDecoupling}
              min={0}
              max={1}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setConfig({ alphaDecoupling: v })}
            />
            <LabeledSlider
              label="Evidence Word Budget"
              symbol="B"
              value={config.wordBudget}
              min={250}
              max={1000}
              step={50}
              format={(v) => `${v} w`}
              onChange={(v) => setConfig({ wordBudget: v })}
            />

            <div className="text-[12px] text-slate-400 font-mono mb-2">
              Triage Mode
            </div>
            <div className="flex flex-col gap-2 mb-6">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setConfig({ mode: m.id })}
                  className={`text-left px-3 py-2.5 rounded-xl border text-[12px] font-mono transition ${
                    config.mode === m.id
                      ? "border-admit text-admit bg-admit/10"
                      : "border-slate-800/80 text-slate-500 bg-slate-950/50 hover:text-slate-300"
                  }`}
                >
                  [{m.label}]
                  <span className="block text-[10px] text-slate-600 mt-0.5">
                    {m.hint}
                  </span>
                </button>
              ))}
            </div>

            <button
              onClick={() => {
                onOpenChange(false);
                rerun();
              }}
              className="w-full font-mono text-[12.5px] px-4 py-2.5 rounded-xl border border-glow bg-glow/10 text-glow hover:bg-glow/20 hover:shadow-[0_0_18px_rgba(0,242,254,0.25)] transition"
            >
              APPLY &amp; RE-RUN TRIAGE
            </button>

            <Dialog.Close
              className="absolute top-5 right-5 text-slate-500 hover:text-slate-200 text-lg leading-none"
              aria-label="Close"
            >
              ×
            </Dialog.Close>
          </motion.div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
