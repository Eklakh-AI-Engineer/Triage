"use client";

import * as Dialog from "@radix-ui/react-dialog";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { motion } from "framer-motion";
import { useWorkbench } from "@/lib/store";
import type { TriageMode } from "@/lib/types";

const MODES: { id: TriageMode; label: string; hint: string }[] = [
  { id: "strict", label: "Strict gating", hint: "Recommended — high threshold, aggressive abstention" },
  { id: "lenient", label: "Lenient screening", hint: "Standard multi-hop retrieval" },
  { id: "stress", label: "Stress / adversarial", hint: "Injects noisy PubMed distractors" },
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
    <div className="mb-7">
      <div className="flex justify-between items-baseline mb-3">
        <span className="text-[12.5px] text-ink">
          {label}{" "}
          <span className="font-mono text-[10.5px] text-faint">{symbol}</span>
        </span>
        <span className="font-mono text-[12.5px] text-ink">{format(value)}</span>
      </div>
      <SliderPrimitive.Root
        className="relative flex items-center select-none touch-none w-full h-4"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0])}
        aria-label={label}
      >
        <SliderPrimitive.Track className="bg-line relative grow rounded-full h-[3px]">
          <SliderPrimitive.Range className="absolute bg-ink rounded-full h-full" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className="block w-3.5 h-3.5 rounded-full bg-surface border-[1.5px] border-ink cursor-grab focus:outline-none transition-transform hover:scale-110"
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
        <Dialog.Overlay className="fixed inset-0 bg-ink/15 z-40" />
        <Dialog.Content asChild>
          <motion.div
            initial={{ x: 60, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 60, opacity: 0 }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="panel fixed top-4 right-4 bottom-4 w-[min(94vw,400px)] z-50 p-7 overflow-y-auto rounded-2xl"
          >
            <Dialog.Title className="font-display text-[18px] font-medium tracking-tight text-ink">
              Gate configuration
            </Dialog.Title>
            <Dialog.Description className="text-[12px] text-muted mt-1 mb-8">
              Dual-gate statistical filter · soft Gram-Schmidt decoupling
            </Dialog.Description>

            <LabeledSlider
              label="Abstention sensitivity"
              symbol="τ entropy"
              value={config.tauEntropy}
              min={0.05}
              max={0.6}
              step={0.01}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setConfig({ tauEntropy: v })}
            />
            <LabeledSlider
              label="Decoupling strength"
              symbol="α ortho"
              value={config.alphaDecoupling}
              min={0}
              max={1}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setConfig({ alphaDecoupling: v })}
            />
            <LabeledSlider
              label="Evidence word budget"
              symbol="B"
              value={config.wordBudget}
              min={250}
              max={1000}
              step={50}
              format={(v) => `${v} words`}
              onChange={(v) => setConfig({ wordBudget: v })}
            />

            <div className="text-[12.5px] text-ink mb-3">Triage mode</div>
            <div className="flex flex-col gap-2 mb-8">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setConfig({ mode: m.id })}
                  className={`text-left px-3.5 py-2.5 rounded-xl border text-[12.5px] transition ${
                    config.mode === m.id
                      ? "border-ink bg-canvas text-ink"
                      : "border-line text-muted hover:border-line-strong hover:text-ink"
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        config.mode === m.id ? "bg-ink" : "bg-line-strong"
                      }`}
                    />
                    {m.label}
                  </span>
                  <span className="block text-[11px] text-faint mt-1 ml-3.5">
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
              className="w-full text-[13px] font-medium py-2.5 rounded-lg bg-ink text-canvas hover:opacity-85 transition"
            >
              Apply &amp; re-run
            </button>

            <Dialog.Close
              className="absolute top-6 right-6 text-faint hover:text-ink transition text-lg leading-none"
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
