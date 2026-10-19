"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`glass-panel p-5 ${className}`}>{children}</div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-[13px] uppercase tracking-[0.14em] text-slate-500 mb-4 font-sans">
      {children}
    </h2>
  );
}

export function Badge({
  children,
  dotColor,
  className = "",
}: {
  children: ReactNode;
  dotColor?: string;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-slate-800/80 bg-slate-950/60 text-[11.5px] font-mono ${className}`}
    >
      {dotColor && (
        <span
          className="w-[7px] h-[7px] rounded-full pulse-dot"
          style={{ background: dotColor, boxShadow: `0 0 8px ${dotColor}` }}
        />
      )}
      {children}
    </span>
  );
}

export function Gauge({
  value,
  color = "#00f2fe",
  height = 8,
  className = "",
}: {
  value: number;
  color?: string;
  height?: number;
  className?: string;
}) {
  return (
    <div
      className={`w-full rounded-md bg-slate-800/60 overflow-hidden ${className}`}
      style={{ height }}
    >
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="h-full rounded-md"
        style={{
          background: `linear-gradient(90deg, #00f2fe, ${color})`,
          boxShadow: `0 0 12px ${color}66`,
        }}
      />
    </div>
  );
}

export const ENTITY_COLORS: Record<string, string> = {
  drug: "#60a5fa",
  disease: "#f87171",
  gene: "#c084fc",
  protein: "#a78bfa",
  phenotype: "#34d399",
  adverse_effect: "#fb923c",
};

export function EntityChip({
  name,
  type,
  confidence,
}: {
  name: string;
  type: string;
  confidence?: number;
}) {
  const color = ENTITY_COLORS[type] ?? "#94a3b8";
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      layout
      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono border"
      style={{
        color,
        borderColor: `${color}55`,
        background: `${color}14`,
      }}
    >
      {name}
      {confidence !== undefined && (
        <span className="text-slate-500">{confidence.toFixed(2)}</span>
      )}
    </motion.span>
  );
}
