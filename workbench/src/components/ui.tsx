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
  return <div className={`panel p-6 ${className}`}>{children}</div>;
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <div className="mb-4">
      <h2 className="font-display text-[17px] font-medium tracking-tight text-ink">
        {children}
      </h2>
    </div>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-faint mt-0.5">
      {children}
    </span>
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
      className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-md border border-line bg-surface text-[11px] font-mono text-muted ${className}`}
    >
      {dotColor && (
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: dotColor }}
        />
      )}
      {children}
    </span>
  );
}

export function Gauge({
  value,
  color = "#c15f3c",
  height = 6,
  className = "",
}: {
  value: number;
  color?: string;
  height?: number;
  className?: string;
}) {
  return (
    <div
      className={`w-full rounded-full bg-line overflow-hidden ${className}`}
      style={{ height }}
    >
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="h-full rounded-full"
        style={{ background: color }}
      />
    </div>
  );
}

export const ENTITY_COLORS: Record<string, string> = {
  drug: "#3d6fa8",
  disease: "#b4543f",
  gene: "#7b5cb8",
  protein: "#4a8a8a",
  phenotype: "#3e8e6b",
  adverse_effect: "#c08a2d",
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
  const color = ENTITY_COLORS[type] ?? "#75726a";
  return (
    <motion.span
      initial={{ opacity: 0, y: 2 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      layout
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-mono border"
      style={{
        color,
        borderColor: `${color}30`,
        background: `${color}0a`,
      }}
    >
      {name}
      {confidence !== undefined && (
        <span className="text-faint">{confidence.toFixed(2)}</span>
      )}
    </motion.span>
  );
}
