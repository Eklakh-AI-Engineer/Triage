"use client";

import { motion } from "framer-motion";
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { useWorkbench } from "@/lib/store";
import { Badge, Panel, SectionTitle } from "./ui";

export function BenchmarkTelemetry() {
  const benchmark = useWorkbench((s) => s.benchmark);

  return (
    <Panel>
      <SectionTitle>Live Clinical Validation Telemetry</SectionTitle>
      {!benchmark ? (
        <div className="h-[300px] flex items-center justify-center text-[11.5px] font-mono text-slate-600">
          loading benchmark telemetry…
        </div>
      ) : (
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="w-full lg:w-[46%] h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={benchmark.radar} outerRadius="72%">
                <PolarGrid stroke="rgba(30,41,59,0.9)" />
                <PolarAngleAxis
                  dataKey="dimension"
                  tick={{ fill: "#64748b", fontSize: 9, fontFamily: "var(--font-jetbrains)" }}
                />
                <PolarRadiusAxis
                  domain={[0, 100]}
                  tick={{ fill: "#475569", fontSize: 8 }}
                  axisLine={false}
                />
                <Radar
                  name="Baseline LLM"
                  dataKey="baseline"
                  stroke="#64748b"
                  fill="#64748b"
                  fillOpacity={0.08}
                  strokeWidth={1}
                />
                <Radar
                  name="Vanilla RAG"
                  dataKey="vanilla"
                  stroke="#f59e0b"
                  fill="#f59e0b"
                  fillOpacity={0.1}
                  strokeWidth={1.2}
                />
                <Radar
                  name="TRIAGE"
                  dataKey="triage"
                  stroke="#00f2fe"
                  fill="#00f2fe"
                  fillOpacity={0.16}
                  strokeWidth={1.8}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(2,6,23,0.92)",
                    border: "1px solid rgba(0,242,254,0.4)",
                    borderRadius: 12,
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: 11,
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex-1 min-w-0">
            <h3 className="text-[11px] uppercase tracking-[0.12em] text-slate-500 mb-2">
              8-Criteria Win-Rate vs Vanilla RAG
            </h3>
            <div className="max-h-[190px] overflow-y-auto pr-1">
              {benchmark.winRate.map((w) => (
                <div key={w.criterion} className="my-1.5">
                  <div className="flex justify-between text-[10.5px] font-mono">
                    <span className="text-slate-400">{w.criterion}</span>
                    <span className="text-slate-500">{w.rate}%</span>
                  </div>
                  <div className="h-[7px] rounded-md bg-slate-800/60 overflow-hidden mt-0.5">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${w.rate}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className="h-full rounded-md"
                      style={{
                        background:
                          w.rate > 60
                            ? "linear-gradient(90deg,#00f2fe,#10b981)"
                            : "linear-gradient(90deg,#00f2fe,#f59e0b)",
                        boxShadow: "0 0 10px rgba(0,242,254,0.25)",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-2 mt-3">
              <Badge
                dotColor={benchmark.mcnemar.significant ? "#10b981" : "#f59e0b"}
                className={
                  benchmark.mcnemar.significant
                    ? "border-admit/60 text-admit"
                    : "border-observe/60 text-observe"
                }
              >
                McNemar χ²={benchmark.mcnemar.statistic} · p &lt;{" "}
                {benchmark.mcnemar.pValue < 0.001
                  ? "0.001"
                  : benchmark.mcnemar.pValue.toFixed(3)}{" "}
                · {benchmark.mcnemar.significant ? "SIGNIFICANT" : "NS"}
              </Badge>
              <Badge>n = {benchmark.nQueries} queries · 5-dim judged</Badge>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
