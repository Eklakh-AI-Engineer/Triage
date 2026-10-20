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
import { Panel } from "./ui";

export function BenchmarkTelemetry() {
  const benchmark = useWorkbench((s) => s.benchmark);

  return (
    <Panel>
      <div className="mb-4">
        <h2 className="font-display text-[17px] font-medium tracking-tight text-ink">
          Validation telemetry
        </h2>
        <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-faint mt-0.5">
          base-paper benchmark · live
        </span>
      </div>
      {!benchmark ? (
        <div className="h-[300px] flex items-center justify-center text-[11.5px] font-mono text-faint">
          loading benchmark telemetry…
        </div>
      ) : (
        <div className="flex flex-col lg:flex-row gap-5">
          <div className="w-full lg:w-[46%] h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={benchmark.radar} outerRadius="72%">
                <PolarGrid stroke="#e9e6de" />
                <PolarAngleAxis
                  dataKey="dimension"
                  tick={{
                    fill: "#75726a",
                    fontSize: 9.5,
                    fontFamily: "var(--font-mono)",
                  }}
                />
                <PolarRadiusAxis
                  domain={[0, 100]}
                  tick={{ fill: "#a8a49b", fontSize: 8 }}
                  axisLine={false}
                />
                <Radar
                  name="Baseline LLM"
                  dataKey="baseline"
                  stroke="#a8a49b"
                  fill="#a8a49b"
                  fillOpacity={0.06}
                  strokeWidth={1}
                />
                <Radar
                  name="Vanilla RAG"
                  dataKey="vanilla"
                  stroke="#c08a2d"
                  fill="#c08a2d"
                  fillOpacity={0.08}
                  strokeWidth={1.2}
                />
                <Radar
                  name="TRIAGE"
                  dataKey="triage"
                  stroke="#c15f3c"
                  fill="#c15f3c"
                  fillOpacity={0.12}
                  strokeWidth={1.6}
                />
                <Tooltip
                  contentStyle={{
                    background: "#ffffff",
                    border: "1px solid #e9e6de",
                    borderRadius: 10,
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "#1f1e1d",
                    boxShadow: "0 4px 16px rgba(31,30,29,0.06)",
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex-1 min-w-0">
            <h3 className="text-[12.5px] font-medium text-ink mb-1">
              8-criteria win-rate vs vanilla RAG
            </h3>
            <div className="max-h-[186px] overflow-y-auto pr-1 -mr-1">
              {benchmark.winRate.map((w) => (
                <div key={w.criterion} className="my-2">
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-muted">{w.criterion}</span>
                    <span className="font-mono text-ink">{w.rate}%</span>
                  </div>
                  <div className="h-[4px] rounded-full bg-line overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${w.rate}%` }}
                      transition={{ duration: 0.7, ease: "easeOut" }}
                      className="h-full rounded-full"
                      style={{
                        background:
                          w.rate > 60 ? "#3e8e6b" : "#c08a2d",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-line">
              {benchmark.mcnemar.significant ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-admit/30 bg-admit/5 text-admit font-mono text-[11px]">
                  <span className="w-1.5 h-1.5 rounded-full bg-admit" />
                  McNemar p &lt; 0.001 · significant
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-line text-muted font-mono text-[11px]">
                  <span className="w-1.5 h-1.5 rounded-full bg-line-strong" />
                  McNemar not significant
                </span>
              )}
              <span className="inline-flex items-center px-2.5 py-1 rounded-md border border-line text-muted font-mono text-[11px]">
                n = {benchmark.nQueries}
              </span>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
