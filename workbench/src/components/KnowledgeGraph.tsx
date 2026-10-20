"use client";

import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import { motion } from "framer-motion";
import { useEffect, useMemo } from "react";
import { useWorkbench } from "@/lib/store";
import { ENTITY_COLORS, Panel, SectionTitle } from "./ui";
import type { GraphNode, SubgraphResponse } from "@/lib/types";

const TYPE_LABEL: Record<string, string> = {
  drug: "Drug",
  disease: "Disease",
  gene: "Gene",
  protein: "Target Protein",
  phenotype: "Phenotype",
  adverse_effect: "Adverse Effect",
};

function ClinicalNode({ data }: { data: Record<string, unknown> }) {
  const label = data.label as string;
  const type = data.type as string;
  const selected = data.selected as boolean;
  const dimmed = data.dimmed as boolean;
  const color = ENTITY_COLORS[type] ?? "#94a3b8";
  return (
    <div
      className={`px-2.5 py-1.5 rounded-lg border font-mono text-[10px] flex items-center gap-2 transition-all duration-300 ${
        dimmed ? "opacity-30" : "opacity-100"
      }`}
      style={{
        borderColor: selected ? color : `${color}55`,
        background: selected ? `${color}26` : "rgba(2,6,23,0.85)",
        boxShadow: selected ? `0 0 16px ${color}66` : "none",
        color: "#cbd5e1",
      }}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <span
        className="w-2 h-2 rounded-full shrink-0"
        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
      />
      {label}
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}

const nodeTypes = { clinical: ClinicalNode };

function layout(subgraph: SubgraphResponse): Node[] {
  const center = subgraph.centerId;
  const others = subgraph.nodes.filter((n) => n.id !== center);
  const radius = 170;
  return subgraph.nodes.map((n) => {
    if (n.id === center) {
      return {
        id: n.id,
        type: "clinical",
        position: { x: 0, y: 0 },
        data: { label: n.label, type: n.type },
      };
    }
    const idx = others.findIndex((o) => o.id === n.id);
    const angle = (idx / others.length) * Math.PI * 2;
    return {
      id: n.id,
      type: "clinical",
      position: { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius * 0.72 },
      data: { label: n.label, type: n.type },
    };
  });
}

export function KnowledgeGraph() {
  const subgraph = useWorkbench((s) => s.subgraph);
  const selectedNode = useWorkbench((s) => s.selectedNode);
  const selectNode = useWorkbench((s) => s.selectNode);
  const phase = useWorkbench((s) => s.phase);

  const [nodes, , onNodesChange] = useNodesState<Node>([]);
  const [edges, , onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    if (!subgraph) return;
    selectNode(null);
  }, [subgraph, selectNode]);

  const viewNodes = useMemo(() => {
    if (!subgraph) return [];
    return layout(subgraph).map((n) => ({
      ...n,
      data: {
        ...n.data,
        selected: selectedNode === n.id,
        dimmed:
          selectedNode !== null &&
          selectedNode !== n.id &&
          !subgraph.edges.some(
            (e) =>
              (e.source === selectedNode && e.target === n.id) ||
              (e.target === selectedNode && e.source === n.id)
          ),
      },
    }));
  }, [subgraph, selectedNode]);

  const viewEdges = useMemo(() => {
    if (!subgraph) return [];
    return subgraph.edges.map((e) => {
      const incident =
        selectedNode !== null &&
        (e.source === selectedNode || e.target === selectedNode);
      const color = incident ? "#00f2fe" : "rgba(100,116,139,0.55)";
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.relation,
        animated: incident,
        style: {
          stroke: color,
          strokeWidth: incident ? 2.2 : 1.2,
          opacity:
            selectedNode !== null && !incident ? 0.25 : 1,
        },
        labelStyle: { fill: incident ? "#00f2fe" : "#94a3b8" },
        labelBgStyle: { fill: "#0b0f19" },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
      };
    });
  }, [subgraph, selectedNode]);

  const selectedData: GraphNode | undefined = subgraph?.nodes.find(
    (n) => n.id === selectedNode
  );
  const incidentEdges = subgraph?.edges.filter(
    (e) => e.source === selectedNode || e.target === selectedNode
  );
  const totalTrace =
    subgraph?.nodes.reduce((a, n) => a + n.trace, 0) ?? 1;

  return (
    <Panel>
      <SectionTitle>
        Knowledge Subgraph — 2-hop localized reasoning (PrimeKG)
      </SectionTitle>
      <div className="h-[320px] rounded-xl border border-slate-800/60 bg-slate-950/40 overflow-hidden">
        {subgraph ? (
          <ReactFlow
            nodes={viewNodes.length ? viewNodes : nodes}
            edges={viewEdges.length ? viewEdges : edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) =>
              selectNode(selectedNode === node.id ? null : node.id)
            }
            fitView
            minZoom={0.4}
            maxZoom={1.8}
          >
            <Background gap={22} color="rgba(30,41,59,0.5)" />
            <Controls
              showInteractive={false}
              className="!bg-slate-900/80 !border-slate-700 [&>button]:!bg-slate-900/80 [&>button]:!border-slate-700 [&>button]:!fill-slate-300"
            />
          </ReactFlow>
        ) : (
          <div className="h-full flex items-center justify-center text-[11.5px] font-mono text-slate-600">
            {phase === "idle"
              ? "subgraph materializes after first triage run…"
              : "expanding 2-hop neighborhood…"}
          </div>
        )}
      </div>

      <AnimateTracePanel
        selected={selectedData}
        incident={incidentEdges ?? []}
        contribution={
          selectedData ? selectedData.trace / totalTrace : 0
        }
      />
    </Panel>
  );
}

function AnimateTracePanel({
  selected,
  incident,
  contribution,
}: {
  selected?: GraphNode;
  incident: { relation: string; weight: number }[];
  contribution: number;
}) {
  return (
    <motion.div
      initial={false}
      animate={{
        opacity: selected ? 1 : 0.55,
        borderColor: selected
          ? `${ENTITY_COLORS[selected.type] ?? "#475569"}88`
          : "rgba(30,41,59,0.8)",
      }}
      className="mt-3 rounded-xl border bg-slate-950/40 p-3 font-mono text-[11px]"
    >
      {selected ? (
        <>
          <div className="flex items-center justify-between">
            <span className="text-slate-200">
              {selected.label}{" "}
              <span style={{ color: ENTITY_COLORS[selected.type] }}>
                ({TYPE_LABEL[selected.type] ?? selected.type})
              </span>
            </span>
            <span className="text-glow">
              trace σ·w = {selected.trace.toFixed(2)} ·{" "}
              {(contribution * 100).toFixed(1)}% of context
            </span>
          </div>
          <div className="mt-1.5 text-slate-500 leading-relaxed">
            incident relations:{" "}
            {incident.length > 0
              ? incident
                  .map((e) => `${e.relation} (${e.weight.toFixed(2)})`)
                  .join(" · ")
              : "none — isolated in the 2-hop cut"}
          </div>
        </>
      ) : (
        <span className="text-slate-500">
          select a node to isolate its reasoning path and trace score
          contribution
        </span>
      )}
    </motion.div>
  );
}
