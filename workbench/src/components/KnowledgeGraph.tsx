"use client";

import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Node,
} from "@xyflow/react";
import { motion } from "framer-motion";
import { useEffect, useMemo } from "react";
import { useWorkbench } from "@/lib/store";
import { ENTITY_COLORS, Panel } from "./ui";
import type { GraphNode, SubgraphResponse } from "@/lib/types";

const TYPE_LABEL: Record<string, string> = {
  drug: "Drug",
  disease: "Disease",
  gene: "Gene",
  protein: "Target protein",
  phenotype: "Phenotype",
  adverse_effect: "Adverse effect",
};

function ClinicalNode({ data }: { data: Record<string, unknown> }) {
  const label = data.label as string;
  const type = data.type as string;
  const selected = data.selected as boolean;
  const dimmed = data.dimmed as boolean;
  const color = ENTITY_COLORS[type] ?? "#75726a";
  return (
    <div
      className={`px-2.5 py-1.5 rounded-lg border font-mono text-[10px] flex items-center gap-2 transition-all duration-300 ${
        dimmed ? "opacity-25" : "opacity-100"
      }`}
      style={{
        borderColor: selected ? color : "#e9e6de",
        background: selected ? `${color}12` : "#ffffff",
        color: "#1f1e1d",
      }}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <span
        className="w-1.5 h-1.5 rounded-full shrink-0"
        style={{ background: color }}
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
      const color = incident ? "#c15f3c" : "#c9c5bb";
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.relation,
        animated: incident,
        style: {
          stroke: color,
          strokeWidth: incident ? 1.8 : 1,
          opacity: selectedNode !== null && !incident ? 0.25 : 1,
        },
        labelStyle: { fill: incident ? "#c15f3c" : "#75726a" },
        labelBgStyle: { fill: "#faf9f5" },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
      };
    });
  }, [subgraph, selectedNode]);

  const selectedData: GraphNode | undefined = subgraph?.nodes.find(
    (n) => n.id === selectedNode
  );
  const incidentEdges = subgraph?.edges.filter(
    (e) => e.source === selectedNode || e.target === selectedNode
  );
  const totalTrace = subgraph?.nodes.reduce((a, n) => a + n.trace, 0) ?? 1;

  return (
    <Panel>
      <div className="mb-4">
        <h2 className="font-display text-[17px] font-medium tracking-tight text-ink">
          Knowledge subgraph
        </h2>
        <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-faint mt-0.5">
          2-hop reasoning neighborhood · PrimeKG
        </span>
      </div>
      <div className="h-[320px] rounded-xl border border-line bg-canvas overflow-hidden">
        {subgraph ? (
          <ReactFlow
            nodes={viewNodes}
            edges={viewEdges}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) =>
              selectNode(selectedNode === node.id ? null : node.id)
            }
            fitView
            minZoom={0.4}
            maxZoom={1.8}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={24}
              size={1}
              color="#d8d4ca"
            />
            <Controls
              showInteractive={false}
              className="!border-line [&>button]:!border-line [&>button]:!bg-surface [&>button]:!fill-muted"
            />
          </ReactFlow>
        ) : (
          <div className="h-full flex items-center justify-center text-[11.5px] font-mono text-faint">
            {phase === "idle"
              ? "subgraph materializes after the first triage run"
              : "expanding 2-hop neighborhood…"}
          </div>
        )}
      </div>

      <motion.div
        initial={false}
        animate={{
          opacity: selectedData ? 1 : 0.6,
          borderColor: selectedData
            ? `${ENTITY_COLORS[selectedData.type] ?? "#d8d4ca"}66`
            : "#e9e6de",
        }}
        className="mt-3 rounded-xl border bg-canvas p-3.5 font-mono text-[11px]"
      >
        {selectedData ? (
          <>
            <div className="flex items-center justify-between">
              <span className="text-ink">
                {selectedData.label}{" "}
                <span style={{ color: ENTITY_COLORS[selectedData.type] }}>
                  ({TYPE_LABEL[selectedData.type] ?? selectedData.type})
                </span>
              </span>
              <span className="text-accent">
                trace {selectedData.trace.toFixed(2)} ·{" "}
                {((selectedData.trace / totalTrace) * 100).toFixed(1)}% of context
              </span>
            </div>
            <div className="mt-1.5 text-faint leading-relaxed">
              {incidentEdges && incidentEdges.length > 0
                ? incidentEdges
                    .map((e) => `${e.relation} (${e.weight.toFixed(2)})`)
                    .join(" · ")
                : "no incident relations in the 2-hop cut"}
            </div>
          </>
        ) : (
          <span className="text-muted">
            select a node to isolate its reasoning path and trace contribution
          </span>
        )}
      </motion.div>
    </Panel>
  );
}
