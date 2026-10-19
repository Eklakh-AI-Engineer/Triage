import { NextRequest, NextResponse } from "next/server";
import { resolveCase } from "@/lib/cases";
import { simulateTriage } from "@/lib/triageEngine";
import type { TriageQueryRequest } from "@/lib/types";

// POST /api/triage/query
// Standalone: deterministic mock simulation of the TRIAGE pipeline.
// With TRIAGE_API_BASE set: proxies the Python backend (same contract).

const BASE = process.env.TRIAGE_API_BASE;

export async function POST(req: NextRequest) {
  const body = (await req.json()) as TriageQueryRequest;

  if (BASE) {
    const upstream = await fetch(`${BASE}/api/triage/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  const clinicalCase = resolveCase(body.query);
  const result = simulateTriage(body.query, body.config, clinicalCase);
  return NextResponse.json(result);
}
