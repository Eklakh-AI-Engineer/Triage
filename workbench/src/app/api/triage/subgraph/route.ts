import { NextRequest, NextResponse } from "next/server";
import { resolveCase } from "@/lib/cases";

// GET /api/triage/subgraph?query=...
// Returns the localized 2-hop clinical reasoning subgraph.

const BASE = process.env.TRIAGE_API_BASE;

export async function GET(req: NextRequest) {
  const query = req.nextUrl.searchParams.get("query") ?? "";

  if (BASE) {
    const upstream = await fetch(
      `${BASE}/api/triage/subgraph?query=${encodeURIComponent(query)}`
    );
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  const clinicalCase = resolveCase(query);
  return NextResponse.json(clinicalCase.subgraph);
}
