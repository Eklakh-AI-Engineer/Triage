import { NextResponse } from "next/server";
import { BENCHMARK_MOCK } from "@/lib/cases";

// GET /api/triage/benchmark
// Base-paper evaluation telemetry: 5-dim judged scores, 8-criteria
// selection win-rate, McNemar significance.

const BASE = process.env.TRIAGE_API_BASE;

export async function GET() {
  if (BASE) {
    const upstream = await fetch(`${BASE}/api/triage/benchmark`);
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  return NextResponse.json(BENCHMARK_MOCK);
}
