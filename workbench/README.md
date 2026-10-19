# TRIAGE Workbench

Production clinical intelligence dashboard for TRIAGE (Gated Evidence Selection
and Context Decoupling for Clinical RAG). Next.js 16 App Router, TypeScript,
Tailwind CSS v4, Radix UI, Framer Motion, Recharts, @xyflow/react.

## Run

```bash
cd workbench
npm install
npm run dev        # http://localhost:3000
# or production:
npm run build && npm run start
```

Runs fully standalone out of the box: all six workbench sections are driven by
a deterministic mock triage engine (`src/lib/triageEngine.ts`) and medically
grounded mock cases (`src/lib/cases.ts` — Metformin/CKD, Parkinson DBS,
SERPINA6/prednisolone).

## Wiring the Python backend

The UI and the built-in route handlers share one contract
(`src/lib/types.ts`):

| Endpoint                 | Method | Contract               |
| ------------------------ | ------ | ---------------------- |
| `/api/triage/query`      | POST   | `TriageQueryResponse`  |
| `/api/triage/subgraph`  | GET    | `SubgraphResponse`     |
| `/api/triage/benchmark` | GET    | `BenchmarkResponse`    |

Two wiring modes:

- **Client-direct (recommended):** set `NEXT_PUBLIC_TRIAGE_API_BASE` (e.g.
  `http://localhost:8000`). The client calls the Python backend directly.
- **Server-proxy:** set `TRIAGE_API_BASE` on the Next.js server. The built-in
  route handlers proxy the backend, keeping the Python origin private.

With neither set, every request resolves locally (mock engine + mock routes).

## Gate controls

- `tau_entropy` — abstention sensitivity; raises the admission bar and the JEV
  penalty. High tau + Stress mode demonstrates the ABSTAIN path (gate
  suppresses retrieved distractors; parametric fallback).
- `alpha` — soft Gram-Schmidt decoupling strength; cancels redundant clinical
  vectors into E/R/M sub-channels.
- Word budget, triage mode (Strict / Lenient / Stress — Stress injects
  adversarial PubMed distractors).

## Layout

```
src/
  app/
    api/triage/{query,subgraph,benchmark}/route.ts   # proxy-or-mock handlers
    layout.tsx / page.tsx / globals.css
  components/
    HudHeader.tsx          # telemetry HUD + presets
    QueryDeck.tsx          # intake + live SapBERT entity chips
    GateConfigDrawer.tsx   # tau/alpha/budget sliders + triage mode
    TriageRoom.tsx         # 3-stage evidence funnel + dual-gate verdict
    Arena.tsx              # baseline vs vanilla-RAG vs TRIAGE + citation flyout
    KnowledgeGraph.tsx     # 2-hop PrimeKG subgraph (trace score isolation)
    BenchmarkTelemetry.tsx # 5-dim radar + 8-criteria win-rate + McNemar
    ui.tsx / Workbench.tsx
  lib/
    types.ts     # API contracts (single source of truth)
    cases.ts     # mock clinical cases + benchmark telemetry
    triageEngine.ts  # deterministic standalone pipeline simulation
    api.ts       # client: live backend or mock
    store.ts     # zustand workbench state
```
