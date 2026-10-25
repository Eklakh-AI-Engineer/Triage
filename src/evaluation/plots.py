"""Result figures derived ONLY from the results JSON - TRI-ARM PROTOCOL.

Every figure compares all three systems side-by-side:
  1. Baseline LLM   (no retrieval; parametric knowledge)
  2. Vanilla RAG    (standard dense top-K)
  3. HCAD-RAG       (proposed architecture)
No chart, panel, or metric may reduce the comparison to two arms. Each
renderer validates that all three systems are present before drawing, and
each series carries a fixed visual identity (color, marker, line style)
so figures stay comparable across runs and report sections.

Figure types mirror the base paper: dimension bars (Fig.3/4), difficulty
tiers (Fig.5), criteria win-rate panels (Fig.6), latency-performance
trade-off (Fig.7a), MedHallu difficulty F1 bars, plus the tri-arm
citation-grounding comparison.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Must be set BEFORE importing matplotlib: the import itself validates
# MPLBACKEND. Notebook kernels export the inline backend here, which crashes
# script-mode runs, so force Agg rather than setdefault.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

DIMS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability", "overall_score"]

# Tri-arm visual identity: muted grey baseline, classic blue vanilla,
# emerald green HCAD - with distinct markers and line styles for the
# trajectory curves (o / s / ^, solid / dashed / dash-dot).
SYSTEM_ORDER = ["Baseline LLM", "Vanilla RAG", "HCAD-RAG"]
SYSTEM_STYLE = {
    "Baseline LLM": {"color": "#6c757d", "marker": "o", "linestyle": "--"},
    "Vanilla RAG": {"color": "#0d6efd", "marker": "s", "linestyle": "-."},
    "HCAD-RAG": {"color": "#198754", "marker": "^", "linestyle": "-"},
}
# Selection panels, fixed order: A vs baseline, B vs vanilla, C the
# baseline-vs-vanilla delta. Older results files may lack Panel C.
SELECTION_PANELS = ["HCAD_vs_Baseline", "HCAD_vs_Vanilla", "Vanilla_vs_Baseline"]
PANEL_TITLES = {
    "HCAD_vs_Baseline": "Panel A: HCAD-RAG vs Baseline LLM",
    "HCAD_vs_Vanilla": "Panel B: HCAD-RAG vs Vanilla RAG",
    "Vanilla_vs_Baseline": "Panel C: Vanilla RAG vs Baseline LLM",
}
PANEL_DEFAULT_TARGET = {
    "HCAD_vs_Baseline": "HCAD-RAG",
    "HCAD_vs_Vanilla": "HCAD-RAG",
    "Vanilla_vs_Baseline": "Vanilla RAG",
}


def _require_three_systems(scores: dict, context: str) -> list[str]:
    """Tri-arm gate: refuse to render any 2-way comparison (protocol rule)."""
    present = [s for s in SYSTEM_ORDER if s in scores]
    missing = [s for s in SYSTEM_ORDER if s not in scores]
    if missing:
        raise ValueError(
            f"Tri-arm protocol violated in {context}: missing systems {missing}. "
            "Every figure must compare Baseline LLM, Vanilla RAG and HCAD-RAG "
            "side-by-side; refusing to draw a 2-way comparison."
        )
    return present


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_dimension_scores(results: dict):
    """Grouped bars: 5 dimensions + Overall per system (base-paper Fig. 3/4).

    Three grouped bars per dimension (grey/blue/green), each with its
    numeric value labelled on top.
    """
    scores = results.get("scoring_based_assessment", {})
    if not scores:
        return None
    _require_three_systems(scores, "dimension scores")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(SYSTEM_ORDER)
    width = 0.8 / n
    for i, sys_name in enumerate(SYSTEM_ORDER):
        xs = [d + (i - (n - 1) / 2) * width for d in range(len(DIMS))]
        ys = [scores[sys_name].get(d, 0) for d in DIMS]
        bars = ax.bar(xs, ys, width=width, label=sys_name,
                      color=SYSTEM_STYLE[sys_name]["color"])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([d.replace("_score", "").capitalize() for d in DIMS], rotation=20)
    ax.set_ylabel("Score (0-100)")
    ax.set_ylim(0, 100)
    ax.set_title("Base-paper 5-dimension scoring - three-way comparison")
    ax.legend()
    return _save(fig, "dimension_scores.png")


def plot_difficulty_tiers(results: dict):
    """Overall score vs 1/2/3-stage difficulty per system (base-paper Fig. 5).

    Three distinct trajectories (markers o/s/^, line styles --/-./-) so the
    degradation slopes can be contrasted directly.
    """
    by_tier = results.get("scoring_by_difficulty_tier", {})
    if not by_tier:
        return None
    _require_three_systems(next(iter(by_tier.values()), {}), "difficulty tiers")
    tiers = sorted(by_tier, key=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sys_name in SYSTEM_ORDER:
        style = SYSTEM_STYLE[sys_name]
        ys = [by_tier.get(t, {}).get(sys_name, 0.0) for t in tiers]
        ax.plot(range(len(tiers)), ys, marker=style["marker"],
                linestyle=style["linestyle"], color=style["color"],
                label=sys_name, linewidth=2, markersize=7)
        for x, y in zip(range(len(tiers)), ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), fontsize=7, ha="center",
                        color=style["color"])
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t}-stage" for t in tiers])
    ax.set_ylabel("Overall score (0-100)")
    ax.set_title("Robustness across question difficulty - three-way comparison")
    ax.legend()
    return _save(fig, "difficulty_tiers.png")


def plot_win_rates(results: dict):
    """Per-criteria selection win-rate panels (base-paper Fig. 6, tri-arm).

    Renders the available panels in fixed order - A: HCAD vs Baseline,
    B: HCAD vs Vanilla, C: Vanilla vs Baseline (present in results files
    produced after the tri-arm evaluator change). The baseline arm is
    never omitted: it appears in Panels A and C.
    """
    sel = results.get("selection_based_assessment", {})
    panels = [p for p in SELECTION_PANELS if sel.get(p)]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5),
                             squeeze=False)
    for ax, comp in zip(axes[0], panels):
        data = sel[comp]
        target = data.get("target_system", PANEL_DEFAULT_TARGET.get(comp, "HCAD-RAG"))
        crit = data.get("criteria_win_rates_pct", {})
        crit_keys = list(crit)
        bars = ax.bar(range(len(crit_keys)), list(crit.values()),
                      color=SYSTEM_STYLE.get(target, SYSTEM_STYLE["HCAD-RAG"])["color"])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        ax.set_xticks(range(len(crit_keys)))
        ax.set_xticklabels([c.capitalize() for c in crit_keys], rotation=30, fontsize=8)
        ax.axhline(50, color="red", ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"{target} win rate (%)")
        ax.set_title(f"{PANEL_TITLES[comp]}\n({target} win %, 50% parity dashed)",
                     fontsize=10)
    fig.suptitle("Selection-based assessment (8 clinical criteria) - tri-arm panels",
                 y=1.02)
    return _save(fig, "win_rates.png")


def plot_latency_performance(results: dict):
    """Latency vs quality scatter (base-paper Fig. 7a) - all three points.

    The no-retrieval baseline sits at 0.0 s retrieval latency by
    definition; it is injected when absent so the trade-off always shows
    all three arms, each annotated with (latency, score).
    """
    scores = results.get("scoring_based_assessment", {})
    lat = dict(results.get("retrieval_latency_seconds", {}))
    if not scores:
        return None
    _require_three_systems(scores, "latency trade-off")
    if "Baseline LLM" in scores:
        lat.setdefault("Baseline LLM", 0.0)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for sys_name in SYSTEM_ORDER:
        if sys_name not in lat:
            continue
        seconds = lat[sys_name]
        y = scores.get(sys_name, {}).get("overall_score", 0)
        style = SYSTEM_STYLE[sys_name]
        ax.scatter(seconds, y, s=140, color=style["color"], marker=style["marker"],
                   zorder=3, label=sys_name)
        ax.annotate(f"{sys_name}\n({seconds:.2f}s, {y:.1f})", (seconds, y),
                    textcoords="offset points", xytext=(10, -6), fontsize=8)
    ax.set_xlabel("Mean retrieval latency (s)")
    ax.set_ylabel("Overall score (0-100)")
    ax.set_title("Speed vs quality trade-off - three-way comparison")
    ax.legend(fontsize=8)
    return _save(fig, "latency_performance.png")


def plot_citation_grounding(results: dict):
    """Citation & evidence grounding - all three arms (tri-arm protocol).

    The Baseline LLM is the zero-citation ungrounded reference (0%
    coverage, 0 citations); Vanilla and HCAD show what retrieval
    introduces. Three panels: citation coverage, total citations, and
    mean citation precision.
    """
    citations = results.get("citation_precision", {})
    scores = results.get("scoring_based_assessment", {})
    if not citations:
        return None
    _require_three_systems(scores, "citation grounding")
    if "Baseline LLM" not in citations:
        n_baseline = citations.get("Vanilla RAG", {}).get("n_answers", 0)
        citations = dict(citations)
        citations["Baseline LLM"] = {
            "n_answers_with_citations": 0, "n_answers": n_baseline,
            "total_citations": 0, "mean_precision": None,
        }
    coverage, totals, precisions = [], [], []
    for sys_name in SYSTEM_ORDER:
        c = citations.get(sys_name, {})
        n_ans = c.get("n_answers", 0) or 1
        coverage.append((c.get("n_answers_with_citations", 0) / n_ans) * 100.0)
        totals.append(c.get("total_citations", 0))
        mp = c.get("mean_precision")
        precisions.append(mp * 100.0 if mp is not None else 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    specs = [("Citation coverage (% of answers)", coverage, 100),
             ("Total [Evidence i] citations", totals, None),
             ("Mean citation precision (%)", precisions, 100)]
    for ax, (title, ys, ymax) in zip(axes, specs):
        bars = ax.bar(SYSTEM_ORDER, ys,
                      color=[SYSTEM_STYLE[s]["color"] for s in SYSTEM_ORDER])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_title(title, fontsize=10)
        if ymax:
            ax.set_ylim(0, ymax)
    fig.suptitle(
        "Citation & evidence grounding - tri-arm comparison "
        "(Baseline LLM is the zero-citation ungrounded reference)", y=1.04)
    return _save(fig, "citation_grounding.png")


def plot_hallucination_tiers(hallu_path: Path | None = None):
    """MedHallu F1 by difficulty tier (the claim the base paper never tests).

    Single-system by design (a detection-ability claim, not a tri-arm
    generation comparison)."""
    if hallu_path is None or not Path(hallu_path).exists():
        return None
    data = json.loads(Path(hallu_path).read_text())
    tiers = ["easy", "medium", "hard"]
    ys = [data.get(t, {}).get("f1", 0) for t in tiers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tiers, ys, color="#198754")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(f"MedHallu detection F1 ({data.get('system', 'system')})")
    return _save(fig, "hallucination_tiers.png")


def plot_all(results: dict, hallu_path: Path | None = None) -> list:
    return [p for p in [plot_dimension_scores(results), plot_difficulty_tiers(results),
                        plot_win_rates(results), plot_latency_performance(results),
                        plot_citation_grounding(results),
                        plot_hallucination_tiers(hallu_path)] if p]
