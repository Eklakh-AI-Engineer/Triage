"""Result figures derived ONLY from the results JSON.

Mirrors the base paper's figure types: dimension bars (Fig.3/4), difficulty
tiers (Fig.5), criteria win-rates (Fig.6), latency-performance trade-off
(Fig.7a), and MedHallu difficulty F1 bars.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Must be set BEFORE importing matplotlib: the import itself validates
# MPLBACKEND, and inherited values like 'module://matplotlib_inline...' from
# notebook kernels crash on machines where that backend is unavailable.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

DIMS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability", "overall_score"]
COLORS = {"Baseline LLM": "#6c757d", "Vanilla RAG": "#0d6efd", "HCAD-RAG": "#198754"}

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def plot_dimension_scores(results: dict):
    """Grouped bars: 5 dimensions + Overall per system (base-paper Fig. 3/4)."""
    scores = results.get("scoring_based_assessment", {})
    if not scores:
        return None
    systems = list(scores)
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(systems)
    width = 0.8 / max(n, 1)
    for i, sys_name in enumerate(systems):
        xs = [d + (i - (n - 1) / 2) * width for d in range(len(DIMS))]
        ys = [scores[sys_name].get(d, 0) for d in DIMS]
        ax.bar(xs, ys, width=width, label=sys_name, color=COLORS.get(sys_name))
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([d.replace("_score", "").capitalize() for d in DIMS], rotation=20)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("Base-paper 5-dimension scoring by system")
    ax.legend()
    return _save(fig, "dimension_scores.png")

def plot_difficulty_tiers(results: dict):
    """Overall score vs 1/2/3-stage difficulty per system (base-paper Fig. 5)."""
    by_tier = results.get("scoring_by_difficulty_tier", {})
    if not by_tier:
        return None
    tiers = sorted(by_tier, key=int)
    systems = list(next(iter(by_tier.values()), {}))
    fig, ax = plt.subplots(figsize=(8, 5))
    for sys_name in systems:
        ys = [by_tier.get(t, {}).get(sys_name, 0.0) for t in tiers]
        ax.plot(range(len(tiers)), ys, marker="o", label=sys_name, color=COLORS.get(sys_name))
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t}-stage" for t in tiers])
    ax.set_ylabel("Overall score (0-100)")
    ax.set_title("Robustness across question difficulty")
    ax.legend()
    return _save(fig, "difficulty_tiers.png")

def plot_win_rates(results: dict):
    """Per-criteria selection win-rates (base-paper Fig. 6)."""
    sel = results.get("selection_based_assessment", {})
    if not sel:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    comparisons = list(sel)
    width = 0.8 / max(len(comparisons), 1)
    for i, comp in enumerate(comparisons):
        crit = sel[comp].get("criteria_win_rates_pct", {})
        xs = [j + (i - (len(comparisons) - 1) / 2) * width for j in range(len(crit))]
        ax.bar(xs, list(crit.values()), width=width, label=f"{comp} (HCAD win %)")
    if comparisons:
        crit_keys = list(sel[comparisons[0]].get("criteria_win_rates_pct", {}))
        ax.set_xticks(range(len(crit_keys)))
        ax.set_xticklabels([c.capitalize() for c in crit_keys], rotation=30)
    ax.axhline(50, color="red", ls="--", lw=1, label="50% parity")
    ax.set_ylabel("HCAD-RAG win rate (%)")
    ax.set_title("Selection-based assessment (8 criteria)")
    ax.legend(fontsize=8)
    return _save(fig, "win_rates.png")

def plot_latency_performance(results: dict):
    """Latency vs quality scatter (base-paper Fig. 7a)."""
    lat = results.get("retrieval_latency_seconds", {})
    scores = results.get("scoring_based_assessment", {})
    if not lat:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for sys_name, seconds in lat.items():
        y = scores.get(sys_name, {}).get("overall_score", 0)
        ax.scatter(seconds, y, s=120, color=COLORS.get(sys_name))
        ax.annotate(f"{sys_name}\n({seconds:.2f}s, {y:.1f})", (seconds, y),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Median retrieval latency (s)")
    ax.set_ylabel("Overall score (0-100)")
    ax.set_title("Speed vs quality trade-off")
    return _save(fig, "latency_performance.png")

def plot_hallucination_tiers(hallu_path: Path | None = None):
    """MedHallu F1 by difficulty tier (the claim the base paper never tests)."""
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
                        plot_hallucination_tiers(hallu_path)] if p]
