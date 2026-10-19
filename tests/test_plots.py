import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")

def test_plots_render_from_results_json(tmp_path, monkeypatch):
    from src.evaluation import plots
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    results = {
        "scoring_based_assessment": {
            "Baseline LLM": {d: 70 for d in plots.DIMS},
            "Vanilla RAG": {d: 75 for d in plots.DIMS},
            "HCAD-RAG": {d: 82 for d in plots.DIMS},
        },
        "scoring_by_difficulty_tier": {
            "1": {"Baseline LLM": 72, "Vanilla RAG": 76, "HCAD-RAG": 80},
            "2": {"Baseline LLM": 70, "Vanilla RAG": 75, "HCAD-RAG": 84},
            "3": {"Baseline LLM": 69, "Vanilla RAG": 73, "HCAD-RAG": 86},
        },
        "selection_based_assessment": {
            "HCAD_vs_Vanilla": {"criteria_win_rates_pct": {c: 60 for c in
                ["comprehensiveness", "empowerment", "accuracy", "relevance",
                 "coherence", "clarity", "logical", "flexibility"]}},
        },
        "retrieval_latency_seconds": {"Baseline LLM": 0.01, "Vanilla RAG": 0.2, "HCAD-RAG": 0.9},
    }
    out = plots.plot_all(results, hallu_path=None)
    assert len(out) == 4, out
    for p in out:
        assert Path(p).exists() and Path(p).stat().st_size > 5000
