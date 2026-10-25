import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")


def _triarm_results():
    """Complete tri-arm fixture: three systems, three selection panels,
    baseline latency/citation arms, three difficulty tiers."""
    return {
        "scoring_based_assessment": {
            "Baseline LLM": {d: 70 for d in
                             ["comprehensiveness", "diversity", "empowerment",
                              "logical", "readability", "overall_score"]},
            "Vanilla RAG": {d: 75 for d in
                            ["comprehensiveness", "diversity", "empowerment",
                             "logical", "readability", "overall_score"]},
            "HCAD-RAG": {d: 82 for d in
                         ["comprehensiveness", "diversity", "empowerment",
                          "logical", "readability", "overall_score"]},
        },
        "scoring_by_difficulty_tier": {
            "1": {"Baseline LLM": 72, "Vanilla RAG": 76, "HCAD-RAG": 80},
            "2": {"Baseline LLM": 70, "Vanilla RAG": 75, "HCAD-RAG": 84},
            "3": {"Baseline LLM": 69, "Vanilla RAG": 73, "HCAD-RAG": 86},
        },
        "selection_based_assessment": {
            "HCAD_vs_Baseline": {
                "target_system": "HCAD-RAG", "overall_win_rate_pct": 55.0,
                "criteria_win_rates_pct": {c: 60 for c in
                    ["comprehensiveness", "empowerment", "accuracy", "relevance",
                     "coherence", "clarity", "logical", "flexibility"]}},
            "HCAD_vs_Vanilla": {
                "target_system": "HCAD-RAG", "overall_win_rate_pct": 52.0,
                "criteria_win_rates_pct": {c: 58 for c in
                    ["comprehensiveness", "empowerment", "accuracy", "relevance",
                     "coherence", "clarity", "logical", "flexibility"]}},
            "Vanilla_vs_Baseline": {
                "target_system": "Vanilla RAG", "overall_win_rate_pct": 61.0,
                "criteria_win_rates_pct": {c: 62 for c in
                    ["comprehensiveness", "empowerment", "accuracy", "relevance",
                     "coherence", "clarity", "logical", "flexibility"]}},
        },
        "retrieval_latency_seconds": {
            "Baseline LLM": 0.0, "Vanilla RAG": 0.2, "HCAD-RAG": 0.9,
        },
        "citation_precision": {
            "Baseline LLM": {"n_answers_with_citations": 0, "n_answers": 50,
                             "total_citations": 0, "mean_precision": None},
            "Vanilla RAG": {"n_answers_with_citations": 1, "n_answers": 50,
                            "total_citations": 2, "mean_precision": 1.0},
            "HCAD-RAG": {"n_answers_with_citations": 30, "n_answers": 50,
                         "total_citations": 98, "mean_precision": 0.9667},
        },
    }


def test_plots_render_from_results_json(tmp_path, monkeypatch):
    from src.evaluation import plots
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    out = plots.plot_all(_triarm_results(), hallu_path=None)
    assert len(out) == 5, out
    for p in out:
        assert Path(p).exists() and Path(p).stat().st_size > 5000


def test_triarm_gate_refuses_two_way_comparison(tmp_path, monkeypatch):
    """Tri-arm protocol: a figure missing any of the three systems must be
    refused, never silently rendered as a 2-way comparison."""
    from src.evaluation import plots
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    results = _triarm_results()
    # remove the baseline arm from EVERY data section the renderers read
    del results["scoring_based_assessment"]["Baseline LLM"]
    for tier in results["scoring_by_difficulty_tier"].values():
        del tier["Baseline LLM"]
    results["retrieval_latency_seconds"].pop("Baseline LLM", None)
    results["citation_precision"].pop("Baseline LLM", None)
    for renderer in (plots.plot_dimension_scores, plots.plot_difficulty_tiers,
                     plots.plot_latency_performance, plots.plot_citation_grounding):
        try:
            renderer(results)
        except ValueError as exc:
            assert "Tri-arm protocol" in str(exc), renderer.__name__
        else:
            raise AssertionError(
                f"{renderer.__name__} rendered a 2-way comparison")
