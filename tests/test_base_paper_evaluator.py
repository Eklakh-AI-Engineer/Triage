import sys
import os
import types
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.base_paper_evaluator import (
    BasePaperEvaluator,
    ScoringResult,
    SelectionResult,
)

class TestBasePaperEvaluator(unittest.TestCase):
    def test_incomplete_scores_are_rejected(self):
        mock_judge = MagicMock()
        mock_judge.chat.return_value = '{"comprehensiveness": 80}'
        evaluator = BasePaperEvaluator(judge_llm=mock_judge)
        with self.assertRaisesRegex(ValueError, "missing score fields"):
            evaluator.evaluate_scoring("q", "a", "reference")

    def test_scoring_fallback_on_error(self):
        mock_judge = MagicMock()
        mock_judge.chat.return_value = '{"comprehensiveness": 80}'
        evaluator = BasePaperEvaluator(judge_llm=mock_judge)
        res = evaluator.evaluate_scoring("q", "a", "reference", fallback_on_error=True)
        self.assertEqual(res.overall_score, 50.0)
        self.assertEqual(res.comprehensiveness, 50.0)

    def test_selection_fallback_on_error(self):
        mock_judge = MagicMock()
        mock_judge.chat.return_value = '{"overall_winner": "bad"}'
        evaluator = BasePaperEvaluator(judge_llm=mock_judge)
        res = evaluator.evaluate_selection("q", "a", "b", "ref", fallback_on_error=True)
        self.assertEqual(res["overall_winner"], "Tie")
        self.assertTrue(all(w == "Tie" for w in res["criteria_winners"].values()))

    def test_json_parsing_and_scoring(self):
        mock_judge = MagicMock()
        mock_judge.chat.return_value = """
        ```json
        {
            "comprehensiveness": 88,
            "diversity": 82,
            "empowerment": 90,
            "logical": 92,
            "readability": 95,
            "rationale": "High clinical grounding with zero hallucinations."
        }
        ```
        """
        evaluator = BasePaperEvaluator(judge_llm=mock_judge)
        res = evaluator.evaluate_scoring(
            question="What is the treatment for Central Neurogenic Hyperventilation?",
            answer="Primary therapy targets the underlying intracranial etiology...",
            reference_document="Pontine lesions and cerebral lymphoma can trigger hyperventilation...",
        )

        self.assertIsInstance(res, ScoringResult)
        self.assertEqual(res.comprehensiveness, 88.0)
        self.assertEqual(res.diversity, 82.0)
        self.assertEqual(res.empowerment, 90.0)
        self.assertEqual(res.logical, 92.0)
        self.assertEqual(res.readability, 95.0)
        expected_overall = (88 + 82 + 90 + 92 + 95) / 5.0
        self.assertAlmostEqual(res.overall_score, expected_overall)
        self.assertIn("High clinical grounding", res.rationale)

    def test_scoring_prompt_supplies_evidence_for_grounding_check(self):
        mock_judge = MagicMock()
        mock_judge.chat.return_value = (
            '{"comprehensiveness":80,"diversity":80,"empowerment":80,'
            '"logical":80,"readability":80}'
        )
        evaluator = BasePaperEvaluator(judge_llm=mock_judge)
        evaluator.evaluate_scoring(
            "q", "The evidence supports this. [Evidence 1]", "reference",
            answer_evidence="[Evidence 1] supporting relation",
        )
        prompt = mock_judge.chat.call_args.args[0][1]["content"]
        self.assertIn("[Evidence 1] supporting relation", prompt)
        self.assertIn("citation tags alone earn no credit", prompt)

    def test_selection_parsing_and_mapping(self):
        mock_judge = MagicMock()
        mock_judge.chat.return_value = """
        {
            "comprehensiveness": {"winner": "Answer 1", "reason": "More detailed"},
            "empowerment": {"winner": "Answer 1", "reason": "No hallucination"},
            "accuracy": {"winner": "Answer 1", "reason": "Factual alignment"},
            "relevance": {"winner": "Answer 1", "reason": "Direct"},
            "coherence": {"winner": "Answer 1", "reason": "Smooth flow"},
            "clarity": {"winner": "Answer 1", "reason": "Crisp"},
            "logical": {"winner": "Answer 1", "reason": "Consistent"},
            "flexibility": {"winner": "Answer 2", "reason": "Handles nuance better"},
            "overall_winner": "Answer 1",
            "summary": "System A is significantly more reliable."
        }
        """
        evaluator = BasePaperEvaluator(judge_llm=mock_judge)
        res = evaluator.evaluate_selection(
            question="Management of neurodegenerative tremor?",
            answer_a="HCAD-RAG answer...",
            answer_b="Vanilla RAG answer...",
            reference_document="Clinical guideline text...",
            label_a="HCAD-RAG",
            label_b="Vanilla RAG",
            evidence_a="[Evidence 1] HCAD fact",
            evidence_b="[Context 1] Vanilla fact",
        )

        self.assertEqual(res["overall_winner"], "HCAD-RAG")
        self.assertEqual(res["criteria_winners"]["accuracy"], "HCAD-RAG")
        self.assertEqual(res["criteria_winners"]["flexibility"], "Vanilla RAG")
        prompt = mock_judge.chat.call_args.args[0][1]["content"]
        self.assertIn("[Evidence 1] HCAD fact", prompt)
        self.assertIn("[Context 1] Vanilla fact", prompt)

    def test_decision_gate_keeps_more_relation_diverse_evidence(self):
        from src.core.decision_gate import DecisionGate

        gate = DecisionGate(use_jev=False, max_evidence_items=8, token_budget_words=350)
        candidates = []
        for index in range(10):
            relation = "TREATS" if index < 5 else f"RELATION_{index}"
            candidates.append({
                "document": f"candidate {index} evidence",
                "metadata": {
                    "head_name": f"head {index}",
                    "relation": relation,
                    "tail_name": f"tail {index}",
                },
                "phi_final": 0.9 - index / 100,
            })
        retained = gate.verify_and_prune_candidates(candidates, "question")
        self.assertEqual(len(retained), 8)
        self.assertGreaterEqual(len({c["metadata"]["relation"] for c in retained}), 4)

    def test_curated_fallback_truncates_without_medqa(self):
        """Audit decision: the MedQA fallback is REMOVED - the base paper
        never uses MCQ benchmarks. Oversize requests truncate to the curated
        pool; question-format mixing is gone."""
        from src.evaluation.run_base_paper_benchmark import load_clinical_benchmark_samples

        samples = load_clinical_benchmark_samples(8)
        self.assertEqual(len(samples), 5)  # curated pool only, truncated
        self.assertTrue(all(s["source"] == "curated" for s in samples))

    def test_heuristic_recovery(self):
        evaluator = BasePaperEvaluator(judge_llm=MagicMock())
        malformed = 'Random prefix {"comprehensiveness": 85, "diversity": 75, "empowerment": 88, "logical": 90, "readability": 92} trailing text'
        data = evaluator._extract_json(malformed)
        self.assertEqual(data["comprehensiveness"], 85.0)
        self.assertEqual(data["readability"], 92.0)

    def test_judge_uses_distinct_ollama_model(self):
        from src.config import JUDGE_HF_MODEL, JUDGE_OLLAMA_MODEL
        from src.core.llm_client import UnifiedLLMClient

        client = UnifiedLLMClient(model=JUDGE_HF_MODEL, provider="ollama")
        self.assertEqual(client.model, JUDGE_OLLAMA_MODEL)

    def test_qwen3_ollama_judge_disables_thinking_for_parseable_json(self):
        from src.config import JUDGE_HF_MODEL, JUDGE_OLLAMA_MODEL
        from src.core.llm_client import UnifiedLLMClient

        response_content = (
            '{"comprehensiveness":80,"diversity":75,"empowerment":85,'
            '"logical":90,"readability":95}'
        )
        ollama_module = types.ModuleType("ollama")
        ollama_module.chat = MagicMock(
            return_value=types.SimpleNamespace(
                message=types.SimpleNamespace(content=response_content)
            )
        )
        import src.config as _cfg
        saved = _cfg.JUDGE_OLLAMA_MODEL
        _cfg.JUDGE_OLLAMA_MODEL = JUDGE_OLLAMA_MODEL = "qwen3:14b"  # qwen3 think-path test
        try:
            client = UnifiedLLMClient(model=JUDGE_HF_MODEL, provider="ollama")
        except Exception:
            _cfg.JUDGE_OLLAMA_MODEL = saved
            raise

        with patch.dict(sys.modules, {"ollama": ollama_module}):
            response = client.chat(
                [{"role": "user", "content": "Return valid JSON scores."}]
            )

        self.assertEqual(response, response_content)
        self.assertEqual(client.model, JUDGE_OLLAMA_MODEL)
        self.assertIs(ollama_module.chat.call_args.kwargs["think"], False)
        _cfg.JUDGE_OLLAMA_MODEL = saved  # do not pollute later tests

    def test_huggingface_failure_falls_back_to_distinct_judge(self):
        from src.config import JUDGE_HF_MODEL, JUDGE_OLLAMA_MODEL
        from src.core.llm_client import UnifiedLLMClient

        ollama_module = types.ModuleType("ollama")
        ollama_module.chat = MagicMock(
            return_value=types.SimpleNamespace(
                message=types.SimpleNamespace(content='{"ok": true}')
            )
        )
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch.dict(
            sys.modules, {"ollama": ollama_module}
        ):
            client = UnifiedLLMClient(model=JUDGE_HF_MODEL, provider="huggingface")
            client._hf_client = MagicMock()
            client._hf_client.chat.completions.create.side_effect = RuntimeError("offline")
            # Kaggle may have a real Ollama HTTP server running. Force the
            # HTTP path offline so this test reaches the mocked SDK fallback.
            with patch("time.sleep"), patch(
                "src.core.llm_client.requests.post", side_effect=RuntimeError("offline")
            ):
                response = client.chat([{"role": "user", "content": "test"}])

        self.assertEqual(response, '{"ok": true}')
        # Audit fix (item 24): fallback is per-call - the instance must NOT be
        # permanently degraded to Ollama (a judge client previously flipped
        # silently mid-benchmark, which is exactly what the committed artifact
        # showed: judge qwen3:14b instead of the configured 72B).
        self.assertEqual(client.provider, "huggingface")
        self.assertEqual(ollama_module.chat.call_args.kwargs["model"], JUDGE_OLLAMA_MODEL)

    def test_failed_answer_is_not_accepted_for_scoring(self):
        from src.evaluation.run_base_paper_benchmark import _validate_generated_response

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            _validate_generated_response("HCAD-RAG", "[Error: model unavailable]")
        self.assertEqual(_validate_generated_response("HCAD-RAG", "Answer."), "Answer.")

if __name__ == "__main__":
    unittest.main()

class TestNvidiaNimProvider(unittest.TestCase):
    def test_nvidia_judge_and_fallback_model_routing(self):
        from src.core.llm_client import UnifiedLLMClient
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test", "LLM_PROVIDER": "nvidia"}):
            import src.config as cfg
            cfg.NVIDIA_API_KEY = "nvapi-test"
            client = UnifiedLLMClient(model=cfg.JUDGE_HF_MODEL, provider="nvidia")
            self.assertEqual(client.provider, "nvidia")
            self.assertEqual(client.model, cfg.JUDGE_NVIDIA_MODEL)  # judge routed to NIM judge model
            gen = UnifiedLLMClient(provider="nvidia")
            self.assertEqual(gen.model, cfg.LLM_NVIDIA_MODEL)

    def test_nvidia_chat_uses_openai_compatible_endpoint(self):
        import types as _t
        fake_resp = _t.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"ok": true}'}}]},
        )
        import src.core.llm_client as mod
        captured = {}
        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"], captured["headers"], captured["payload"] = url, headers, json
            return fake_resp
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test", "LLM_PROVIDER": "nvidia"}):
            import src.config as cfg
            from src.core.llm_client import UnifiedLLMClient
            cfg.NVIDIA_API_KEY = "nvapi-test"
            client = UnifiedLLMClient(provider="nvidia")
            with patch.object(mod.requests, "post", fake_post):
                out = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out, '{"ok": true}')
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["headers"]["Authorization"], "Bearer nvapi-test")
