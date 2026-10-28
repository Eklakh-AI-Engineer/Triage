import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.integrity import (
    ArtifactIntegrityError,
    build_integrity_manifest,
    validate_integrity_manifest,
    write_integrity_manifest,
)
from src.evaluation.claim_factuality import score_claim_factuality


class TestArtifactIntegrity(unittest.TestCase):
    def test_manifest_records_hashes_and_index_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "kg.csv"
            source.write_text("relation,x,y\ntreats,a,b\n", encoding="utf-8")
            manifest = build_integrity_manifest(
                input_paths=[source], indexed_document_count=1, expected_document_count=1,
                relation_counts={"treats": 1}, corpus_composition={"primekg": 1},
                embedding_model="test-embed-v1", index_name="triage", index_version="v1",
                build_version="test-build",
            )
            target = write_integrity_manifest(manifest, Path(tmp) / "manifest.json")
            loaded = validate_integrity_manifest(
                target, required_input_paths=[source], expected_index_name="triage"
            )
            self.assertEqual(loaded["inputs"][str(source)]["sha256"], manifest["inputs"][str(source)]["sha256"])
            self.assertEqual(loaded["index"]["indexed_document_count"], 1)

    def test_missing_manifest_fails_closed(self):
        with self.assertRaisesRegex(ArtifactIntegrityError, "manifest is missing"):
            validate_integrity_manifest("definitely-not-present.json")

    def test_changed_source_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "kg.csv"
            source.write_text("original", encoding="utf-8")
            manifest = build_integrity_manifest(
                input_paths=[source], indexed_document_count=1, expected_document_count=1,
                relation_counts={"r": 1}, corpus_composition={"primekg": 1},
                embedding_model="e1", index_name="triage", index_version="i1", build_version="b1",
            )
            target = write_integrity_manifest(manifest, Path(tmp) / "manifest.json")
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactIntegrityError, "hash mismatch"):
                validate_integrity_manifest(target, required_input_paths=[source])

    def test_incomplete_relation_coverage_is_rejected(self):
        with self.assertRaisesRegex(ArtifactIntegrityError, "Relation coverage"):
            build_integrity_manifest(
                input_paths=[], indexed_document_count=1, expected_document_count=1,
                relation_counts={}, corpus_composition={"primekg": 1},
                embedding_model="e1", index_name="triage", index_version="i1", build_version="b1",
            )


class TestClaimFactuality(unittest.TestCase):
    def test_confusion_matrix_uses_independent_claim_labels(self):
        rows = [
            {"claim_id": "c1", "question_id": "q1", "annotator_id": "clinician-1", "claim": "Claim one", "label": "supported"},
            {"claim_id": "c2", "question_id": "q1", "annotator_id": "clinician-1", "claim": "Claim two", "label": "contradicted"},
            {"claim_id": "c3", "question_id": "q2", "annotator_id": "clinician-2", "claim": "Claim three", "label": "insufficient_evidence"},
            {"claim_id": "c4", "question_id": "q2", "annotator_id": "clinician-2", "claim": "Claim four", "label": "supported"},
        ]
        result = score_claim_factuality(rows, {"c1": False, "c2": True, "c3": False, "c4": True})
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertEqual(result["n_evaluated"], 4)
        self.assertEqual(result["label_source"], "independent_claim_annotation")

    def test_missing_predictions_are_not_counted_as_correct(self):
        rows = [{"claim_id": "c1", "question_id": "q1", "annotator_id": "a1", "claim": "claim", "label": "supported"}]
        result = score_claim_factuality(rows, {})
        self.assertEqual(result["n_evaluated"], 0)
        self.assertEqual(result["missing_prediction_claim_ids"], ["c1"])
        self.assertEqual(result["coverage"], 0.0)

    def test_labels_cannot_be_inferred_or_omitted(self):
        rows = [{"claim_id": "c1", "question_id": "q1", "annotator_id": "a1", "claim": "claim"}]
        with self.assertRaisesRegex(ValueError, "invalid independent label"):
            score_claim_factuality(rows, {"c1": False})


if __name__ == "__main__":
    unittest.main()
