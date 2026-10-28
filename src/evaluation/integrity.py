"""Integrity manifests and fail-closed validation for benchmark artifacts.

A benchmark manifest is intentionally explicit: builds must record the inputs,
index counts, relation coverage, corpus composition and software/model versions.
The benchmark must not infer that an artifact is valid merely because it opens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class ArtifactIntegrityError(RuntimeError):
    """Raised when an evaluation artifact is missing or fails validation."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ArtifactIntegrityError(f"Required input file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_integrity_manifest(
    *,
    input_paths: Iterable[str | Path],
    indexed_document_count: int,
    expected_document_count: int | None,
    relation_counts: dict[str, int],
    corpus_composition: dict[str, int],
    embedding_model: str,
    index_name: str,
    index_version: str,
    build_version: str,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest from explicit build observations."""
    if indexed_document_count <= 0:
        raise ArtifactIntegrityError("Indexed document count must be greater than zero")
    if expected_document_count is not None and indexed_document_count != expected_document_count:
        raise ArtifactIntegrityError(
            f"Indexed document count mismatch: expected {expected_document_count}, "
            f"observed {indexed_document_count}"
        )
    if not relation_counts or any(int(count) < 0 for count in relation_counts.values()):
        raise ArtifactIntegrityError("Relation coverage is missing or contains negative counts")
    if sum(int(v) for v in relation_counts.values()) <= 0:
        raise ArtifactIntegrityError("Relation coverage contains no indexed relations")
    if not corpus_composition or any(int(count) < 0 for count in corpus_composition.values()):
        raise ArtifactIntegrityError("Corpus composition is missing or contains negative counts")
    if sum(int(v) for v in corpus_composition.values()) <= 0:
        raise ArtifactIntegrityError("Corpus composition is empty")
    required_versions = {
        "embedding_model": embedding_model,
        "index_name": index_name,
        "index_version": index_version,
        "build_version": build_version,
    }
    missing = [key for key, value in required_versions.items() if not str(value).strip()]
    if missing:
        raise ArtifactIntegrityError(f"Missing required version metadata: {', '.join(missing)}")

    inputs = {}
    for raw_path in input_paths:
        path = Path(raw_path)
        inputs[str(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if not inputs:
        raise ArtifactIntegrityError("At least one source input hash is required")

    return {
        "schema_version": 1,
        "inputs": inputs,
        "index": {
            "name": index_name,
            "version": index_version,
            "embedding_model": embedding_model,
            "indexed_document_count": int(indexed_document_count),
            "expected_document_count": expected_document_count,
            "relation_counts": {str(k): int(v) for k, v in sorted(relation_counts.items())},
            "corpus_composition": {str(k): int(v) for k, v in sorted(corpus_composition.items())},
        },
        "build_version": build_version,
    }


def validate_integrity_manifest(
    manifest_path: str | Path,
    *,
    required_input_paths: Iterable[str | Path] = (),
    expected_index_name: str | None = None,
    min_documents: int = 1,
) -> dict[str, Any]:
    """Validate manifest completeness, source hashes and optional index constraints."""
    path = Path(manifest_path)
    if not path.is_file():
        raise ArtifactIntegrityError(
            f"Integrity manifest is missing: {path}. Rebuild the corpus/index before benchmarking."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Cannot read integrity manifest {path}: {exc}") from exc

    if manifest.get("schema_version") != 1:
        raise ArtifactIntegrityError("Unsupported or missing integrity manifest schema_version")
    inputs = manifest.get("inputs")
    index = manifest.get("index")
    if not isinstance(inputs, dict) or not inputs:
        raise ArtifactIntegrityError("Manifest has no source input hashes")
    if not isinstance(index, dict):
        raise ArtifactIntegrityError("Manifest has no index metadata")
    for field in ("name", "version", "embedding_model", "indexed_document_count",
                  "relation_counts", "corpus_composition"):
        if field not in index or index[field] in (None, "", {}):
            raise ArtifactIntegrityError(f"Manifest is missing index.{field}")
    count = index["indexed_document_count"]
    if not isinstance(count, int) or count < min_documents:
        raise ArtifactIntegrityError(
            f"Index contains {count!r} documents; minimum required is {min_documents}"
        )
    if expected_index_name and index["name"] != expected_index_name:
        raise ArtifactIntegrityError(
            f"Index name mismatch: expected {expected_index_name!r}, got {index['name']!r}"
        )
    relations = index["relation_counts"]
    composition = index["corpus_composition"]
    if not isinstance(relations, dict) or not relations or sum(relations.values()) <= 0:
        raise ArtifactIntegrityError("Manifest has no usable relation coverage")
    if not isinstance(composition, dict) or not composition or sum(composition.values()) <= 0:
        raise ArtifactIntegrityError("Manifest has no usable corpus composition")
    if not str(manifest.get("build_version", "")).strip():
        raise ArtifactIntegrityError("Manifest is missing build_version")

    # Verify every recorded source by default; a manifest with stale paths
    # or hashes is not sufficient provenance for a benchmark.
    for raw_record_path, record in inputs.items():
        input_path = Path(raw_record_path)
        actual = sha256_file(input_path)
        if not isinstance(record, dict) or record.get("sha256") != actual:
            raise ArtifactIntegrityError(f"Input hash mismatch for {input_path}")
    for raw_path in required_input_paths:
        input_path = Path(raw_path)
        if str(input_path) not in inputs:
            raise ArtifactIntegrityError(f"Manifest does not record required input: {input_path}")
    return manifest


def write_integrity_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write a manifest atomically so a partial write cannot appear valid."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
