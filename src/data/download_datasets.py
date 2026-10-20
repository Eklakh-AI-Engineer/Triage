"""
Download and prepare all datasets for the HCAD-RAG pipeline.

Datasets :
  1. PrimeKG  - Biomedical knowledge graph (Harvard Dataverse, ~3 GB CSV).
  2. MedHallu - Medical hallucination benchmark with labeled & artificial splits.

Usage:
    python -m src.data.download_datasets          # download everything
    python -m src.data.download_datasets --only primekg
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.config import (
    MEDHALLU_CONFIG_ARTIFICIAL,
    MEDHALLU_CONFIG_LABELED,
    MEDHALLU_DATASET_ID,
    PRIMEKG_FILENAME,
    PRIMEKG_URL,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)

# Constants
_CHUNK_SIZE = 8192  # bytes per streaming chunk

# PrimeKG

def download_primekg(force: bool = False) -> Path:
    """Download the PrimeKG knowledge-graph CSV from Harvard Dataverse.

    The file is large (~3 GB), so we stream it with a ``tqdm`` progress bar.
    If the file already exists and *force* is ``False`` the download is skipped.

    Args:
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded CSV.

    Raises:
        requests.HTTPError: If the download request fails.
        OSError: If writing to disk fails.
    """
    dest = RAW_DATA_DIR / "primekg" / PRIMEKG_FILENAME
    alt_dest = RAW_DATA_DIR / PRIMEKG_FILENAME

    if (dest.exists() or alt_dest.exists()) and not force:
        target = dest if dest.exists() else alt_dest
        logger.info("PrimeKG already downloaded at %s - skipping.", target)
        return target

    logger.info("Downloading PrimeKG from %s …", PRIMEKG_URL)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # retries with backoff and atomic rename - a
    # truncated download previously left a corrupt kg.csv silently in place.
    import os as _os
    session = requests.Session()
    retries = requests.adapters.Retry(total=3, backoff_factor=2.0,
                                     status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
    tmp_dest = dest.with_suffix(".part")
    try:
        response = session.get(PRIMEKG_URL, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to start PrimeKG download: %s", exc)
        raise

    total_size = int(response.headers.get("content-length", 0))

    try:
        with (
            open(tmp_dest, "wb") as fout,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="PrimeKG",
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                fout.write(chunk)
                pbar.update(len(chunk))
    except OSError as exc:
        logger.error("Error writing PrimeKG to %s: %s", tmp_dest, exc)
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise

    if tmp_dest.exists():
        expected = int(response.headers.get("content-length", 0))
        actual = tmp_dest.stat().st_size
        if expected and actual != expected:
            tmp_dest.unlink()
            raise IOError(
                f"PrimeKG download incomplete: {actual} bytes of {expected} - "
                "size validation failed "
            )
        _os.replace(tmp_dest, dest)  # atomic publish

    logger.info("PrimeKG saved to %s (%.2f MB).", dest, dest.stat().st_size / 1e6)
    return dest

# MedHallu

def download_medhallu(force: bool = False) -> Path:
    """Load MedHallu from HuggingFace (labeled + artificial) and save as JSON.

    Saves each config/split pair as a separate JSON file under
    ``RAW_DATA_DIR / 'medhallu/'``.

    Args:
        force: Re-download even if files already exist.

    Returns:
        Path to the ``medhallu/`` directory.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
        Exception: On HuggingFace Hub download errors.
    """
    out_dir = RAW_DATA_DIR / "medhallu"

    # Quick existence check: if both labeled and artificial dirs have content
    labeled_dir = out_dir / MEDHALLU_CONFIG_LABELED
    artificial_dir = out_dir / MEDHALLU_CONFIG_ARTIFICIAL
    if (
        not force
        and labeled_dir.exists()
        and artificial_dir.exists()
        and any(labeled_dir.iterdir())
        and any(artificial_dir.iterdir())
    ):
        logger.info("MedHallu already downloaded at %s - skipping.", out_dir)
        return out_dir

    logger.info(
        "Loading MedHallu dataset from HuggingFace (%s) …", MEDHALLU_DATASET_ID
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.error(
            "The 'datasets' library is required to download MedHallu. "
            "Install it with: pip install datasets"
        )
        raise ImportError(
            "Missing dependency: pip install datasets"
        ) from exc

    for config_name in [MEDHALLU_CONFIG_LABELED, MEDHALLU_CONFIG_ARTIFICIAL]:
        config_dir = out_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        try:
            ds = load_dataset(
                MEDHALLU_DATASET_ID,
                config_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to load MedHallu config '%s': %s", config_name, exc
            )
            raise

        for split_name in ds:
            records = [dict(row) for row in ds[split_name]]
            file_path = config_dir / f"{split_name}.json"
            with open(file_path, "w", encoding="utf-8") as fout:
                json.dump(records, fout, indent=2, ensure_ascii=False)
            logger.info(
                "MedHallu %s/%s: %d samples → %s",
                config_name,
                split_name,
                len(records),
                file_path,
            )

    return out_dir

# Orchestrator

def download_all(
    force: bool = False,
    only: Optional[str] = None,
) -> dict[str, Path]:
    """Download all (or a specific) dataset.

    Args:
        force: Re-download even if files already exist.
        only:  If provided, only download the named dataset
               (one of ``'primekg'``, ``'medhallu'``).

    Returns:
        Mapping of dataset name → output path.

    Args (of ``only``):
        one of ``'primekg'``, ``'medhallu'``, ``'pmc'``.
    """
    from src.data.download_pmc import download_pmc as _download_pmc
    downloaders = {
        "primekg": download_primekg,
        "medhallu": download_medhallu,
        "pmc": _download_pmc,
    }

    if only is not None:
        key = only.lower()
        if key not in downloaders:
            raise ValueError(
                f"Unknown dataset '{only}'. Choose from {list(downloaders)}."
            )
        downloaders = {key: downloaders[key]}

    results: dict[str, Path] = {}
    for name, fn in downloaders.items():
        logger.info("=" * 60)
        logger.info("Downloading: %s", name)
        logger.info("=" * 60)
        try:
            results[name] = fn(force=force)
        except Exception:
            logger.exception("Failed to download %s - continuing.", name)

    return results

# CLI entry-point

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download datasets for HCAD-RAG."
    )
    parser.add_argument(
        "--only",
        choices=["primekg", "medhallu"],
        default=None,
        help="Download only the specified dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    args = parser.parse_args()

    paths = download_all(force=args.force, only=args.only)
    for dataset_name, path in paths.items():
        print(f"  {dataset_name}: {path}")
