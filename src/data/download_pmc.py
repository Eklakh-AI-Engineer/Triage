"""Download open-access clinical articles from Europe PMC as plain text.

Keyless REST API (https://europepmc.org/RestfulWebService):
  1. Search open-access PMC articles with a clinical query.
  2. Fetch each article's full-text XML.
  3. Extract readable body text and save as ``data/raw/pmc/article_N.txt`` -
     the exact format ``data/prepare_chunks.py`` consumes.

Count is controlled by ``PMC_NUM_ARTICLES`` (default 1000; ~15 min at the
default 0.1 s inter-request delay). Failures are per-article and skipped;
the downloader resumes after existing articles.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import requests

from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
DEFAULT_QUERY = '(OPEN_ACCESS:y AND SRC:PMC) AND (TITLE:"clinical" OR ABSTRACT:"clinical")'
_MIN_WORDS = 150  # skip stubs/errata without real body text


def _fulltext_to_text(xml_bytes: bytes) -> str:
    """Readable body text of a JATS full-text XML, falling back to abstract."""
    root = ET.fromstring(xml_bytes)
    body = root.find(".//body")
    if body is not None:
        parts = ["".join(p.itertext()).strip() for p in body.iter("p")]
        parts = [p for p in parts if p]
        if parts:
            return "\n\n".join(parts)
    abstract = root.find(".//abstract")
    if abstract is not None:
        parts = ["".join(p.itertext()).strip() for p in abstract.iter("p")]
        parts = [p for p in parts if p]
        if parts:
            return "\n\n".join(parts)
    return ""


def _search_pmcs(query: str, num_articles: int) -> list[dict]:
    """Open-access hits with pmcid + abstract (abstract is the fallback text)."""
    hits: list[dict] = []
    cursor = "*"
    while len(hits) < num_articles:
        resp = requests.get(
            f"{_BASE}/search",
            params={
                "query": query,
                "format": "json",
                "pageSize": 100,
                "cursorMark": cursor,
                "resultType": "core",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
        for r in results:
            pmcid = (r.get("pmcid") or "").strip()
            if pmcid:
                hits.append({
                    "pmcid": pmcid,
                    "abstract": (r.get("abstractText") or "").strip(),
                })
                if len(hits) >= num_articles:
                    break
        nxt = data.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
    return hits


def download_pmc(
    num_articles: Optional[int] = None,
    force: bool = False,
    query: str = DEFAULT_QUERY,
) -> Path:
    """Fetch ``num_articles`` open-access clinical articles to data/raw/pmc/."""
    if num_articles is None:
        num_articles = int(os.getenv("PMC_NUM_ARTICLES", "1000"))
    if num_articles <= 0:
        raise ValueError("num_articles must be positive")

    out_dir = RAW_DATA_DIR / "pmc"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("article_*.txt"))
    if existing and not force:
        logger.info("PMC: %d articles already present - skipping.", len(existing))
        return out_dir
    start_idx = len(existing)

    pmcids = _search_pmcs(query, num_articles)
    logger.info("PMC: fetching %d OA clinical articles...", len(pmcids))

    saved = 0
    failed_status: dict[int, int] = {}
    for offset, hit in enumerate(pmcids):
        idx = start_idx + offset
        try:
            # NOTE: the full-text endpoint is /rest/{PMCID}/fullTextXML -
            # there is NO extra /PMC/ source segment (that form 404s).
            resp = requests.get(f"{_BASE}/{hit['pmcid']}/fullTextXML", timeout=60)
            if resp.status_code != 200:
                failed_status[resp.status_code] = failed_status.get(resp.status_code, 0) + 1
            text = ""
            if resp.status_code == 200:
                text = _fulltext_to_text(resp.content)
            if len(text.split()) < _MIN_WORDS:
                # Not every OA article ships full-text XML; the abstract
                # from the search result keeps the corpus complete.
                text = hit["abstract"]
            if len(text.split()) < _MIN_WORDS:
                continue
            (out_dir / f"article_{idx}.txt").write_text(text, encoding="utf-8")
            saved += 1
            if saved % 100 == 0:
                logger.info("PMC: saved %d/%d articles", saved, len(pmcids))
        except Exception as exc:
            logger.debug("PMC %s failed: %s", hit["pmcid"], exc)
        time.sleep(0.1)

    if failed_status:
        logger.warning("PMC: non-200 full-text responses: %s", failed_status)
    logger.info("PMC: saved %d articles to %s", saved, out_dir)
    if saved == 0 and start_idx == 0:
        raise RuntimeError("PMC download produced no usable articles")
    return out_dir
