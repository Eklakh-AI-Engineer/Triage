"""Tests for the Europe PMC downloader (src/data/download_pmc.py)."""

import pytest

from src.data.download_pmc import _fulltext_to_text

FULLTEXT_XML = b"""<?xml version="1.0"?>
<article>
  <front>
    <abstract><p>Metformin is contraindicated in severe renal impairment.</p></abstract>
  </front>
  <body>
    <sec><title>Introduction</title>
      <p>Chronic kidney disease affects drug clearance substantially.</p>
    </sec>
    <sec><title>Discussion</title>
      <p>Lactic acidosis risk rises with declining eGFR.</p>
      <p>Monitoring is recommended between eGFR 30 and 45.</p>
    </sec>
  </body>
</article>"""

ABSTRACT_ONLY_XML = b"""<?xml version="1.0"?>
<article>
  <front>
    <abstract><p>Abstract-only open access article about Parkinson's disease.</p></abstract>
  </front>
</article>"""

EMPTY_XML = b"""<?xml version="1.0"?>
<article><front><journal-meta/></front></article>"""


def test_fulltext_extraction_prefers_body():
    text = _fulltext_to_text(FULLTEXT_XML)
    assert "Chronic kidney disease affects drug clearance" in text
    assert "Monitoring is recommended" in text
    # Body text wins over the abstract.
    assert "contraindicated" not in text
    assert len(text.split()) >= 150 or text.count("\n\n") >= 1


def test_fulltext_extraction_falls_back_to_abstract():
    text = _fulltext_to_text(ABSTRACT_ONLY_XML)
    assert "Parkinson's disease" in text


def test_fulltext_extraction_empty_returns_blank():
    assert _fulltext_to_text(EMPTY_XML) == ""


def test_downloader_registered_in_download_all():
    from src.data.download_datasets import download_all
    import inspect
    src = inspect.getsource(download_all)
    assert "download_pmc" in src
    assert "pmc" in src
