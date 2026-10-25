import logging
import re
from typing import Any

from src.config import NER_MODELS

logger = logging.getLogger(__name__)

EntityDict = dict[str, list[dict[str, Any]]]

# Label mapping from SciSpacy model labels → unified HCAD-RAG labels
_BC5CDR_LABEL_MAP = {
    "CHEMICAL": "Drug",
    "DISEASE": "Disease",
}

_JNLPBA_LABEL_MAP = {
    "PROTEIN": "Protein",
    "DNA": "Gene",      # DNA mentions often correspond to genes
    "RNA": "Gene",      # RNA mentions mapped to Gene for simplicity
    "CELL_TYPE": "Cell",
    "CELL_LINE": "Cell",
}

# Canonical empty result
_EMPTY_ENTITIES: EntityDict = {
    "Drug": [],
    "Disease": [],
    "Protein": [],
    "Gene": [],
    "Cell": [],
}

def _empty_entities() -> EntityDict:
    """Return a fresh copy of the empty entity dict."""
    return {k: list(v) for k, v in _EMPTY_ENTITIES.items()}

def _deduplicate_spans(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove overlapping entity spans, keeping longer spans.

    For two entities whose character ranges overlap, the one with the
    longer text span is retained.  Ties are broken by earlier start
    position.
    """
    if not entities:
        return []

    # Sort by span length descending, then start ascending
    sorted_ents = sorted(entities, key=lambda e: (-(e["end"] - e["start"]), e["start"]))
    kept: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    for ent in sorted_ents:
        start, end = ent["start"], ent["end"]
        overlaps = any(
            not (end <= occ_start or start >= occ_end)
            for occ_start, occ_end in occupied
        )
        if not overlaps:
            kept.append(ent)
            occupied.append((start, end))

    # Return in document order
    return sorted(kept, key=lambda e: e["start"])

# BiomedicalNER - SciSpacy-backed extractor
class BiomedicalNER:
    """Biomedical NER using SciSpacy models .

    Loads two SciSpacy models:
      - ``en_ner_bc5cdr_md`` for DISEASE + CHEMICAL (mapped to Drug)
      - ``en_ner_jnlpba_md`` for PROTEIN, DNA, RNA, CELL_TYPE, CELL_LINE

    Falls back to :class:`SimpleNER` if models cannot be loaded.

    Parameters
    ----------
    lazy : bool
        If True, defer model loading until first use.
    """

    def __init__(self, lazy: bool = False) -> None:
        self._models_loaded = False
        self._nlp_bc5cdr = None
        self._nlp_jnlpba = None
        self._fallback: SimpleNER | None = None
        self.degraded: bool = False  # surface degraded mode to callers

        if not lazy:
            self._load_models()

    # Internal helpers
    def _load_models(self) -> None:
        """Load each SciSpacy model independently .

        Previously both models loaded in one try-block, so a missing jnlpba
        threw the whole loader into the regex fallback even when bc5cdr was
        available. Each model now fails independently; the regex fallback is
        used only for the entity types with no loaded model, and `degraded`
        is set so callers can detect reduced-capability mode.
        """
        if self._models_loaded:
            return

        bc5cdr_name = NER_MODELS.get("drug_disease", "en_ner_bc5cdr_md")
        jnlpba_name = NER_MODELS.get("molecular", "en_ner_jnlpba_md")

        try:
            import spacy
        except ImportError as exc:
            logger.warning("spaCy not installed (%s). Using SimpleNER fallback.", exc)
            self._fallback = SimpleNER()
            self.degraded = True
            self._models_loaded = True
            return

        import gc
        import time

        for name, attr in ((bc5cdr_name, "_nlp_bc5cdr"), (jnlpba_name, "_nlp_jnlpba")):
            # The observed Kaggle failure ("Python int too large to convert
            # to C ssize_t" during spacy.load) is intermittent: the same
            # models load fine in the test-suite and demo processes of the
            # same session. Retry with a GC pause covers the transient
            # class; the final failure keeps its full traceback so the
            # persistent class can be root-caused from the log.
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    logger.info("Loading SciSpacy model: %s%s", name,
                                "" if attempt == 1 else f" (attempt {attempt}/{attempts})")
                    setattr(self, attr, spacy.load(name))
                    break
                except Exception as exc:
                    if attempt == attempts:
                        from importlib.metadata import version as _pkg_version
                        import numpy
                        try:
                            _stack = (f"murmurhash {_pkg_version('murmurhash')}, "
                                      f"numpy {numpy.__version__}, "
                                      f"spacy {spacy.__version__}")
                        except Exception:
                            _stack = "(version lookup failed)"
                        logger.warning(
                            "Failed to load %s after %d attempts. "
                            "Resolved stack: %s.",
                            name, attempts, _stack,
                            exc_info=True)
                        self.degraded = True
                    else:
                        logger.warning("Load attempt %d/%d for %s failed (%s); retrying.",
                                       attempt, attempts, name, exc)
                    gc.collect()
                    time.sleep(2.0)

        if self._nlp_bc5cdr is None and self._nlp_jnlpba is None:
            self._fallback = SimpleNER()
        self._models_loaded = True

    def _extract_with_model(
        self,
        text: str,
        nlp,
        label_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Run a single spaCy NER model and return mapped entities."""
        results: list[dict[str, Any]] = []
        doc = nlp(text)
        for ent in doc.ents:
            mapped_label = label_map.get(ent.label_)
            if mapped_label:
                results.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": mapped_label,
                })
        return results

    # Public API
    def extract_entities(self, text: str) -> EntityDict:
        """Extract biomedical entities from *text*.

        Returns
        -------
        dict[str, list[dict]]
            Keys: ``"Drug"``, ``"Disease"``, ``"Protein"``, ``"Gene"``,
            ``"Cell"``.  Each value is a list of entity dicts with keys
            ``text``, ``start``, ``end``, ``label``.
        """
        self._load_models()

        # Audit item 20: the fallback covers only what no loaded model covers.
        if self._fallback is not None and self._nlp_bc5cdr is None and self._nlp_jnlpba is None:
            if self.degraded:
                logger.warning("BiomedicalNER running in DEGRADED (regex) mode.")
            return self._fallback.extract_entities(text)

        entities = _empty_entities()
        all_spans: list[dict[str, Any]] = []

        # Run bc5cdr for drugs & diseases
        if self._nlp_bc5cdr is not None:
            all_spans.extend(
                self._extract_with_model(text, self._nlp_bc5cdr, _BC5CDR_LABEL_MAP)
            )

        # Run jnlpba for molecular entities
        if self._nlp_jnlpba is not None:
            all_spans.extend(
                self._extract_with_model(text, self._nlp_jnlpba, _JNLPBA_LABEL_MAP)
            )

        # Deduplicate overlapping spans
        all_spans = _deduplicate_spans(all_spans)

        # Bin into categories
        for ent in all_spans:
            label = ent["label"]
            if label in entities:
                entities[label].append(ent)

        logger.debug(
            "Extracted entities: %s",
            {k: len(v) for k, v in entities.items()},
        )
        return entities

# SimpleNER - Regex / word-list fallback

# Curated word lists for basic matching (case-insensitive)
_DRUG_PATTERNS: list[str] = [
    r"\b(?:metformin|aspirin|ibuprofen|acetaminophen|amoxicillin|lisinopril"
    r"|atorvastatin|omeprazole|amlodipine|metoprolol|losartan|simvastatin"
    r"|levothyroxine|gabapentin|prednisone|furosemide|warfarin|heparin"
    r"|insulin|penicillin|ciprofloxacin|azithromycin|doxycycline"
    r"|hydroxychloroquine|remdesivir|dexamethasone|clopidogrel|tamoxifen"
    r"|rituximab|trastuzumab|nivolumab|pembrolizumab|erlotinib|sorafenib"
    r"|gefitinib|sunitinib|paclitaxel|cisplatin|doxorubicin|cyclophosphamide"
    r"|fluorouracil|capecitabine|carboplatin|docetaxel|vincristine)\b",
]

_DISEASE_PATTERNS: list[str] = [
    r"\b(?:diabetes|hypertension|cancer|asthma|pneumonia|stroke|epilepsy"
    r"|alzheimer(?:'?s)?|parkinson(?:'?s)?|hepatitis|cirrhosis|arthritis"
    r"|leukemia|lymphoma|melanoma|carcinoma|sarcoma|fibrosis|anemia"
    r"|thrombosis|embolism|myocardial\s+infarction|heart\s+failure"
    r"|coronary\s+artery\s+disease|atrial\s+fibrillation|copd"
    r"|chronic\s+obstructive\s+pulmonary\s+disease"
    r"|type\s+[12]\s+diabetes|breast\s+cancer|lung\s+cancer"
    r"|prostate\s+cancer|colorectal\s+cancer|renal\s+failure"
    r"|kidney\s+disease|liver\s+disease|sepsis|meningitis"
    r"|tuberculosis|malaria|hiv|aids|covid-?19|influenza)\b",
]

# Audit item 20: the previous generic alternative `[A-Z]{2,5}\d{0,2}` was
# compiled with re.IGNORECASE, matching every 2-5 letter word ("the", "is").
# The fallback now uses a curated list only, case-sensitively.
_PROTEIN_PATTERNS: list[str] = [
    r"\b(?:p53|BRCA[12]|TNF-?(?:alpha|α)?|IL-?\d+|interferon|interleukin"
    r"|hemoglobin|albumin|fibrinogen|collagen|elastin|keratin|actin|myosin"
    r"|insulin\s+receptor|EGFR|VEGF|HER2|BCL-?2|caspase-?\d*|kinase)\b",
]

_GENE_PATTERNS: list[str] = [
    r"\b(?:TP53|BRCA[12]|KRAS|BRAF|EGFR|ALK|ROS1|MET|NTRK[123]?"
    r"|PIK3CA|PTEN|RB1|APC|MLH1|MSH[26]|CDH1|VHL|NF[12]"
    r"|JAK[123]|STAT[1-6]|MAPK|ERK|AKT[123]?|mTOR|NOTCH[1-4]"
    r"|WNT|CTNNB1|MYC|FOS|JUN|SRC|ABL[12]?|BCR)\b",
]

_NER_SINGLETON: "BiomedicalNER | None" = None


def get_biomedical_ner(lazy: bool = False) -> "BiomedicalNER":
    """Process-wide cached BiomedicalNER.

    The SciSpacy models take ~10-15 s to load. Pipeline components
    (HCADRetriever, SemanticDecoupler, pre-warm helpers) each used to
    construct their own instance, so model loading repeated mid-benchmark
    (observed as 40-54 s retrieval spikes on the first evaluated
    questions). All components now share one instance.
    """
    global _NER_SINGLETON
    if _NER_SINGLETON is None:
        _NER_SINGLETON = BiomedicalNER(lazy=lazy)
    elif not lazy and not _NER_SINGLETON._models_loaded:
        _NER_SINGLETON._load_models()
    return _NER_SINGLETON


class SimpleNER:
    """Lightweight regex-based NER fallback.

    Uses curated word lists for common drug, disease, protein, and gene
    names.  Suitable when SciSpacy is not installed.
    """

    def __init__(self) -> None:
        # Audit item 20: Protein/Gene patterns stay case-sensitive; only the
        # curated drug/disease lists are case-insensitive.
        self._patterns: dict[str, list[re.Pattern]] = {
            "Drug": [re.compile(p, re.IGNORECASE) for p in _DRUG_PATTERNS],
            "Disease": [re.compile(p, re.IGNORECASE) for p in _DISEASE_PATTERNS],
            "Protein": [re.compile(p) for p in _PROTEIN_PATTERNS],
            "Gene": [re.compile(p) for p in _GENE_PATTERNS],
        }
        logger.info("SimpleNER initialised with regex word lists.")

    def extract_entities(self, text: str) -> EntityDict:
        """Extract entities using regex pattern matching.

        Returns the same dict structure as :class:`BiomedicalNER`.
        """
        entities = _empty_entities()
        all_spans: list[dict[str, Any]] = []

        for label, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    all_spans.append({
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "label": label,
                    })

        # Deduplicate
        all_spans = _deduplicate_spans(all_spans)

        for ent in all_spans:
            label = ent["label"]
            if label in entities:
                entities[label].append(ent)

        logger.debug(
            "SimpleNER extracted entities: %s",
            {k: len(v) for k, v in entities.items()},
        )
        return entities
