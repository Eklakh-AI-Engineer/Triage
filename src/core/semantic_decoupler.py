from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import torch

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    GS_EPS,
    GS_RESIDUAL_TOL,
    HCAD_MATRIX_SHAPE,
    NEUTRAL_MODIFIER_TEXT,
    NEUTRAL_RELATION_TEXT,
    NUM_SEMANTIC_COMPONENTS,
    ORTHO_STRENGTH,
)

logger = logging.getLogger(__name__)

# Type alias for entity dicts
EntityDict = dict[str, list[dict[str, Any]]]

# Relation keyword lists used to extract the relation span from text.
# These overlap with RelationExtractor but are kept here to avoid a
# circular import at module level.
_RELATION_KEYWORDS: list[str] = [
    # Sorted longest-first so greedy matching works correctly.
    "is used to treat", "used in the treatment of", "provides relief for",
    "is effective against", "is indicated for", "is contraindicated in",
    "should not be used with", "must not be given", "is incompatible with",
    "not recommended for", "has interaction with", "may interact with",
    "co-administered with", "has affinity for", "selectively targets",
    "is a substrate of", "is an inhibitor of", "increases expression of",
    "decreases expression of", "elevates levels of", "reduces levels of",
    "suppresses expression of", "increases activity of",
    "decreases activity of", "induces expression of",
    "associated with", "correlated with", "implicated in",
    "linked to", "related to", "observed in",
    "treats", "causes", "leads to", "results in", "induces", "triggers",
    "precipitates", "provokes", "contraindicated", "interacts with",
    "combined with", "targets", "binds to", "acts on",
    "modulates", "inhibits", "blocks", "suppresses", "antagonises",
    "antagonizes", "attenuates", "represses",
    "activates", "stimulates", "enhances", "promotes",
    "upregulates", "up-regulates", "downregulates", "down-regulates",
    "potentiates", "agonises", "agonizes",
    "biomarker for", "predictive of", "diagnostic indicator of",
    "approved for", "prescribed for", "recommended for",
    "manages", "alleviates", "ameliorates",
]

_MODIFIER_KEYWORDS: list[str] = [
    "severe", "mild", "moderate", "acute", "chronic", "advanced",
    "early-stage", "late-stage", "refractory", "resistant", "progressive",
    "terminal", "critical", "benign", "malignant", "metastatic",
    "not", "no", "never", "without", "avoid", "contraindicated",
    "should not", "must not", "do not", "doesn't", "does not",
    "isn't", "is not", "cannot", "can't",
    "during", "after", "before", "following", "prior to", "upon",
    "within", "initially", "subsequently", "eventually",
]

class SemanticDecoupler:
    """Decouples biomedical text into Entity, Relation, and Modifier embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        lazy: bool = False,
    ) -> None:
        self._model_name = model_name or EMBEDDING_MODEL_NAME
        self._model = None
        self._dim = EMBEDDING_DIMENSION

        # Lazy-imported extractors (only instantiated when needed)
        self._ner = None
        self._rex = None

        if not lazy:
            self._load_model()

    # Model loading
    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            import torch
            _device = "cpu"
            if torch.cuda.is_available():
                try:
                    free_mem, _ = torch.cuda.mem_get_info()
                    if free_mem > 1200 * 1024 * 1024:  # require 1.2 GB free VRAM
                        _device = "cuda"
                except Exception:
                    pass
            logger.info("Loading embedding model: %s on %s", self._model_name, _device)
            try:
                self._model = SentenceTransformer(self._model_name, device=_device, local_files_only=True)
            except Exception:
                try:
                    self._model = SentenceTransformer(self._model_name, device=_device)
                except Exception:
                    logger.warning("Failed to load on %s, falling back to CPU", _device)
                    self._model = SentenceTransformer(self._model_name, device="cpu")
            # Verify dimension
            test_vec = self._model.encode("test", convert_to_numpy=True)
            actual_dim = test_vec.shape[0]
            if actual_dim != self._dim:
                logger.warning(
                    "Model dimension %d ≠ config EMBEDDING_DIMENSION %d. "
                    "Updating internal dimension.",
                    actual_dim,
                    self._dim,
                )
                self._dim = actual_dim
            logger.info(
                "Embedding model loaded (dim=%d).", self._dim
            )
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise

    def _get_ner(self):
        """Lazily import and instantiate NER extractor."""
        if self._ner is None:
            from src.core.ner_extractor import get_biomedical_ner
            self._ner = get_biomedical_ner(lazy=True)
        return self._ner

    def _get_rex(self):
        """Lazily import and instantiate RelationExtractor."""
        if self._rex is None:
            from src.core.relation_extractor import RelationExtractor
            self._rex = RelationExtractor(use_llm_fallback=False)
        return self._rex

    # Text component extraction
    @staticmethod
    def _extract_entity_text(
        text: str, entities: EntityDict | None
    ) -> str:
        """Extract the primary entity mention from *text*.

        If entities are provided, concatenate entity surface forms.
        Otherwise fall back to the full text (the embedding model will
        capture the salient entity).
        """
        if entities:
            entity_texts: list[str] = []
            for etype, elist in entities.items():
                for ent in elist:
                    if isinstance(ent, dict):
                        entity_text = ent.get("text", "")
                        if entity_text and isinstance(entity_text, str):
                            entity_texts.append(entity_text)
            if entity_texts:
                return " ".join(entity_texts)
        # Fallback: use the full text
        return text or ""

    @staticmethod
    def _extract_relation_text(
        text: str, relations: list[dict[str, Any]] | None
    ) -> str:
        """Extract the predicate / action verb phrase.

        If explicit relations are given, use their relation type names.
        Otherwise scan the text for known relation keywords.
        """
        if relations:
            rel_parts: list[str] = []
            for rel in relations:
                if isinstance(rel, dict):
                    rel_text = rel.get("relation", "")
                    if rel_text and isinstance(rel_text, str):
                        rel_parts.append(rel_text.replace("_", " ").lower())
            if rel_parts:
                return " ".join(rel_parts)

        # Scan for relation keywords in the text
        text_lower = (text or "").lower()
        found: list[str] = []
        for kw in _RELATION_KEYWORDS:
            if kw in text_lower:
                found.append(kw)
        if found:
            return " ".join(found)

        # a fixed neutral anchor - NEVER the full text.
        # Returning the full text made R == E and collapsed the HCAD matrix
        # to rank 2 (verified cos(E, R_orth) = 1.0).
        return NEUTRAL_RELATION_TEXT

    @staticmethod
    def _extract_modifier_text(
        text: str,
        entities: EntityDict | None,
        relations: list[dict[str, Any]] | None,
    ) -> str:
        """Extract everything that is NOT entity or relation.

        This includes severity qualifiers, negation cues, and temporal
        context.
        """
        text = text or ""

        # If relations carry a modifier field, use it
        if relations:
            modifiers = [
                rel.get("modifier", "")
                for rel in relations
                if isinstance(rel, dict) and rel.get("modifier") and rel.get("modifier") != "neutral"
            ]
            if modifiers:
                return " ".join(modifiers)

        # Scan for modifier keywords
        text_lower = text.lower()
        found: list[str] = []
        for kw in _MODIFIER_KEYWORDS:
            if kw in text_lower:
                found.append(kw)
        if found:
            return " ".join(found)

        # Fallback: residual text after stripping entity spans and relation keywords
        residual = text
        if entities:
            for _etype, elist in entities.items():
                for ent in elist:
                    if isinstance(ent, dict):
                        entity_text = ent.get("text", "")
                        if entity_text:
                            residual = residual.replace(entity_text, "")
        # Also strip matched relation keywords so modifier is clean
        for kw in _RELATION_KEYWORDS:
            if kw in residual.lower():
                residual = re.sub(re.escape(kw), "", residual, flags=re.IGNORECASE)
        residual = re.sub(r"\s+", " ", residual).strip()
        # the residual frequently re-embeds the entity
        # text (span stripping fails on joined multi-entity strings), which
        # collides the modifier channel with the entity channel.
        return residual if residual else NEUTRAL_MODIFIER_TEXT

    # Encoding & PyTorch Tensor conversion
    def _encode(self, text: str) -> torch.Tensor:
        """Encode a text string into a (d,) PyTorch Tensor."""
        self._load_model()
        with torch.no_grad():
            try:
                vec_np = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as e:
                logger.warning("Encoding failed on device (%s), falling back to CPU", e)
                self._model = self._model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                vec_np = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return torch.from_numpy(vec_np.astype(np.float32))

    # PyTorch Gram-Schmidt orthogonalization
    @staticmethod
    def _orthogonalize(
        E: torch.Tensor, R: torch.Tensor, M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply soft Gram-Schmidt to reduce semantic interference while preserving relation semantics.

        Uses a 0.7/0.3 blend between the orthogonalised and original
        directions so that meaningful semantic overlap (e.g. entity-
        relation coupling in biomedical text) is partially preserved.

        Parameters
        ----------
        E, R, M : torch.Tensor
            Raw embedding tensors of shape (d,).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Soft-orthonormalised E, R, M PyTorch Tensors.
        """
        eps = GS_EPS
        eta = ORTHO_STRENGTH

        # E stays as-is (first basis vector)
        E_norm = torch.linalg.norm(E)
        E_orth = E / (E_norm + eps)

        R_orth = SemanticDecoupler._soft_component(R, [E_orth], eta, eps)
        M_orth = SemanticDecoupler._soft_component(M, [E_orth, R_orth], eta, eps)

        return E_orth, R_orth, M_orth

    @staticmethod
    def _absent_anchor(basis: list[torch.Tensor], d: int) -> torch.Tensor:
        """Deterministic unit vector orthogonal to all `basis` vectors.

        Audit fix (items 1/2): a degenerate component (residual ~ 0) must not
        be silently restored to its original overlapping direction - that
        collapses the HCAD matrix to rank 2. It is replaced by an orthogonal
        anchor so the 3-row basis stays full-rank and the *absence* is encoded
        as a fixed, query-independent direction.
        """
        best, best_overlap = None, float("inf")
        for i in range(8):
            cand = torch.zeros(d, dtype=torch.float32)
            cand[i * (d // 8) % d] = 1.0
            for b in basis:
                cand = cand - torch.dot(cand, b) * b
            norm = torch.linalg.norm(cand)
            if norm > 1e-6:
                cand = cand / norm
                overlap = max(abs(float(torch.dot(cand, b))) for b in basis) if basis else 0.0
                if overlap < best_overlap:
                    best, best_overlap = cand, overlap
                    if overlap < 0.05:
                        break
        if best is None:
            best = torch.zeros(d, dtype=torch.float32)
            best[-1] = 1.0
        return best

    @staticmethod
    def _soft_component(
        v: torch.Tensor,
        basis: list[torch.Tensor],
        eta: float,
        eps: float,
    ) -> torch.Tensor:
        """Soft Gram-Schmidt for one component with presence handling.

        If the residual against `basis` is below GS_RESIDUAL_TOL, the
        component carries no independent signal: substitute an orthogonal
        anchor (presence = False upstream) instead of the original direction.
        Otherwise blend eta * residual + (1 - eta) * original and renormalize.
        """
        d = v.shape[0]
        v_unit = v / (torch.linalg.norm(v) + eps)
        residual = v_unit.clone()
        for b in basis:
            residual = residual - torch.dot(residual, b) * b
        # one re-orthogonalization pass (modified GS) to bound leakage
        for b in basis:
            residual = residual - torch.dot(residual, b) * b
        residual_norm = torch.linalg.norm(residual)
        if float(residual_norm) < GS_RESIDUAL_TOL:
            return SemanticDecoupler._absent_anchor(basis, d)
        residual_unit = residual / (residual_norm + eps)
        blended = eta * residual_unit + (1.0 - eta) * v_unit
        return blended / (torch.linalg.norm(blended) + eps)

    def decouple(
        self,
        text: str,
        entities: EntityDict | None = None,
        relations: list[dict[str, Any]] | None = None,
        as_numpy: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """Decouple text into the HCAD matrix [E; R; M]."""
        if entities is None:
            entities = self._get_ner().extract_entities(text)
        if relations is None:
            relations = self._get_rex().extract(text, entities)

        entity_text = self._extract_entity_text(text, entities)
        relation_text = self._extract_relation_text(text, relations)
        modifier_text = self._extract_modifier_text(text, entities, relations)

        logger.debug(
            "Decoupled components - E: '%s', R: '%s', M: '%s'",
            entity_text[:80],
            relation_text[:80],
            modifier_text[:80],
        )

        E = self._encode(entity_text)
        R = self._encode(relation_text)
        M = self._encode(modifier_text)

        E_orth, R_orth, M_orth = self._orthogonalize(E, R, M)

        X_HCAD_tensor = torch.stack([E_orth, R_orth, M_orth])  # (3, d)

        assert X_HCAD_tensor.shape == (NUM_SEMANTIC_COMPONENTS, self._dim), (
            f"Expected shape {HCAD_MATRIX_SHAPE}, got {X_HCAD_tensor.shape}"
        )

        if as_numpy:
            return X_HCAD_tensor.detach().cpu().numpy()
        return X_HCAD_tensor

    def decouple_batch(
        self,
        texts: list[str],
        entities_list: list[EntityDict | None] | None = None,
        relations_list: list[list[dict[str, Any]] | None] | None = None,
        as_numpy: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """Batch decouple multiple texts into shape (N, 3, d) using vectorized inference.

        Parameters
        ----------
        texts : list[str]
            List of N texts.
        entities_list : list or None
            Parallel list of entity dicts (one per text).
        relations_list : list or None
            Parallel list of relation lists (one per text).
        as_numpy : bool
            Whether to return np.ndarray (default True) or torch.Tensor.

        Returns
        -------
        np.ndarray | torch.Tensor
            Shape ``(N, 3, d)`` - batch of HCAD matrices.
        """
        n = len(texts)
        if n == 0:
            # honor the tensor contract for empty batches.
            import torch as _torch
            empty = _torch.empty((0, NUM_SEMANTIC_COMPONENTS, self._dim), dtype=_torch.float32)
            return empty.numpy() if as_numpy else empty

        # validate parallel inputs instead of silently
        # padding/truncating.
        if entities_list is not None and len(entities_list) != n:
            raise ValueError(f"entities_list length {len(entities_list)} != {len(texts)} texts")
        if relations_list is not None and len(relations_list) != n:
            raise ValueError(f"relations_list length {len(relations_list)} != {len(texts)} texts")
        if entities_list is None:
            entities_list = [None] * n
        if relations_list is None:
            relations_list = [None] * n

        entity_texts: list[str] = []
        relation_texts: list[str] = []
        modifier_texts: list[str] = []

        for i, text in enumerate(texts):
            ents = entities_list[i] if i < len(entities_list) else None
            rels = relations_list[i] if i < len(relations_list) else None

            # If not provided, run NER / relation extraction
            if ents is None:
                ents = self._get_ner().extract_entities(text)
            if rels is None:
                rels = self._get_rex().extract(text, ents)

            entity_texts.append(self._extract_entity_text(text, ents))
            relation_texts.append(self._extract_relation_text(text, rels))
            modifier_texts.append(self._extract_modifier_text(text, ents, rels))

        # Vectorized batch encoding of all 3*N strings together
        all_texts = entity_texts + relation_texts + modifier_texts
        self._load_model()

        with torch.no_grad():
            try:
                all_vecs_np = self._model.encode(
                    all_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception as e:
                logger.warning("CUDA OOM during batch encode (%s). Falling back embedding model to CPU.", e)
                self._model = self._model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                all_vecs_np = self._model.encode(
                    all_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            all_vecs = torch.from_numpy(all_vecs_np.astype(np.float32))

            E = all_vecs[:n]
            R = all_vecs[n:2*n]
            M = all_vecs[2*n:]

            eps = GS_EPS
            eta = ORTHO_STRENGTH

            # Batched soft Gram-Schmidt with presence handling 
            E_orth = E / (torch.linalg.norm(E, dim=1, keepdim=True) + eps)

            R_unit = R / (torch.linalg.norm(R, dim=1, keepdim=True) + eps)
            R_proj = R_unit - torch.sum(R_unit * E_orth, dim=1, keepdim=True) * E_orth
            R_proj = R_proj - torch.sum(R_proj * E_orth, dim=1, keepdim=True) * E_orth  # re-orth
            R_res = torch.linalg.norm(R_proj, dim=1, keepdim=True)
            R_degenerate = R_res < GS_RESIDUAL_TOL
            R_blend = eta * (R_proj / (R_res + eps)) + (1.0 - eta) * R_unit
            R_orth = R_blend / (torch.linalg.norm(R_blend, dim=1, keepdim=True) + eps)
            for i in range(n):
                if bool(R_degenerate[i]):
                    R_orth[i] = SemanticDecoupler._absent_anchor([E_orth[i]], self._dim)

            M_unit = M / (torch.linalg.norm(M, dim=1, keepdim=True) + eps)
            M_proj = M_unit - torch.sum(M_unit * E_orth, dim=1, keepdim=True) * E_orth
            M_proj = M_proj - torch.sum(M_proj * R_orth, dim=1, keepdim=True) * R_orth
            M_proj = M_proj - torch.sum(M_proj * E_orth, dim=1, keepdim=True) * E_orth  # re-orth
            M_res = torch.linalg.norm(M_proj, dim=1, keepdim=True)
            M_degenerate = M_res < GS_RESIDUAL_TOL
            M_blend = eta * (M_proj / (M_res + eps)) + (1.0 - eta) * M_unit
            M_orth = M_blend / (torch.linalg.norm(M_blend, dim=1, keepdim=True) + eps)
            for i in range(n):
                if bool(M_degenerate[i]):
                    M_orth[i] = SemanticDecoupler._absent_anchor([E_orth[i], R_orth[i]], self._dim)

            batch_tensor = torch.stack([E_orth, R_orth, M_orth], dim=1)

        logger.debug("Vectorized batch decoupled %d texts → shape %s", n, batch_tensor.shape)
        if as_numpy:
            return batch_tensor.detach().cpu().numpy()
        return batch_tensor

