from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch

from src.config import (
    ANISO_ALPHA,
    ANISO_BETA,
    ANISO_GAMMA,
    EMBEDDING_DIMENSION,
    SCORE_WEIGHT_ENTITY,
    SCORE_WEIGHT_MODIFIER,
    SCORE_WEIGHT_RELATION,
)

logger = logging.getLogger(__name__)

_DIM_SPLIT = EMBEDDING_DIMENSION // 3  # 256 for d=768
_DIM_REMAINDER = EMBEDDING_DIMENSION - 2 * _DIM_SPLIT  # handles non-divisible

class AnisotropicTransformer:
    """Computes topology-aware anisotropic weight matrices and structural trace similarity.

    Parameters
    ----------
    alpha : float
        Weight for normalised node degree component.
    beta : float
        Weight for average edge confidence component.
    gamma : float
        Weight for inverse document frequency component.
    dim : int
        Embedding dimension (default 768).
    """

    def __init__(
        self,
        alpha: float = ANISO_ALPHA,
        beta: float = ANISO_BETA,
        gamma: float = ANISO_GAMMA,
        dim: int = EMBEDDING_DIMENSION,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dim = dim
        self._split = dim // 3
        self._remainder = dim - 2 * self._split
        self._max_degree_cache: dict[int, int] = {}  # graph id → max degree

    # Weight computation from graph topology
    def graph_block_stats(self, graph: Any, node_name: str) -> tuple[float, float, float]:
        """Raw (degree, confidence, idf) block stats for a node.

        Public accessor so the learned dynamic tensor can assemble its
        feature vector from exactly the statistics the static heuristic
        consumes - no privileged information.
        """
        return (
            self._compute_degree(graph, node_name),
            self._compute_confidence(graph, node_name),
            self._compute_idf(node_name, None, graph),
        )

    def compute_weights(
        self,
        graph: Any,
        node_name: str,
        entity_counts: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Compute anisotropic weight vector from graph properties.

        Parameters
        ----------
        graph : networkx.MultiDiGraph (or compatible)
            The knowledge graph.
        node_name : str
            Name of the node for which to compute weights.
        entity_counts : dict[str, int] or None
            Document-frequency counts for IDF computation.
            Keys are entity names, values are occurrence counts.
            If None, a uniform frequency of 1.0 is assumed.

        Returns
        -------
        np.ndarray
            Shape ``(d,)`` - normalised weight vector with Σλ_i = 1.
        """
        # ---- Degree component ----
        degree = self._compute_degree(graph, node_name)

        # ---- Confidence component ----
        confidence = self._compute_confidence(graph, node_name)

        # ---- Inverse frequency component ----
        idf = self._compute_idf(node_name, entity_counts, graph)

        # normalize each block to UNIT MASS first, then
        # apply alpha/beta/gamma. Previously the unbounded log-IDF term
        # (~11.07 at PrimeKG scale) captured 75-86% of the mass after the
        # global normalization, destroying the intended 0.4/0.4/0.2 partition
        # (verified numerically).
        eps = 1e-12
        blocks = np.array([
            max(self.alpha * degree, eps),
            max(self.beta * confidence, eps),
            max(self.gamma * idf, eps),
        ], dtype=np.float64)
        blocks /= blocks.sum()

        weights = np.ones(self.dim, dtype=np.float64)
        weights[:self._split] = blocks[0] / self._split
        weights[self._split : 2 * self._split] = blocks[1] / self._split
        weights[2 * self._split :] = blocks[2] / (self.dim - 2 * self._split)

        # Final safety normalization so Σλ_i = 1
        total = np.sum(weights)
        if total > 0:
            weights /= total
        else:
            weights = np.ones(self.dim, dtype=np.float64) / self.dim

        logger.debug(
            "Anisotropic weights for '%s': degree=%.4f, conf=%.4f, idf=%.4f, "
            "sum=%.6f",
            node_name, degree, confidence, idf, np.sum(weights),
        )
        return weights.astype(np.float32)

    def compute_default_weights(self) -> np.ndarray:
        """Return uniform weights when no graph is available.

        Returns
        -------
        np.ndarray
            Shape ``(d,)`` with Σλ_i = 1 (uniform).
        """
        return np.ones(self.dim, dtype=np.float32) / self.dim

    # Node resolution helper
    @staticmethod
    def _resolve_node(graph: Any, node_name: Any) -> tuple[Any, Any]:
        """Extract NetworkX graph and resolve node key (name -> id mapping if needed).

        Audit fix: `getattr(graph, "graph", graph)` is wrong for RAW NetworkX
        graphs - they expose a `.graph` *attribute dict* `{}`, so the resolver
        returned an empty dict, failed every lookup, and silently zeroed the
        degree/confidence terms. Only unwrap wrapper objects whose `.graph` is
        itself a graph (has `__contains__` and `degree`).
        """
        if graph is None or node_name is None:
            return None, node_name
        cand = getattr(graph, "graph", None)
        if cand is not None and not isinstance(cand, dict) and hasattr(cand, "__contains__"):
            graph_nx = cand
        else:
            graph_nx = graph
        if not hasattr(graph_nx, "__contains__"):
            return None, node_name

        if node_name in graph_nx:
            return graph_nx, node_name

        # Try name_to_node from wrapper or graph
        name_to_node = getattr(graph, "name_to_node", None) or getattr(graph_nx, "name_to_node", None)
        if name_to_node and isinstance(node_name, str):
            if node_name in name_to_node:
                return graph_nx, name_to_node[node_name]
            node_lower = node_name.lower()
            if node_lower in name_to_node:
                return graph_nx, name_to_node[node_lower]
            # Try case-insensitive scan in name_to_node keys
            for k, v in name_to_node.items():
                if k.lower() == node_lower:
                    return graph_nx, v

        return graph_nx, node_name

    # Graph property helpers
    def _compute_degree(self, graph: Any, node_name: str) -> float:
        """Normalised degree: degree(v) / max_degree(G).

        Caches ``max_degree(G)`` per graph to avoid O(N) scan on every call.
        """
        try:
            graph_nx, node_id = self._resolve_node(graph, node_name)
            if graph_nx is None or node_id not in graph_nx:
                return 0.0
            deg = graph_nx.degree(node_id)
            if graph_nx.number_of_nodes() == 0:
                return 0.0
            # key the max-degree cache by a graph signature
            # (nodes/edges), not id() - id reuse after GC could poison it.
            sig = (graph_nx.number_of_nodes(), graph_nx.number_of_edges())
            graph_id = id(graph_nx)
            cached = self._max_degree_cache.get(graph_id)
            if cached is None or cached[0] != sig:
                cached = (sig, max(d for _, d in graph_nx.degree()))
                self._max_degree_cache[graph_id] = cached
            max_deg = cached[1]
            return deg / max_deg if max_deg > 0 else 0.0
        except Exception:
            return 0.0

    def _compute_confidence(self, graph: Any, node_name: str) -> float:
        """Average edge weight (confidence) for the node's edges."""
        try:
            graph_nx, node_id = self._resolve_node(graph, node_name)
            if graph_nx is None or node_id not in graph_nx:
                return 1.0
            # consider BOTH directions, since edges(node)
            # on a MultiDiGraph yields out-edges only.
            edges = list(graph_nx.edges(node_id, data=True))
            try:
                import networkx as nx
                if isinstance(graph_nx, (nx.DiGraph, nx.MultiDiGraph)):
                    edges += [
                        (u, v, {"relation": d.get("relation"), "confidence": d.get("confidence"), "weight": d.get("weight")})
                        for u, v, k, d in graph_nx.in_edges(node_id, data=True, keys=True)
                    ]
            except Exception:
                pass
            if not edges:
                return 0.5  # neutral prior for OOV nodes (was a silently high 1.0)
            total_weight = sum(
                edata.get("weight", edata.get("confidence", 1.0))
                for _, _, edata in edges
            )
            return float(np.clip(total_weight / len(edges), 0.0, 1.0))
        except Exception:
            return 1.0

    def _compute_idf(
        self,
        node_name: str,
        entity_counts: dict[str, int] | None,
        graph: Any,
    ) -> float:
        """IDF-style weighting: rarer entities get higher weights.

        IDF = log(N / (1 + df))   where N = total entities, df = count
        """
        try:
            graph_nx = getattr(graph, "graph", graph)
            if entity_counts is None:
                N = max(graph_nx.number_of_nodes(), 1) if graph_nx is not None and hasattr(graph_nx, "number_of_nodes") else 1000
                df = 1  # assume each entity seen once
            else:
                N = max(sum(entity_counts.values()), 1)
                df = entity_counts.get(node_name, entity_counts.get(str(node_name).lower(), 1))
            raw = max(0.0, math.log(N / (1 + df)))
            # squash to (0,1) so IDF shares the scale of
            # degree/confidence before block mixing.
            return raw / (1.0 + raw)
        except Exception:
            return 0.5

    # W_A property
    def W_A(self, weights: np.ndarray) -> np.ndarray:
        """Build the full diagonal weight matrix.

        Parameters
        ----------
        weights : np.ndarray
            Shape ``(d,)`` - the weight vector from :meth:`compute_weights`.

        Returns
        -------
        np.ndarray
            Shape ``(d, d)`` - diagonal matrix ``diag(weights)``.
        """
        return np.diag(weights)

    # Transform
    @staticmethod
    def transform(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply anisotropic transformation.

        Since W_A is diagonal, ``X @ diag(w)`` simplifies to
        element-wise multiplication ``X * w``.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(3, d)`` (HCAD matrix) or ``(d,)`` (single vector).
        weights : np.ndarray
            Shape ``(d,)`` - anisotropic weights.

        Returns
        -------
        np.ndarray
            Same shape as *X*, with anisotropic weighting applied.
        """
        return X * weights

    # Structural similarity
    @staticmethod
    def structural_similarity(
        Q: np.ndarray | torch.Tensor,
        C: np.ndarray | torch.Tensor,
        weights: np.ndarray | torch.Tensor,
    ) -> float:
        """Compute Score = Trace(Q × diag(weights) × Cᵀ) using PyTorch Tensors.

        Calculated via PyTorch matrix linear algebra:
            Score = torch.trace(Q @ torch.diag(w) @ C.T)
        """
        # Tr(Q diag(w) C^T) == sum(Q * w * C) elementwise.
        # The previous implementation materialized a 768x768 diagonal matrix
        # and round-tripped tiny tensors through CUDA per call.
        Q_np = Q.cpu().numpy() if isinstance(Q, torch.Tensor) else np.asarray(Q)
        C_np = C.cpu().numpy() if isinstance(C, torch.Tensor) else np.asarray(C)
        w_np = weights.cpu().numpy() if isinstance(weights, torch.Tensor) else np.asarray(weights)
        return float(np.sum(Q_np * w_np * C_np))

    # Component-wise similarities + Φ_Final
    @staticmethod
    def compute_component_similarities(
        Q: np.ndarray,
        C: np.ndarray,
        weights: np.ndarray,
        alpha: float = SCORE_WEIGHT_ENTITY,
        beta: float = SCORE_WEIGHT_RELATION,
        gamma: float = SCORE_WEIGHT_MODIFIER,
        dim: int = EMBEDDING_DIMENSION,
    ) -> dict[str, float]:
        """Per-component weighted cosine similarity and final composite score.

        Computes:
            S_X = (Q_X · w · C_X) / (‖Q_X·√w‖ × ‖C_X·√w‖)  for X ∈ {E, R, M}
            Φ_Final = α·S_E + β·S_R + γ·S_M

        Since α + β + γ = 1.0, Φ_Final is bounded to [-1, 1].

        Parameters
        ----------
        Q : np.ndarray  - shape (3, d), query HCAD matrix.
        C : np.ndarray  - shape (3, d), candidate HCAD matrix.
        weights : np.ndarray - shape (d,), anisotropic weights.
        alpha, beta, gamma : float - score weight coefficients.
        dim : int - embedding dimension (retained for API compatibility).

        Returns
        -------
        dict[str, float]
            Keys: ``"entity"``, ``"relation"``, ``"modifier"``,
            ``"final"`` (Φ_Final).
        """
        def _weighted_cosine(q: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
            """Weighted cosine similarity: (q·w·c) / (‖q·√w‖ × ‖c·√w‖)."""
            num = np.sum(q * w * c)
            denom = np.sqrt(np.sum(w * q**2)) * np.sqrt(np.sum(w * c**2))
            return float(num / (denom + 1e-8))

        S_E = _weighted_cosine(Q[0], C[0], weights)
        S_R = _weighted_cosine(Q[1], C[1], weights)
        S_M = _weighted_cosine(Q[2], C[2], weights)

        # alpha + beta + gamma = 1.0, so phi_final ∈ [-1, 1]
        phi_final = alpha * S_E + beta * S_R + gamma * S_M

        return {
            "entity": S_E,
            "relation": S_R,
            "modifier": S_M,
            "final": phi_final,
        }

    @staticmethod
    def compute_channel_similarities(
        Q: np.ndarray,
        C_batch: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Per-channel weighted cosine similarities S_k(Q, C_i).

        Returns
        -------
        np.ndarray - shape (N, 3): S[:, k] is the weighted cosine between the
        query's channel k and every candidate's channel k. Phi_Final is a
        channel-weighted combination of these columns, which makes listwise
        training of the channel weights (alpha, beta, gamma) differentiable:
        the columns do not depend on the channel weights being learned.
        """
        if len(C_batch) == 0:
            return np.empty((0, 3), dtype=np.float32)

        Q = np.asarray(Q)
        C_batch = np.asarray(C_batch)
        out = np.zeros((len(C_batch), 3), dtype=np.float32)
        for c_idx in range(3):
            q = Q[c_idx]
            c_cand = C_batch[:, c_idx, :]
            q_w = q * weights
            num = np.dot(c_cand, q_w)
            denom_q = np.sqrt(np.sum(weights * q**2))
            denom_c = np.sqrt(np.sum(weights * (c_cand**2), axis=1))
            out[:, c_idx] = num / (denom_q * denom_c + 1e-8)
        return out

    @staticmethod
    def compute_batch_component_similarities(
        Q: np.ndarray,
        C_batch: np.ndarray,
        weights: np.ndarray,
        alpha: float = SCORE_WEIGHT_ENTITY,
        beta: float = SCORE_WEIGHT_RELATION,
        gamma: float = SCORE_WEIGHT_MODIFIER,
    ) -> np.ndarray:
        """Vectorized computation of Φ_Final across a batch of candidate HCAD matrices.

        Parameters
        ----------
        Q : np.ndarray - shape (3, d), query HCAD matrix.
        C_batch : np.ndarray - shape (N, 3, d), batch of N candidate HCAD matrices.
        weights : np.ndarray - shape (d,), anisotropic weights.
        alpha, beta, gamma : float - component weighting coefficients.

        Returns
        -------
        np.ndarray - shape (N,), Φ_Final scores for all N candidates.
        """
        if len(C_batch) == 0:
            return np.empty((0,), dtype=np.float32)

        S = AnisotropicTransformer.compute_channel_similarities(Q, C_batch, weights)
        coeffs = np.array([alpha, beta, gamma], dtype=np.float32)
        return S @ coeffs

    @staticmethod
    def compute_masked_similarities(
        Q: np.ndarray,
        C_batch: np.ndarray,
        weights: np.ndarray,
        query_presence: np.ndarray,
        candidate_presence: np.ndarray,
        alpha: float = SCORE_WEIGHT_ENTITY,
        beta: float = SCORE_WEIGHT_RELATION,
        gamma: float = SCORE_WEIGHT_MODIFIER,
        missing_prior: float = 0.20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query-conditioned channel-presence masking (Deliverable-B).

        Channel weights are renormalized over the channels the QUERY
        actively uses:

            omega_k = w_k * p_k^Q / sum_j w_j * p_j^Q

        A candidate that is missing a query-active channel (p_k^Q = 1,
        p_k^C = 0) receives a calibrated prior similarity S_k^missing
        (default 0.20) for that channel instead of the meaningless
        neutral-anchor cosine. This is deliberately NOT the mutual product
        p_k^Q * p_k^C: that variant disabled the modifier channel for every
        neutrally-phrased query and let unannotated prose outscore
        structured triples.

        Parameters
        ----------
        Q : np.ndarray - shape (3, d), query HCAD matrix.
        C_batch : np.ndarray - shape (N, 3, d), candidate HCAD matrices.
        weights : np.ndarray - shape (d,), anisotropic weights.
        query_presence : np.ndarray - shape (3,), bool; channel presence of the query.
        candidate_presence : np.ndarray - shape (N, 3), bool; channel presence per candidate.
        alpha, beta, gamma : float - base channel weights w_k.
        missing_prior : float - S_k^missing in [0.15, 0.25].

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (phi (N,), S_masked (N, 3)).
        """
        if len(C_batch) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        S = AnisotropicTransformer.compute_channel_similarities(Q, C_batch, weights)

        p_Q = np.asarray(query_presence, dtype=bool).ravel()
        p_C = np.asarray(candidate_presence, dtype=bool).reshape(len(C_batch), 3)
        w = np.array([alpha, beta, gamma], dtype=np.float64)

        active = w * p_Q.astype(np.float64)
        denom = float(active.sum())
        if denom <= 0.0:
            # Degenerate guard: a query whose channels all read absent must
            # not collapse the denominator to 0 (total-retrieval-abstention
            # bug). Fall back to the entity channel at full weight - the
            # query text itself is always real content.
            omega = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            omega = active / denom

        S_masked = np.asarray(S, dtype=np.float64).copy()
        missing = p_Q[None, :] & ~p_C
        S_masked[missing] = missing_prior

        phi = S_masked @ omega
        return phi.astype(np.float32), S_masked.astype(np.float32)
