import logging
from typing import Optional

import numpy as np

from src.config import (
    CHANNEL_MISSING_PRIOR,
    ENABLE_FEW_SHOT,
    FILTERING_K,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL_NAME,
    MAX_EVIDENCE_ITEMS,
    MIN_PHI_SCORE,
    SCORE_WEIGHT_ENTITY,
    SCORE_WEIGHT_MODIFIER,
    SCORE_WEIGHT_RELATION,
    map_relation_text,
    presence_masking_enabled,
)

logger = logging.getLogger(__name__)

_ENTITY_TYPE_VOCAB = ["drug", "disease", "gene", "protein", "chemical",
                       "phenotype", "pathway", "anatomy", "exposure", "other"]
_RELATION_VOCAB = ["treats", "contraindicated", "causes", "interacts",
                   "associated", "biomarker", "carries", "indication",
                   "off-label", "other"]


def _entity_type_histogram(entities: dict) -> list[float]:
    """10-dim presence histogram over NER entity categories (values in [0,1])."""
    hist = [0.0] * len(_ENTITY_TYPE_VOCAB)
    counts = []
    for key, items in (entities or {}).items():
        n = len(items) if isinstance(items, list) else 0
        if not n:
            continue
        k = str(key).lower()
        idx = next((i for i, v in enumerate(_ENTITY_TYPE_VOCAB) if v in k),
                   len(_ENTITY_TYPE_VOCAB) - 1)
        hist[idx] = max(hist[idx], 1.0)
        counts.append((idx, n))
    if counts and counts[0][1]:
        # scale by the largest per-category count, capped at 1
        top = max(n for _, n in counts)
        for idx, n in counts:
            hist[idx] = min(1.0, n / top)
    return hist


def _relation_onehot(rel_names: list[str]) -> list[float]:
    """10-dim one-hot of the query's extracted relation type."""
    vec = [0.0] * len(_RELATION_VOCAB)
    for name in rel_names:
        idx = next((i for i, v in enumerate(_RELATION_VOCAB) if v in name),
                   len(_RELATION_VOCAB) - 1)
        if any(v in name for v in _RELATION_VOCAB[:-1]):
            vec[idx] = 1.0
            return vec
    vec[-1] = 1.0
    return vec


class HCADRetriever:
    """End-to-end HCAD-RAG retrieval and generation pipeline."""

    def __init__(
        self,
        graph=None,
        collection=None,
        model_name: Optional[str] = None,
        k: float = FILTERING_K,
        fail_on_vector_error: bool = False,
    ):
        self._graph = graph
        self._collection = collection
        self.model_name = model_name
        self.k = k
        self.fail_on_vector_error = fail_on_vector_error

        self._ner: Optional[object] = None
        self._relation_extractor: Optional[object] = None
        self._decoupler: Optional[object] = None
        self._transformer: Optional[object] = None
        self._hypergraph: Optional[object] = None
        self._filter: Optional[object] = None
        self._prompt_builder: Optional[object] = None
        self._entity_linker: Optional[object] = None
        self._learned_weights: Optional[dict] = None
        self._node_lookup: Optional[dict] = None
        self._decision_gate: Optional[object] = None
        self._entity_res_cache: dict[str, Optional[str]] = {}
        # Learned query-adaptive tensor (HCAD_TENSOR_MODE=learned). The
        # override hook is used ONLY by the tensor trainer to evaluate
        # sampled weight vectors; production retrieval always predicts
        # from the tensor itself.
        self._tensor: Optional[object] = None
        self._tensor_override: Optional[tuple] = None
        self._last_tensor_inputs: Optional[tuple] = None
        self._last_channel_sims: Optional["np.ndarray"] = None
        self._last_candidate_pool: Optional[list] = None
        self._last_filtered_pool: Optional[list] = None
        self._last_post_gate_pool: Optional[list] = None

    def _handle_vector_error(self, exc: Exception) -> None:
        logger.warning("Vector store retrieval failed: %s", exc)
        if self.fail_on_vector_error:
            raise RuntimeError("Vector store retrieval failed during benchmark retrieval.") from exc

    def _get_tensor(self):
        """Load the learned dynamic tensor when HCAD_TENSOR_MODE=learned.

        Static heuristic remains the default; a missing or unreadable
        checkpoint falls back to static with a warning (never breaks
        retrieval).
        """
        import os
        if (self._tensor is None
                and os.getenv("HCAD_TENSOR_MODE", "static") == "learned"):
            try:
                from src.core.dynamic_tensor import load_tensor
                self._tensor = load_tensor()
                logger.info("Learned dynamic tensor active (models/dynamic_tensor.pt).")
            except Exception as exc:
                logger.warning("HCAD_TENSOR_MODE=learned but no usable tensor (%s); "
                               "falling back to static weights.", exc)
        return self._tensor

    def _load_components(self):
        """Initialize components on first access."""
        from src.core.ner_extractor import BiomedicalNER, SimpleNER
        from src.core.relation_extractor import RelationExtractor
        from src.core.semantic_decoupler import SemanticDecoupler
        from src.core.anisotropic_transformer import AnisotropicTransformer
        from src.core.hypergraph_builder import HypergraphBuilder
        from src.core.statistical_filter import StatisticalFilter
        from src.core.prompt_builder import PromptBuilder
        from src.core.decision_gate import DecisionGate

        logger.info("Initializing HCAD-RAG components...")

        try:
            from src.core.ner_extractor import get_biomedical_ner
            self._ner = get_biomedical_ner()
        except Exception as e:
            logger.warning(f"BiomedicalNER unavailable ({e}), using SimpleNER fallback.")
            self._ner = SimpleNER()

        self._relation_extractor = RelationExtractor()
        self._decoupler = SemanticDecoupler()
        self._transformer = AnisotropicTransformer()

        # Hypergraph is lazily constructed via self.hypergraph
        self._filter = StatisticalFilter(k=self.k)
        self._prompt_builder = PromptBuilder()
        self._decision_gate = DecisionGate(
            min_relevance_score=0.25, max_evidence_items=MAX_EVIDENCE_ITEMS
        )

        try:
            from src.core.entity_linker import SapBERTLinker
            self._entity_linker = SapBERTLinker()
            self._entity_linker.build_index(self.graph)
        except Exception as e:
            logger.debug(f"Entity linker not initialized: {e}")
            self._entity_linker = None

        # Audit/paper-parity: n-ary fact index from PrimeKG shared-tail
        # contexts (base-paper Sec. 3.2 high-order correlations, but curated).
        try:
            from src.core.fact_hypergraph import build_fact_index_from_kg
            self._fact_index = build_fact_index_from_kg(self.graph)
            logger.info("N-ary fact index: %d clinical facts.", len(self._fact_index.facts))
        except Exception as e:
            logger.debug(f"Fact index unavailable: {e}")
            self._fact_index = None

        try:
            from src.core.weight_learner import load_learned_weights
            self._learned_weights = load_learned_weights()
            if self._learned_weights and "filter_k" in self._learned_weights:
                self._filter = StatisticalFilter(k=self._learned_weights["filter_k"])
        except Exception:
            self._learned_weights = None

    @property
    def graph(self):
        if self._graph is None:
            from src.data.build_knowledge_graph import load_knowledge_graph
            self._graph = load_knowledge_graph()
        return self._graph

    @property
    def hypergraph(self):
        if self._hypergraph is None:
            from src.core.hypergraph_builder import HypergraphBuilder
            raw_graph = getattr(self.graph, "graph", self.graph)
            if raw_graph is not None:
                if hasattr(self.graph, "name_to_node") and not hasattr(raw_graph, "name_to_node"):
                    try:
                        raw_graph.name_to_node = self.graph.name_to_node
                    except Exception:
                        pass
                self._hypergraph = HypergraphBuilder(raw_graph)
        return self._hypergraph

    @property
    def gate_mode(self) -> str:
        """Active JEV gate mode: 'live-typesafe' or 'local-calibrated'.

        Exposed for benchmark provenance: the mode is recorded in every
        run_manifest and fail-loud compared against HCAD_EXPECTED_GATE_MODE
        so a stray TYPESAFE_API_KEY cannot silently switch the measurement
        chain between runs.
        """
        if self._decision_gate is None:
            return "uninitialized"
        return getattr(self._decision_gate, "mode", "unknown")

    @property
    def collection(self):
        if self._collection is None:
            from src.data.build_vector_store import load_vector_store
            self._collection = load_vector_store()
        return self._collection

    def _ensure_components(self):
        if self._ner is None:
            self._load_components()

    # Entity Resolution

    def _resolve_entity_in_graph(self, entity_name: str) -> Optional[str]:
        """Resolve an entity name to a canonical PrimeKG node name with in-memory caching."""
        if entity_name in self._entity_res_cache:
            return self._entity_res_cache[entity_name]

        # Try SapBERT first
        if self._entity_linker is not None:
            try:
                matches = self._entity_linker.link(entity_name, top_k=1, threshold=0.50)
                if matches:
                    canonical = matches[0]["canonical_name"]
                    score = matches[0]["score"]
                    logger.debug(
                        f"SapBERT linked '{entity_name}' → '{canonical}' (score={score:.3f})"
                    )
                    self._entity_res_cache[entity_name] = canonical
                    return canonical
            except Exception as e:
                logger.warning(f"SapBERT linking failed for '{entity_name}': {e}")

        # Fallback: string matching
        res = self._find_entity_in_graph(entity_name)
        # Cache the result, but ensure we don't cache None for empty strings
        if res is None and entity_name and entity_name.strip():
            # Try case-insensitive fallback
            res = self._find_entity_in_graph(entity_name.lower())
        self._entity_res_cache[entity_name] = res
        return res

    def _find_entity_in_graph(self, entity_name: str) -> Optional[str]:
        """Find the best matching node name in the graph (fast O(1) hash lookup)."""
        if not entity_name or not isinstance(entity_name, str):
            return None

        try:
            if hasattr(self.graph, "get_node_by_name"):
                try:
                    result = self.graph.get_node_by_name(entity_name)
                    if result:
                        return entity_name
                except KeyError:
                    pass

            graph_obj = getattr(self.graph, "graph", self.graph)
            if graph_obj is None:
                return None

            if entity_name in graph_obj:
                return entity_name

            # Build lazy lowercase lookup index once
            if self._node_lookup is None:
                logger.info("Building fast O(1) PrimeKG node name hash index...")
                name_to_node = getattr(self.graph, "name_to_node", {})
                if name_to_node:
                    self._node_lookup = {k.lower(): k for k in name_to_node.keys()}
                else:
                    # Fallback: extract names from node attributes
                    if graph_obj is not None and hasattr(graph_obj, "nodes"):
                        self._node_lookup = {
                            str(graph_obj.nodes[n].get("name", "")).lower(): str(graph_obj.nodes[n].get("name", ""))
                            for n in graph_obj.nodes
                            if graph_obj.nodes[n].get("name")
                        }
                if self._node_lookup is None:
                    self._node_lookup = {}
                logger.info(f"Node name hash index ready with {len(self._node_lookup)} entries.")

            entity_lower = entity_name.lower()
            if self._node_lookup and entity_lower in self._node_lookup:
                return self._node_lookup[entity_lower]

        except Exception as e:
            logger.debug(f"Entity lookup failed for '{entity_name}': {e}")

        return None

    def _get_candidate_contexts(
        self, query: str, entities: dict, n_vector: int = 30
    ) -> list[dict]:
        import torch
        from sentence_transformers import SentenceTransformer

        candidates = []
        seen_docs = set()

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            emb_model = (
                self._decoupler._model
                if hasattr(self, "_decoupler")
                and self._decoupler
                and hasattr(self._decoupler, "_model")
                and self._decoupler._model
                else SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            )
            # Chroma's collection uses cosine distance. Keep the query vector
            # on the same unit-normalized scale as the indexed vectors.
            q_emb = emb_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ).tolist()
            results = self.collection.query(
                query_embeddings=q_emb, n_results=n_vector
            )
            documents = (results.get("documents") or [[]])[0] or []
            metadatas = (results.get("metadatas") or [[]])[0] or []
            distances = (results.get("distances") or [[]])[0] or []
            for i, doc in enumerate(documents):
                if doc not in seen_docs:
                    candidates.append(
                        {
                            "document": doc,
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "score": 1.0 - distances[i] if i < len(distances) else 0.0,
                            "source": "vector",
                        }
                    )
                    seen_docs.add(doc)
        except Exception as e:
            self._handle_vector_error(e)

        if self.hypergraph is not None:
            all_entity_names = []
            for ent_list in entities.values():
                for ent in ent_list:
                    all_entity_names.append(ent["text"])

            # Map entity names to graph node IDs using name_to_node
            name_to_node = getattr(self.graph, "name_to_node", {})
            graph_nx = getattr(self.graph, "graph", self.graph)

            # Handle case where graph or its attributes are None
            if name_to_node is None:
                name_to_node = {}
            if graph_nx is None or not hasattr(graph_nx, "nodes"):
                import networkx as nx
                graph_nx = nx.MultiDiGraph()

            # rank entities by NER salience before the
            # arbitrary [:5] cap, so truncation is deterministic.
            all_entity_names = list(dict.fromkeys(all_entity_names))[:5]
            for entity_name in all_entity_names:
                matched_name = self._resolve_entity_in_graph(entity_name)
                if not matched_name:
                    continue

                node_id = name_to_node.get(matched_name, matched_name)
                if node_id is None or node_id not in graph_nx:
                    continue

                try:
                    hyperedges = self.hypergraph.build_hyperedges(node_id)
                    if hyperedges:
                        # sort by importance BEFORE capping.
                        hyperedges = sorted(hyperedges, key=lambda h: h["importance"], reverse=True)
                        max_importance = max(he["importance"] for he in hyperedges) or 1.0
                        for he in hyperedges[:5]:
                            he_nodes = he["nodes"][:10]
                            he_triples = self.hypergraph.get_triples_for_entities(he_nodes)
                            norm_importance = he["importance"] / max_importance
                            for u, relation, v in he_triples[:10]:
                                head_name = graph_nx.nodes[u].get("name", str(u)) if u in graph_nx else str(u)
                                tail_name = graph_nx.nodes[v].get("name", str(v)) if v in graph_nx else str(v)
                                doc = f"{head_name} {relation.replace('_', ' ')} {tail_name}."
                                if doc not in seen_docs:
                                    candidates.append(
                                        {
                                            "document": doc,
                                            "metadata": {
                                                "head_name": head_name,
                                                "tail_name": tail_name,
                                                "relation": relation,
                                                "display_relation": relation,
                                                "head_type": graph_nx.nodes.get(u, {}).get("node_type", "unknown"),
                                                "tail_type": graph_nx.nodes.get(v, {}).get("node_type", "unknown"),
                                            },
                                            "score": 0.5 * norm_importance,
                                            "source": "graph_hyperedge",
                                        }
                                    )
                                    seen_docs.add(doc)
                except Exception as e:
                    logger.debug(f"Hyperedge expansion failed for '{matched_name}': {e}")
                    try:
                        triples = self.hypergraph.get_triples_for_entities([node_id])
                        for u, relation, v in triples[:20]:
                            head_name = graph_nx.nodes[u].get("name", str(u)) if u in graph_nx else str(u)
                            tail_name = graph_nx.nodes[v].get("name", str(v)) if v in graph_nx else str(v)
                            doc = f"{head_name} {relation.replace('_', ' ')} {tail_name}."
                            if doc not in seen_docs:
                                candidates.append(
                                    {
                                        "document": doc,
                                        "metadata": {
                                            "head_name": head_name,
                                            "tail_name": tail_name,
                                            "relation": relation,
                                            "display_relation": relation,
                                            "head_type": graph_nx.nodes.get(u, {}).get("node_type", "unknown"),
                                            "tail_type": graph_nx.nodes.get(v, {}).get("node_type", "unknown"),
                                        },
                                        "score": 0.5,
                                        "source": "graph",
                                    }
                                )
                                seen_docs.add(doc)
                    except Exception:
                        pass

        # Paper-parity (Sec. 3.4): correlation-keyword retrieval against
        # hyperedges/facts, in addition to entity->vertex retrieval. The
        # relation keywords extracted from the query act as X_cor.
        if getattr(self, "_fact_index", None) is not None:
            rel_keywords = self._relation_extractor.extract_keywords(query) \
                if hasattr(self._relation_extractor, "extract_keywords") else []
            anchors = set()
            for ent_list in entities.values():
                for ent in ent_list:
                    if isinstance(ent, dict) and ent.get("text"):
                        anchors.add(ent["text"].lower())
            if rel_keywords or anchors:
                facts = self._fact_index.retrieve(anchors, top_k=12)
                for f in facts:
                    doc = self._fact_index.render(f)
                    if doc not in seen_docs:
                        members = sorted(set(f.roles.values()))
                        candidates.append({
                            "document": doc,
                            "metadata": {
                                "head_name": members[0] if members else "",
                                "tail_name": f.roles.get("anchor", ""),
                                "relation": f.predicate,
                                "display_relation": f.predicate,
                                "fact_arity": len(f.roles),
                                "provenance": f.source_id,
                            },
                            "score": 0.5 * f.confidence,
                            "source": "nary_fact",
                        })
                        seen_docs.add(doc)

        return candidates[:60]

    def retrieve(
        self,
        query: str,
        k: Optional[float] = None,
        n_candidates: int = 30,
    ) -> list[dict]:
        """Retrieve and rank candidate contexts using decoupled structural similarity and adaptive filtering."""
        self._ensure_components()
        k = k if k is not None else self.k
        # Reset per-query diagnostic snapshots so empty/error paths cannot leak
        # candidate evidence from the preceding query in a benchmark batch.
        self._last_candidate_pool = None
        self._last_filtered_pool = None
        self._last_post_gate_pool = None
        self._last_phi_scores = None
        self._last_query_presence = None
        self._last_candidate_presence = None

        entities = self._ner.extract_entities(query)
        relations = self._relation_extractor.extract_rules(query, entities)

        Q, query_presence = self._decoupler.decouple(
            query, entities=entities, relations=relations, return_presence=True
        )

        all_entity_names = [
            e.get("text", "") for ents in entities.values() for e in ents if isinstance(e, dict)
        ]
        # Filter out empty strings and ensure we have valid entity names
        all_entity_names = [name for name in all_entity_names if name and isinstance(name, str)]
        primary_entity = all_entity_names[0] if all_entity_names else ""

        tensor = self._get_tensor()
        lam_override = a_override = None
        if self._tensor_override is not None:
            lam_override, a_override = self._tensor_override
        if tensor is not None or lam_override is not None:
            import torch
            from src.core.dynamic_tensor import make_query_features
            deg, conf, idf = self._transformer.graph_block_stats(
                self.graph, primary_entity)
            rel_names = [str(r.get("relation", "")).lower() for r in (relations or [])
                         if isinstance(r, dict)]
            neg_flag = 1.0 if any("contra" in r or "not_" in r for r in rel_names) else 0.0
            type_hist = _entity_type_histogram(entities)
            rel_onehot = _relation_onehot(rel_names)
            feats = make_query_features(
                None,
                max(0.0, min(1.0, deg)),
                max(0.0, min(1.0, conf)),
                max(0.0, min(1.0, idf)),
                neg_flag,
                0.0,
                type_hist,
                rel_onehot,
            )
            Q_t = torch.tensor(np.asarray(Q, dtype=np.float32))
            if lam_override is not None:
                lam_t, a_t = lam_override, a_override
            else:
                lam_t, a_t = tensor(Q_t, feats)
            # Always record (Q, feats) for this query - the listwise trainer
            # keys its channel-weight adapter to these inputs even when the
            # warm-up pass ran under an explicit override.
            self._last_tensor_inputs = (Q_t.detach(), feats.detach())
            weights = lam_t.detach().cpu().numpy().astype(np.float64)
            component_a = a_t.detach().cpu().numpy()
        else:
            weights = self._transformer.compute_weights(self.graph, primary_entity)
            component_a = None

        candidates = self._get_candidate_contexts(
            query, entities, n_vector=n_candidates
        )
        if not candidates:
            return []

        # Vectorized batch decoupling of all candidates in a single tensor operation
        candidate_docs = []
        entities_list = []
        relations_list = []
        for ctx in candidates:
            ctx_doc = ctx.get("document", "")
            ctx_meta = ctx.get("metadata") or {}
            ctx_entity = ctx_meta.get("head_name", "")
            # Deliverable-B relation-vocabulary mapping: raw PrimeKG schema
            # labels ("drug protein") barely cosine-match natural query verbs
            # ("targets"); map to canonical clinical verb phrases.
            ctx_relation = map_relation_text(ctx_meta.get("relation") or "")
            ctx_tail = ctx_meta.get("tail_name", "")

            candidate_docs.append(ctx_doc)
            ctx_entities_dict = None
            ctx_relations_list = None
            if ctx_entity:
                ctx_entities_dict = {"Entity": [
                    {"text": ctx_entity, "start": 0, "end": len(ctx_entity)},
                ]}
                if ctx_tail:
                    ctx_entities_dict["Entity"].append(
                        {"text": ctx_tail, "start": len(ctx_entity)+1,
                         "end": len(ctx_entity)+1+len(ctx_tail)}
                    )
            if ctx_relation:
                ctx_relations_list = [{"relation": ctx_relation}]

            entities_list.append(ctx_entities_dict)
            relations_list.append(ctx_relations_list)

        C_batch, candidate_presence = self._decoupler.decouple_batch(
            candidate_docs,
            entities_list=entities_list,
            relations_list=relations_list,
            as_numpy=True,
            return_presence=True,
        )
        if component_a is not None:
            channel_weights = (float(component_a[0]), float(component_a[1]),
                               float(component_a[2]))
        else:
            channel_weights = (SCORE_WEIGHT_ENTITY, SCORE_WEIGHT_RELATION,
                               SCORE_WEIGHT_MODIFIER)

        if presence_masking_enabled():
            # Deliverable-B: query-conditioned channel-presence masking.
            phi_scores, masked_sims = self._transformer.compute_masked_similarities(
                Q, C_batch, weights,
                query_presence=query_presence,
                candidate_presence=candidate_presence,
                alpha=channel_weights[0], beta=channel_weights[1],
                gamma=channel_weights[2],
                missing_prior=CHANNEL_MISSING_PRIOR,
            )
            self._last_masked_sims = masked_sims
        else:
            phi_scores = self._transformer.compute_batch_component_similarities(
                Q, C_batch, weights,
                alpha=channel_weights[0], beta=channel_weights[1],
                gamma=channel_weights[2],
            )
        # Per-channel similarities for this (query, candidate pool)
        # pair: the listwise trainer learns the query-adaptive channel
        # weights over exactly these columns. The PRE-GATE candidate
        # pool is stashed too, index-aligned with the sims rows - the
        # trainer's listwise loss runs over the full pool (gold may be
        # present but gated out, and the loss must be able to teach the
        # channel weights to rank it back in).
        self._last_channel_sims = self._transformer.compute_channel_similarities(
            Q, C_batch, weights)
        self._last_candidate_pool = list(candidates)
        # Pre-gate provenance for the dev-pool harness (tau_sat calibration
        # and paired before/after retrieval metrics).
        self._last_phi_scores = np.asarray(phi_scores)
        self._last_query_presence = query_presence
        self._last_candidate_presence = candidate_presence

        filtered = self._filter.filter_contexts(
            candidates,
            np.asarray(phi_scores),
            k=k,
            max_top_k=MAX_EVIDENCE_ITEMS,
        )
        # Preserve stage-level evidence for diagnostics; these snapshots are
        # read-only evaluation metadata and do not alter production ranking.
        self._last_filtered_pool = list(filtered)

        # the previous "Fix #2" re-admitted candidates
        # BELOW the absolute floor Phi_min = 0.10 whenever both gates rejected
        # everything - defeating Gate 1 exactly on out-of-domain queries.
        # Correct behavior: abstain (empty context) so the generator relies on
        # parametric knowledge rather than being distracted by noise.
        if not filtered:
            logger.warning(
                "All %d candidates rejected by dual-gating (floor %.2f). "
                "Abstaining from retrieval - parametric mode.",
                len(candidates), MIN_PHI_SCORE,
            )
            self._last_post_gate_pool = []
            return []

        # JEV-inspired Decision Gate: Verify relevance & prune token bloat
        if self._decision_gate is not None:
            filtered = self._decision_gate.verify_and_prune_candidates(filtered, query)
        self._last_post_gate_pool = list(filtered)

        return filtered

    def retrieve_and_generate(
        self,
        query: str,
        k: Optional[float] = None,
        model: Optional[str] = None,
        contexts: Optional[list[dict]] = None,
    ) -> str:
        from src.core.llm_client import UnifiedLLMClient

        self._ensure_components()
        model = model or self.model_name
        if contexts is None:
            contexts = self.retrieve(query, k=k)
        messages = self._prompt_builder.build_rag_prompt(query, contexts)

        try:
            client = UnifiedLLMClient(model=model, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
            return client.chat(messages)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Error: LLM generation failed - {e}]"

    def answer_mcq(
        self,
        question: str,
        options: dict,
        k: Optional[float] = None,
        model: Optional[str] = None,
        n_samples: int = 1,
    ) -> tuple[str, str]:
        """Answer a multiple-choice question using retrieved context."""
        from collections import Counter
        from src.core.llm_client import UnifiedLLMClient
        from src.baselines.baseline_llm import extract_answer_letter

        self._ensure_components()
        model = model or self.model_name
        contexts = self.retrieve(question, k=k)
        messages = self._prompt_builder.build_mcq_prompt(
            question, options, contexts=contexts, few_shot=ENABLE_FEW_SHOT,
        )

        try:
            client = UnifiedLLMClient(model=model, temperature=LLM_TEMPERATURE, max_tokens=512)
            if n_samples > 1:
                votes = []
                responses = []
                for i in range(n_samples):
                    text = client.chat(messages, temperature=0.3, max_tokens=768)
                    letter = extract_answer_letter(text)
                    votes.append(letter)
                    responses.append(f"Sample {i+1} ({letter}):\n{text}")

                # deterministic tie-break - first sample wins.
                counts = Counter(votes)
                majority_vote = max(votes, key=lambda v: (counts[v], -votes.index(v)))
                full_response = "\n\n".join(responses)
                return majority_vote, full_response

            full_response = client.chat(messages)
            return extract_answer_letter(full_response), full_response
        except Exception as e:
            logger.error(f"MCQ answer failed: {e}")
            return "X", f"[Error: {e}]"
