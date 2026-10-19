import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ENTITY_LINKING_MODEL, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    logger.warning("faiss is not installed. Entity linking will fail if FAISS is required.")
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("sentence_transformers is not installed.")
    SentenceTransformer = None

try:
    import numpy as np
except ImportError:
    logger.warning("numpy is not installed.")
    np = None

class SapBERTLinker:
    def __init__(self, model_name=None):
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is required for SapBERTLinker")

        self.model_name = model_name or ENTITY_LINKING_MODEL
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                if free_mem > 1200 * 1024 * 1024:
                    device = "cuda"
            except Exception:
                pass
        try:
            self.model = SentenceTransformer(self.model_name, device=device, local_files_only=True)
        except Exception:
            try:
                self.model = SentenceTransformer(self.model_name, device=device)
            except Exception:
                logger.warning(f"Failed to load {self.model_name} on {device}, falling back to CPU.")
                self.model = SentenceTransformer(self.model_name, device="cpu")

        self.index = None
        self.node_names = []
        self.name_to_id = {}

        self.index_path = Path(PROCESSED_DATA_DIR) / "sapbert_node_index.faiss"
        self.names_path = Path(PROCESSED_DATA_DIR) / "sapbert_node_names.json"

    def build_index(self, kg):
        if faiss is None:
            raise ImportError("faiss is required for SapBERTLinker")

        self.name_to_id = kg.name_to_node

        if self.index_path.exists() and self.names_path.exists():
            # bind the cache to the current graph via an
            # ordered-name signature; a stale index previously linked entities
            # to removed graph nodes silently.
            current_names = list(kg.name_to_node.keys()) if getattr(kg, "name_to_node", None) else []
            with open(self.names_path, 'r', encoding='utf-8') as f:
                self.node_names = json.load(f)
            if current_names and self.node_names != current_names:
                logger.warning(
                    "Cached SapBERT index is stale (%d cached vs %d graph names) - rebuilding.",
                    len(self.node_names), len(current_names),
                )
            else:
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded {len(self.node_names)} node names from cache.")
                return
            # fall through to rebuild

        logger.info("Building FAISS index from scratch...")
        self.node_names = list(self.name_to_id.keys())

        if not self.node_names:
            logger.warning("Knowledge graph has no nodes to index.")
            return

        # Encode node names in batches
        logger.info(f"Encoding {len(self.node_names)} node names...")
        embeddings = self.model.encode(self.node_names, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        # L2 normalize for cosine similarity via Inner Product
        faiss.normalize_L2(embeddings)

        # Build index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

        # Save cache
        logger.info("Saving FAISS index and node names to cache...")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.names_path, 'w', encoding='utf-8') as f:
            json.dump(self.node_names, f)

        logger.info("Index built successfully.")

    def link(self, entity_text: str, top_k: int = 3, threshold: float = 0.7) -> list[dict]:
        if self.index is None:
            logger.error("FAISS index is not built. Call build_index() first.")
            return []

        import torch
        try:
            embedding = self.model.encode([entity_text], convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            logger.warning(f"Embedding encoding failed on device, falling back to CPU: {e}")
            self.model = self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            embedding = self.model.encode([entity_text], convert_to_numpy=True, normalize_embeddings=True)
        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:
                name = self.node_names[idx]
                node_id = self.name_to_id.get(name)
                results.append({
                    'canonical_name': name,
                    'node_id': node_id,
                    'score': float(score)
                })

        return results

    def link_entities(self, entities: dict) -> dict:
        linked_entities = {}

        for entity_type, entity_list in entities.items():
            linked_entities[entity_type] = []
            for entity in entity_list:
                if isinstance(entity, str):
                    surface_form = entity
                    entity_data = {'text': surface_form}
                else:
                    surface_form = entity.get('text', '')
                    entity_data = entity.copy()

                if surface_form:
                    matches = self.link(surface_form, top_k=1, threshold=0.7)
                    if matches:
                        match = matches[0]
                        entity_data['canonical_name'] = match['canonical_name']
                        entity_data['node_id'] = match['node_id']
                        entity_data['link_score'] = match['score']
                    else:
                        entity_data['canonical_name'] = None
                        entity_data['node_id'] = None

                linked_entities[entity_type].append(entity_data)

        return linked_entities
