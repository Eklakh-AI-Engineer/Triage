"""
Build and query the PrimeKG biomedical knowledge graph.

Loads the PrimeKG CSV  into a NetworkX MultiDiGraph, builds
lookup indices, and provides a ``KnowledgeGraph`` wrapper with methods for
multi-hop subgraph extraction, neighbor retrieval, and triple enumeration.

The graph is serialized to pickle after the first build so subsequent loads
are near-instantaneous.

Usage:
    python -m src.data.build_knowledge_graph      # build and save graph
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

from src.config import (
    GRAPH_HOP_DEPTH,
    KG_EXCLUDED_RELATIONS,
    MAX_NEIGHBORS_PER_HOP,
    PRIMEKG_FILENAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)

# Paths. The CSV location is resolved at CALL time, not import time:
# on a clean machine the download in `prepare` Step 1 finishes AFTER this
# module is first imported, so a module-level .exists() check would freeze
# on the wrong path.
_PICKLE_PATH: Path = PROCESSED_DATA_DIR / "primekg_graph.gpickle"


def _resolve_csv_path() -> Path:
    subdir = RAW_DATA_DIR / "primekg" / PRIMEKG_FILENAME
    if subdir.exists():
        return subdir
    return RAW_DATA_DIR / PRIMEKG_FILENAME

# Build helpers

def _build_graph_from_csv(csv_path: Path) -> tuple[
    nx.MultiDiGraph,
    dict[str, int],
    dict[str, list[int]],
    dict[int, list[tuple[int, str]]],
]:
    """Parse PrimeKG CSV and construct graph + lookup dictionaries.

    Expected CSV columns :
        relation, display_relation, x_index, y_index,
        x_name, y_name, x_type, y_type

    Returns:
        A 4-tuple of:
        - ``G``: ``nx.MultiDiGraph`` with node/edge attributes.
        - ``name_to_node``: entity name → node index.
        - ``type_to_nodes``: node type → list of node indices.
        - ``node_to_neighbors``: node index → list of (neighbor_index, relation).
    """
    logger.info("Reading PrimeKG CSV from %s …", csv_path)
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except FileNotFoundError:
        logger.error(
            "PrimeKG CSV not found at %s. "
            "Run `python -m src.data.download_datasets` first.",
            csv_path,
        )
        raise
    except Exception as exc:
        logger.error("Error reading PrimeKG CSV: %s", exc)
        raise

    required_cols = {
        "relation", "display_relation",
        "x_index", "y_index",
        "x_name", "y_name",
        "x_type", "y_type",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"PrimeKG CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # apply the DDI exclusion policy IN THE
    # GRAPH BUILDER. The "2.6M DDI edges stripped" claim previously held only
    # for the vector store; the pickled graph retained all 2,672,628 drug_drug
    # edges and the hypergraph/anisotropic paths traversed them.
    if KG_EXCLUDED_RELATIONS:
        before = len(df)
        df = df[~df["relation"].isin(KG_EXCLUDED_RELATIONS)]
        logger.info(
            "Graph relation policy: excluded %s → %d -> %d edges",
            sorted(KG_EXCLUDED_RELATIONS), before, len(df),
        )

    logger.info("Building MultiDiGraph from %d edges …", len(df))

    G = nx.MultiDiGraph()

    name_to_node: dict[str, int] = {}
    type_to_nodes: dict[str, list[int]] = defaultdict(list)
    node_to_neighbors: dict[int, list[tuple[int, str]]] = defaultdict(list)

    # Convert columns to Python lists for 50x faster vector iteration
    x_indices = df["x_index"].astype(int).tolist()
    x_names = df["x_name"].astype(str).tolist()
    x_types = df["x_type"].astype(str).tolist()

    y_indices = df["y_index"].astype(int).tolist()
    y_names = df["y_name"].astype(str).tolist()
    y_types = df["y_type"].astype(str).tolist()

    relations = df["relation"].astype(str).tolist()
    disp_relations = df["display_relation"].astype(str).tolist()

    # Register all nodes first
    for x_idx, x_name, x_type, y_idx, y_name, y_type in zip(
        x_indices, x_names, x_types, y_indices, y_names, y_types
    ):
        if x_idx not in G:
            G.add_node(x_idx, name=x_name, node_type=x_type)
            if x_name not in name_to_node:
                name_to_node[x_name] = x_idx
            type_to_nodes[x_type].append(x_idx)

        if y_idx not in G:
            G.add_node(y_idx, name=y_name, node_type=y_type)
            if y_name not in name_to_node:
                name_to_node[y_name] = y_idx
            type_to_nodes[y_type].append(y_idx)

    # Register all edges
    for src, dst, rel, disp_rel in zip(x_indices, y_indices, relations, disp_relations):
        G.add_edge(src, dst, relation=rel, display_relation=disp_rel)
        node_to_neighbors[src].append((dst, rel))
        node_to_neighbors[dst].append((src, rel))

    # Convert defaultdicts → plain dicts for pickling clarity
    type_to_nodes = dict(type_to_nodes)
    node_to_neighbors = dict(node_to_neighbors)

    _log_graph_stats(G, type_to_nodes)

    return G, name_to_node, type_to_nodes, node_to_neighbors

def _log_graph_stats(
    G: nx.MultiDiGraph,
    type_to_nodes: dict[str, list[int]],
) -> None:
    """Log summary statistics about the knowledge graph."""
    logger.info("=" * 60)
    logger.info("PrimeKG Graph Statistics")
    logger.info("=" * 60)
    logger.info("  Nodes        : %d", G.number_of_nodes())
    logger.info("  Edges        : %d", G.number_of_edges())

    logger.info("  Node types   :")
    for ntype, nodes in sorted(type_to_nodes.items(), key=lambda x: -len(x[1])):
        logger.info("    %-25s %d", ntype, len(nodes))

    # Relation distribution from edges
    rel_counter: Counter[str] = Counter()
    for _, _, data in G.edges(data=True):
        rel_counter[data.get("relation", "UNKNOWN")] += 1
    logger.info("  Relation types (%d unique):", len(rel_counter))
    for rel, cnt in rel_counter.most_common(20):
        logger.info("    %-40s %d", rel, cnt)
    if len(rel_counter) > 20:
        logger.info("    … and %d more", len(rel_counter) - 20)
    logger.info("=" * 60)

def _save_graph(
    G: nx.MultiDiGraph,
    name_to_node: dict[str, int],
    type_to_nodes: dict[str, list[int]],
    node_to_neighbors: dict[int, list[tuple[int, str]]],
    path: Path,
) -> None:
    """Serialize graph + lookup dicts to a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graph": G,
        "name_to_node": name_to_node,
        "type_to_nodes": type_to_nodes,
        "node_to_neighbors": node_to_neighbors,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / 1e6
    logger.info("Graph serialized to %s (%.1f MB).", path, size_mb)

def _load_from_pickle(path: Path) -> tuple[
    nx.MultiDiGraph,
    dict[str, int],
    dict[str, list[int]],
    dict[int, list[tuple[int, str]]],
]:
    """Deserialize graph + lookup dicts from pickle."""
    logger.info("Loading graph from pickle: %s", path)
    with open(path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    return (
        payload["graph"],
        payload["name_to_node"],
        payload["type_to_nodes"],
        payload["node_to_neighbors"],
    )

# Public loader

def load_knowledge_graph(
    csv_path: Optional[Path] = None,
    pickle_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> "KnowledgeGraph":
    """Load (or build) the PrimeKG knowledge graph.

    Priority order:
        1. Load from *pickle_path* if it exists and ``force_rebuild`` is False.
        2. Build from *csv_path*, serialize to *pickle_path*, and return.

    Args:
        csv_path:      Path to ``kg.csv``.  Defaults to ``RAW_DATA_DIR/kg.csv``.
        pickle_path:   Path for the serialized graph.
                       Defaults to ``PROCESSED_DATA_DIR/primekg_graph.gpickle``.
        force_rebuild: Rebuild from CSV even if pickle exists.

    Returns:
        A ``KnowledgeGraph`` instance wrapping the graph and indices.
    """
    csv_path = csv_path or _resolve_csv_path()
    pickle_path = pickle_path or _PICKLE_PATH

    if pickle_path.exists() and not force_rebuild:
        G, n2n, t2n, n2nb = _load_from_pickle(pickle_path)
    else:
        G, n2n, t2n, n2nb = _build_graph_from_csv(csv_path)
        _save_graph(G, n2n, t2n, n2nb, pickle_path)

    return KnowledgeGraph(G, n2n, t2n, n2nb)

# KnowledgeGraph class

class KnowledgeGraph:
    """High-level wrapper around the PrimeKG ``MultiDiGraph``.

    Provides convenient methods used downstream by the HCAD-RAG retrieval
    pipeline (Sections 4.19-4.20).

    Attributes:
        graph:             The underlying ``nx.MultiDiGraph``.
        name_to_node:      Entity name → node index mapping.
        type_to_nodes:     Node type → list of node indices.
        node_to_neighbors: Node index → list of ``(neighbor_idx, relation)``.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        name_to_node: dict[str, int],
        type_to_nodes: dict[str, list[int]],
        node_to_neighbors: dict[int, list[tuple[int, str]]],
    ) -> None:
        self.graph = graph
        self.name_to_node = name_to_node
        self.type_to_nodes = type_to_nodes
        self.node_to_neighbors = node_to_neighbors

    # -- node lookup --------------------------------------------------------

    def get_node_by_name(self, name: str) -> dict:
        """Return the attribute dict for the node with the given name.

        Args:
            name: Entity name (case-sensitive as stored in PrimeKG).

        Returns:
            Dictionary with ``name``, ``node_type``, and the integer index
            under key ``_index``.

        Raises:
            KeyError: If the name is not found in the graph.
        """
        if name not in self.name_to_node:
            raise KeyError(
                f"Entity '{name}' not found in the knowledge graph. "
                "Check spelling or use fuzzy matching upstream."
            )
        idx = self.name_to_node[name]
        attrs = dict(self.graph.nodes[idx])
        attrs["_index"] = idx
        return attrs

    # -- neighbor retrieval -------------------------------------------------

    def get_neighbors(
        self,
        node_name: str,
        relation: Optional[str] = None,
    ) -> list[dict]:
        """Return neighbor information for *node_name*.

        Args:
            node_name: Source entity name.
            relation:  If provided, filter to only this relation type.

        Returns:
            List of dicts, each containing ``neighbor_name``,
            ``neighbor_type``, ``relation``, and ``neighbor_index``.
        """
        idx = self.name_to_node.get(node_name)
        if idx is None:
            logger.warning("Entity '%s' not found - returning empty list.", node_name)
            return []

        neighbors = self.node_to_neighbors.get(idx, [])
        results: list[dict] = []
        for nb_idx, rel in neighbors:
            if relation is not None and rel != relation:
                continue
            nb_data = self.graph.nodes.get(nb_idx, {})
            results.append({
                "neighbor_name": nb_data.get("name", "UNKNOWN"),
                "neighbor_type": nb_data.get("node_type", "UNKNOWN"),
                "relation": rel,
                "neighbor_index": nb_idx,
            })
        return results

    # -- multi-hop subgraph extraction  -------------------

    def get_subgraph(
        self,
        node_name: str,
        hops: int = GRAPH_HOP_DEPTH,
        max_per_hop: int = MAX_NEIGHBORS_PER_HOP,
    ) -> nx.MultiDiGraph:
        """Extract a multi-hop ego subgraph around *node_name*.

        At each hop, at most ``max_per_hop`` neighbors are expanded (sorted
        by degree centrality to prefer well-connected nodes).

        Args:
            node_name:   Centre entity name.
            hops:        Expansion depth (default from ``config.GRAPH_HOP_DEPTH``).
            max_per_hop: Cap per hop (default from ``config.MAX_NEIGHBORS_PER_HOP``).

        Returns:
            A ``MultiDiGraph`` subgraph containing all reached nodes/edges.

        Raises:
            KeyError: If *node_name* is not in the graph.
        """
        start_idx = self.name_to_node.get(node_name)
        if start_idx is None:
            raise KeyError(f"Entity '{node_name}' not in graph.")

        visited: set[int] = {start_idx}
        frontier: set[int] = {start_idx}

        for _hop in range(hops):
            next_frontier: set[int] = set()
            for node in frontier:
                raw_neighbors = self.node_to_neighbors.get(node, [])
                # Sort neighbors by degree ASCENDING - prefer specific,
                # informative nodes over generic hub nodes like "disease"
                # or "protein" that add noise.  Skip nodes with degree
                # > 500 entirely (hub node cap).
                HUB_DEGREE_CAP = 500
                sorted_nbs = sorted(
                    [(nb, rel) for nb, rel in raw_neighbors
                     if self.graph.degree(nb) <= HUB_DEGREE_CAP],
                    key=lambda x: self.graph.degree(x[0]),
                    reverse=False,  # ascending - specific nodes first
                )
                count = 0
                for nb_idx, _rel in sorted_nbs:
                    if count >= max_per_hop:
                        break
                    if nb_idx not in visited:
                        visited.add(nb_idx)
                        next_frontier.add(nb_idx)
                        count += 1
            frontier = next_frontier

        return self.graph.subgraph(visited).copy()

    # -- triple enumeration -------------------------------------------------

    def get_triples_for_entity(
        self,
        entity_name: str,
    ) -> list[tuple[str, str, str]]:
        """Return ``(head_name, relation, tail_name)`` triples involving *entity_name*.

        Both outgoing and incoming edges are included so the entity can appear
        as head *or* tail.

        Args:
            entity_name: The entity to query.

        Returns:
            List of string triples.
        """
        idx = self.name_to_node.get(entity_name)
        if idx is None:
            logger.warning(
                "Entity '%s' not found - returning empty triple list.", entity_name
            )
            return []

        triples: list[tuple[str, str, str]] = []
        entity_label = self.graph.nodes[idx].get("name", entity_name)

        # Outgoing edges (entity is head)
        for _, dst, data in self.graph.out_edges(idx, data=True):
            tail_label = self.graph.nodes[dst].get("name", str(dst))
            rel = data.get("display_relation", data.get("relation", "related_to"))
            triples.append((entity_label, rel, tail_label))

        # Incoming edges (entity is tail)
        for src, _, data in self.graph.in_edges(idx, data=True):
            head_label = self.graph.nodes[src].get("name", str(src))
            rel = data.get("display_relation", data.get("relation", "related_to"))
            triples.append((head_label, rel, entity_label))

        return triples

    # -- convenience --------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KnowledgeGraph(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )

# CLI entry-point

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build or load the PrimeKG knowledge graph and entity linking index."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild graph from CSV even if pickle exists.",
    )
    parser.add_argument(
        "--skip-sapbert",
        action="store_true",
        help="Skip building the SapBERT FAISS entity linking index.",
    )
    args = parser.parse_args()

    # Step 1: Build/load knowledge graph
    kg = load_knowledge_graph(force_rebuild=args.force)
    logger.info("Loaded: %s", kg)

    # Step 2: Build SapBERT FAISS index for entity linking
    if not args.skip_sapbert:
        try:
            from src.core.entity_linker import SapBERTLinker

            logger.info("=" * 60)
            logger.info("Building SapBERT FAISS entity linking index...")
            logger.info("=" * 60)
            linker = SapBERTLinker()
            linker.build_index(kg)
            logger.info(
                "SapBERT index ready: %d nodes indexed.",
                len(linker.node_names),
            )
        except ImportError as e:
            logger.warning(
                "Could not build SapBERT index (missing dependency: %s). "
                "Entity linking will fall back to string matching.", e
            )
        except Exception as e:
            logger.error("SapBERT index build failed: %s", e)

    logger.info("=" * 60)
    logger.info("Knowledge graph preparation complete.")
    logger.info("  Nodes: %d", kg.graph.number_of_nodes())
    logger.info("  Edges: %d", kg.graph.number_of_edges())
    logger.info("=" * 60)
