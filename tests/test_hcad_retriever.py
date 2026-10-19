import unittest
from types import SimpleNamespace

import networkx as nx

from src.core.hcad_retriever import HCADRetriever

class TestHCADRetrieverEntityResolution(unittest.TestCase):
    def setUp(self):
        graph = nx.MultiDiGraph()
        graph.add_node(1, name="Metformin")
        graph.add_node(2, name="Metformin hydrochloride")
        self.retriever = HCADRetriever(
            graph=SimpleNamespace(
                graph=graph,
                name_to_node={"Metformin": 1, "Metformin hydrochloride": 2},
            )
        )

    def test_case_insensitive_exact_name_resolves(self):
        self.assertEqual(
            self.retriever._find_entity_in_graph("METFORMIN"),
            "Metformin",
        )

    def test_unmatched_name_does_not_resolve_by_arbitrary_prefix(self):
        self.assertIsNone(self.retriever._find_entity_in_graph("metformx"))

if __name__ == "__main__":
    unittest.main()
