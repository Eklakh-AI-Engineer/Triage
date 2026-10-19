import unittest

import networkx as nx

from src.core.hypergraph_builder import HypergraphBuilder

class TestHypergraphBuilder(unittest.TestCase):
    def test_name_resolved_center_uses_graph_node_key(self):
        graph = nx.MultiDiGraph()
        graph.add_node(1, name="DrugX", node_type="Drug")
        graph.add_node(2, name="DiseaseY", node_type="Disease")
        graph.add_edge(1, 2, relation="treats")
        graph.name_to_node = {"DrugX": 1}

        hyperedges = HypergraphBuilder(graph).build_hyperedges("DrugX")

        self.assertTrue(hyperedges)
        for hyperedge in hyperedges:
            self.assertIn(1, hyperedge["nodes"])
            self.assertNotIn("DrugX", hyperedge["nodes"])
            self.assertEqual(len(hyperedge["nodes"]), len(hyperedge["types"]))

if __name__ == "__main__":
    unittest.main()
