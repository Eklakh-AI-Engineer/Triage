import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.main import cmd_prepare

class TestPrepareCommand(unittest.TestCase):
    def test_passes_max_rows_to_vector_store_builder(self):
        graph = SimpleNamespace(graph=SimpleNamespace(number_of_nodes=lambda: 0, number_of_edges=lambda: 0))
        collection = SimpleNamespace(count=lambda: 0)
        build_vector_store = Mock(return_value=collection)
        ingest_clinical_chunks = Mock(return_value=0)
        download_module = types.ModuleType("src.data.download_datasets")
        download_module.download_all = Mock(return_value={"primekg": "/tmp/kg.csv"})
        graph_module = types.ModuleType("src.data.build_knowledge_graph")
        graph_module.load_knowledge_graph = Mock(return_value=graph)
        vector_module = types.ModuleType("src.data.build_vector_store")
        vector_module.build_vector_store = build_vector_store
        vector_module.ingest_clinical_chunks = ingest_clinical_chunks
        linker_module = types.ModuleType("src.core.entity_linker")

        class StubLinker:
            node_names = []

            def build_index(self, _graph):
                pass

        linker_module.SapBERTLinker = StubLinker

        class StubPath:
            def __init__(self, path):
                self.path = path

            def exists(self):
                return False

            def glob(self, _pattern):
                return []

        modules = {
            "src.data.download_datasets": download_module,
            "src.data.build_knowledge_graph": graph_module,
            "src.data.build_vector_store": vector_module,
            "src.core.entity_linker": linker_module,
        }
        with (
            patch.dict(sys.modules, modules),
            patch("src.main.Path", StubPath),
            patch("src.main.print"),
        ):
            cmd_prepare(SimpleNamespace(rebuild=False, max_rows=123))

        build_vector_store.assert_called_once()
        self.assertEqual(build_vector_store.call_args.kwargs["max_rows"], 123)

if __name__ == "__main__":
    unittest.main()
