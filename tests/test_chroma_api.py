"""Exercise the Chroma calls and include/offset options used by the pipeline."""

import chromadb


def test_target_chroma_get_and_query_kwargs(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection(name="api_contract")
    collection.add(
        ids=["edge-1", "edge-2"],
        documents=["DrugA treats DiseaseA", "DrugB causes DiseaseB"],
        metadatas=[{"relation": "TREATS"}, {"relation": "CAUSES"}],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    fetched = collection.get(
        where={"relation": "TREATS"}, limit=1, offset=0,
        include=["documents", "metadatas"],
    )
    assert fetched["ids"] == ["edge-1"]
    assert fetched["documents"] == ["DrugA treats DiseaseA"]

    queried = collection.query(
        query_embeddings=[[1.0, 0.0]], n_results=1,
    )
    assert queried["ids"] == [["edge-1"]]
    assert queried["documents"] == [["DrugA treats DiseaseA"]]

    # _stratum_size's binary search relies on offset >= filtered count
    # returning an EMPTY result (its monotone probe predicate).
    past_end = collection.get(
        where={"relation": "TREATS"}, limit=1, offset=1,
        include=["documents", "metadatas"],
    )
    assert past_end["ids"] == []
    assert past_end["documents"] == []

    # Vanilla RAG queries with an explicit include list (distances are
    # allowed on query, unlike on get).
    queried_inc = collection.query(
        query_embeddings=[[1.0, 0.0]], n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    assert queried_inc["ids"] == [["edge-1"]]
    assert queried_inc["distances"] is not None

    client.delete_collection("api_contract")
