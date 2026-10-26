"""P2b: FactIndex role direction, undirected relations, and negation prose.

PrimeKG facts are asymmetric: the shared tail is the anchor and the grouped
heads are the head side. disease_disease is undirected; negative relations
(disease_phenotype_negative) render with exactly one negation token - the
relation name already encodes absence.
"""

import pytest

from src.core.fact_hypergraph import (
    NEGATED_RELATIONS,
    UNDIRECTED_RELATIONS,
    ClinicalFact,
    FactIndex,
    build_fact_index_from_kg,
)

WHITELIST = [
    "contraindication", "indication", "off-label use", "drug_effect",
    "disease_disease", "drug_protein", "disease_protein",
    "disease_phenotype_positive", "disease_phenotype_negative",
    "biomarker", "phenotype_protein", "protein_protein",
]


class _StubKG:
    """Minimal PrimeKG-shaped stub: .graph is a networkx MultiDiGraph."""

    def __init__(self):
        import networkx as nx

        self.graph = nx.MultiDiGraph()
        self.graph.add_node("d1", name="Renal Failure", node_type="disease")
        self.graph.add_node("d2", name="Epilepsy", node_type="disease")
        self.graph.add_node("d3", name="Migraine", node_type="disease")
        self.graph.add_node("dr1", name="Metformin", node_type="drug")
        self.graph.add_node("dr2", name="Glyburide", node_type="drug")
        self.graph.add_node("dr3", name="Phenytoin", node_type="drug")
        self.graph.add_node("dr4", name="Topiramate", node_type="drug")
        self.graph.add_node("p1", name="Lactic Acidosis", node_type="disease")
        self.graph.add_node("p2", name="Seizure", node_type="disease")
        self.graph.add_node("pr1", name="Protein X", node_type="protein")
        self.graph.add_node("pr2", name="Protein Y", node_type="protein")

        # Every relation needs >= 2 heads sharing its tail (min_members=2):
        # the n-ary fact is the shared-tail context itself.
        # contraindication: two drugs share the (disease) tail
        self.graph.add_edge("dr1", "d1", relation="contraindication")
        self.graph.add_edge("dr2", "d1", relation="contraindication")
        # indication: two drugs indicate epilepsy
        self.graph.add_edge("dr3", "d2", relation="indication")
        self.graph.add_edge("dr4", "d2", relation="indication")
        # off-label use: two drugs off-label for migraine
        self.graph.add_edge("dr1", "d3", relation="off-label use")
        self.graph.add_edge("dr2", "d3", relation="off-label use")
        # drug_effect: two drugs cause lactic acidosis
        self.graph.add_edge("dr1", "p1", relation="drug_effect")
        self.graph.add_edge("dr2", "p1", relation="drug_effect")
        # disease_disease (undirected association)
        self.graph.add_edge("d2", "d3", relation="disease_disease")
        self.graph.add_edge("d1", "d3", relation="disease_disease")
        # drug_protein: two drugs target protein X
        self.graph.add_edge("dr1", "pr1", relation="drug_protein")
        self.graph.add_edge("dr2", "pr1", relation="drug_protein")
        # disease_protein: two diseases associate with protein X
        self.graph.add_edge("d1", "pr1", relation="disease_protein")
        self.graph.add_edge("d2", "pr1", relation="disease_protein")
        # disease_phenotype_positive: two diseases show lactic acidosis
        self.graph.add_edge("d1", "p1", relation="disease_phenotype_positive")
        self.graph.add_edge("d2", "p1", relation="disease_phenotype_positive")
        # disease_phenotype_negative (negated relation): renal failure and
        # epilepsy are NOT associated with seizure
        self.graph.add_edge("d1", "p2", relation="disease_phenotype_negative")
        self.graph.add_edge("d2", "p2", relation="disease_phenotype_negative")
        # biomarker: protein X and protein Y are biomarkers of renal failure
        self.graph.add_edge("pr1", "d1", relation="biomarker")
        self.graph.add_edge("pr2", "d1", relation="biomarker")
        # phenotype_protein: two diseases' phenotype involves protein X
        self.graph.add_edge("p1", "pr1", relation="phenotype_protein")
        self.graph.add_edge("p2", "pr1", relation="phenotype_protein")
        # protein_protein: self-pair grouped by tail pr1
        self.graph.add_edge("pr2", "pr1", relation="protein_protein")
        self.graph.add_edge("pr1", "pr1", relation="protein_protein")


def test_all_whitelisted_relations_indexed():
    idx = build_fact_index_from_kg(_StubKG())
    predicates = {f.predicate for f in idx.facts}
    for rel in WHITELIST:
        assert rel in predicates, f"missing relation {rel}"


def test_direction_and_polarity_semantics():
    idx = build_fact_index_from_kg(_StubKG())
    for f in idx.facts:
        if f.predicate in UNDIRECTED_RELATIONS:
            assert f.direction == "undirected"
        else:
            assert f.direction == "directed"
        if f.predicate in NEGATED_RELATIONS:
            assert f.polarity == "negated"
        else:
            assert f.polarity == "asserts"


def test_heads_and_tail_roles_are_asymmetric():
    idx = build_fact_index_from_kg(_StubKG())
    contra = [f for f in idx.facts if f.predicate == "contraindication"][0]
    # grouped heads are the head side; the shared tail is the anchor
    assert sorted(contra.head_names) == ["Glyburide", "Metformin"]
    assert contra.tail_name == "Renal Failure"
    assert "anchor" not in {k for k in contra.roles} or contra.roles["anchor"] == "Renal Failure"


def test_render_directed_relation_is_asymmetric():
    idx = build_fact_index_from_kg(_StubKG())
    contra = [f for f in idx.facts if f.predicate == "contraindication"][0]
    rendered = idx.render(contra)
    assert rendered == "Metformin, Glyburide --[contraindication]--> Renal Failure."
    # the tail must appear AFTER the arrow, not as a co-equal member
    assert rendered.index("-->") < rendered.index("Renal Failure")


def test_render_disease_disease_is_undirected():
    idx = build_fact_index_from_kg(_StubKG())
    dd = [f for f in idx.facts if f.predicate == "disease_disease"][0]
    rendered = idx.render(dd)
    assert "--[disease_disease]--" in rendered
    assert "-->" not in rendered
    assert "undirected" in rendered


def test_render_negative_relation_single_negation():
    idx = build_fact_index_from_kg(_StubKG())
    neg = [f for f in idx.facts if f.predicate == "disease_phenotype_negative"][0]
    rendered = idx.render(neg)
    low = rendered.lower()
    assert "not associated" in low
    # exactly one negation token: no double negation like "not lack"/"not without"
    import re
    assert not re.search(r"\b(not|no|never)\s+(not|no|never|lack of|absence of|without)\b", low)


def test_double_negation_guard_raises():
    fact = ClinicalFact(
        predicate="disease_phenotype_negative",
        roles={"member_0": "Migraine", "anchor": "Renal Failure"},
        polarity="negated",
    )
    with pytest.raises(ValueError, match="[Dd]ouble-negation"):
        FactIndex._single_negation_guard("Migraine is not without Renal Failure.", fact)


def test_asserts_relation_with_negation_cue_is_allowed_when_from_entity_name():
    # An entity name that legitimately contains a negation word must not
    # trip the composition guard (only composed double negation is fatal).
    fact = ClinicalFact(
        predicate="indication",
        roles={"member_0": "Metformin", "anchor": "Absence of Seizures"},
    )
    FactIndex._single_negation_guard(
        "Metformin --[indication]--> Absence of Seizures.", fact
    )


def test_retrieve_still_finds_facts_by_anchor():
    idx = build_fact_index_from_kg(_StubKG())
    hits = idx.retrieve({"Renal Failure"}, top_k=10)
    assert hits, "anchor-based retrieval must still work after P2b"
    assert any(f.predicate == "contraindication" for f in hits)
