"""P2c: negation cue windows terminate at clause boundaries.

A negation cue may not cross a sentence/clause separator ([.;,:]) or a
contrastive conjunction to negate a relation stated in a different clause.
20-sentence regression suite; each case is asserted on the specific
(head, relation, tail) pair so unrelated pair matches cannot mask the
outcome. Sentences use relation keywords in the inflections the pattern
lists actually match (e.g. "combined with", "indicated for", "treats").
"""

from src.core.relation_extractor import RelationExtractor

DRUG = "Drug"
DIS = "Disease"


def _entities(text, specs):
    """Build an entity dict with true char offsets for (text, type) specs."""
    out = {}
    for ent_text, etype in specs:
        start = text.find(ent_text)
        assert start >= 0, f"entity {ent_text!r} not in text"
        out.setdefault(etype, []).append(
            {"text": ent_text, "start": start, "end": start + len(ent_text)}
        )
    return out


def _check(text, specs, head, relation, tail, expect_negated):
    rex = RelationExtractor()
    rels = rex.extract_rules(text, _entities(text, specs))
    for r in rels:
        if r["head"] == head and r["relation"] == relation and r["tail"] == tail:
            assert r["is_negated"] is expect_negated, (
                f"({head}, {relation}, {tail}) in {text!r}: "
                f"is_negated={r['is_negated']}, expected {expect_negated}"
            )
            return
    raise AssertionError(
        f"expected relation ({head}, {relation}, {tail}) not extracted from: {text!r}\n"
        f"got: {[(x['head'], x['relation'], x['tail'], x['is_negated']) for x in rels]}"
    )


# --- Cues in the SAME clause negate the relation (1-5) ---

def test_same_clause_not_indicated_for():
    _check(
        "Metformin is not indicated for lung cancer.",
        [("Metformin", DRUG), ("lung cancer", DIS)],
        "Metformin", "INDICATED_FOR", "lung cancer", True,
    )


def test_same_clause_contraindicated():
    _check(
        "Aspirin is contraindicated in children with viral fever.",
        [("Aspirin", DRUG), ("children with viral fever", DIS)],
        "Aspirin", "CONTRAINDICATED", "children with viral fever", True,
    )


def test_same_clause_should_not_be_combined():
    _check(
        "Warfarin should not be combined with NSAIDs.",
        [("Warfarin", DRUG), ("NSAIDs", DRUG)],
        "Warfarin", "INTERACTS_WITH", "NSAIDs", True,
    )


def test_cue_before_head_negates():
    _check(
        "There is no evidence that Metformin causes weight gain.",
        [("Metformin", DRUG), ("weight gain", DIS)],
        "Metformin", "CAUSES", "weight gain", True,
    )


def test_cue_after_comma_but_same_clause():
    _check(
        "Because of hepatic risk, Drug A is not recommended for liver failure.",
        [("Drug A", DRUG), ("liver failure", DIS)],
        "Drug A", "NOT_RECOMMENDED", "liver failure", True,
    )


# --- Cues separated by a PERIOD must not leak (6-8) ---

def test_period_blocks_forward_leak():
    # canonical P2c failure: the cue from sentence 1 used to negate the
    # relation stated in sentence 2.
    _check(
        "Aspirin should not be given to children. Ibuprofen treats fever.",
        [("Ibuprofen", DRUG), ("fever", DIS)],
        "Ibuprofen", "TREATS", "fever", False,
    )


def test_period_blocks_cue_from_later_sentence():
    _check(
        "Penicillin treats bacterial infections. Aspirin should not be given to children.",
        [("Penicillin", DRUG), ("bacterial infections", DIS)],
        "Penicillin", "TREATS", "bacterial infections", False,
    )


def test_period_with_however_keeps_pairs_separate():
    _check(
        "Drug A treats disease X. However, Drug B is contraindicated in disease Y.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_second_sentence_relation_still_negates():
    _check(
        "Drug A treats disease X. However, Drug B is contraindicated in disease Y.",
        [("Drug B", DRUG), ("disease Y", DIS)],
        "Drug B", "CONTRAINDICATED", "disease Y", True,
    )


# --- Semicolon, comma, colon, contrastive-conjunction boundaries (9-12) ---

def test_semicolon_blocks_leak():
    _check(
        "Drug A treats disease X; Drug B is not indicated for disease Y.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_comma_blocks_leak():
    _check(
        "Drug A treats disease X, but Drug B is not indicated for disease Y.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_colon_blocks_leak():
    # "avoid" is a cue, but the colon ends its clause before the relation.
    _check(
        "Avoid in pregnancy: Drug A treats disease X safely.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_contrastive_whereas_blocks_leak():
    _check(
        "Drug A treats disease X, whereas Drug B is not indicated for disease Y.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


# --- Cue between head and tail (13-16) ---

def test_cue_between_entities_negates():
    _check(
        "Metformin is not associated with lactic acidosis.",
        [("Metformin", DRUG), ("lactic acidosis", DIS)],
        "Metformin", "ASSOCIATED_WITH", "lactic acidosis", True,
    )


def test_two_pairs_in_two_sentences():
    text = "Drug A interacts with Drug B. Drug C is not combined with Drug D."
    _check(
        text,
        [("Drug A", DRUG), ("Drug B", DRUG)],
        "Drug A", "INTERACTS_WITH", "Drug B", False,
    )
    _check(
        text,
        [("Drug C", DRUG), ("Drug D", DRUG)],
        "Drug C", "INTERACTS_WITH", "Drug D", True,
    )


def test_word_boundary_noted_is_not_a_cue():
    # audit item 21 regression: "noted" must not match the "not" cue
    _check(
        "Noted that Drug A treats disease X well.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_cue_after_tail_not_counted():
    # "without difficulty" appears after the tail; the cue window ends at
    # the tail start so it cannot negate the relation.
    _check(
        "Drug A manages disease X without difficulty.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


# --- Contrastive conjunctions inside the lookback window (17-20) ---

def test_yet_blocks_cue_inside_lookback():
    _check(
        "Drug B is not indicated for disease Z, yet Drug A treats disease X.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_although_blocks_cue():
    _check(
        "Although Drug B is not recommended for disease Z, Drug A treats disease X.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_nevertheless_blocks_cue():
    _check(
        "Drug B never treats disease Z; nevertheless Drug A treats disease X.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "TREATS", "disease X", False,
    )


def test_cue_in_same_clause_as_its_own_pair_negates():
    _check(
        "Drug A is not indicated for disease X and Drug B is indicated for disease Y.",
        [("Drug A", DRUG), ("disease X", DIS)],
        "Drug A", "INDICATED_FOR", "disease X", True,
    )


def test_full_twenty_sentence_regression():
    """All 20 regression sentences in one table-driven pass.

    Each row: (sentence, entities, head, relation, tail, expect_negated).
    """
    cases = [
        ("Metformin is not indicated for lung cancer.",
         [("Metformin", DRUG), ("lung cancer", DIS)],
         "Metformin", "INDICATED_FOR", "lung cancer", True),
        ("Aspirin is contraindicated in children with viral fever.",
         [("Aspirin", DRUG), ("children with viral fever", DIS)],
         "Aspirin", "CONTRAINDICATED", "children with viral fever", True),
        ("Warfarin should not be combined with NSAIDs.",
         [("Warfarin", DRUG), ("NSAIDs", DRUG)],
         "Warfarin", "INTERACTS_WITH", "NSAIDs", True),
        ("There is no evidence that Metformin causes weight gain.",
         [("Metformin", DRUG), ("weight gain", DIS)],
         "Metformin", "CAUSES", "weight gain", True),
        ("Because of hepatic risk, Drug A is not recommended for liver failure.",
         [("Drug A", DRUG), ("liver failure", DIS)],
         "Drug A", "NOT_RECOMMENDED", "liver failure", True),
        ("Aspirin should not be given to children. Ibuprofen treats fever.",
         [("Ibuprofen", DRUG), ("fever", DIS)],
         "Ibuprofen", "TREATS", "fever", False),
        ("Penicillin treats bacterial infections. Aspirin should not be given to children.",
         [("Penicillin", DRUG), ("bacterial infections", DIS)],
         "Penicillin", "TREATS", "bacterial infections", False),
        ("Drug A treats disease X. However, Drug B is contraindicated in disease Y.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Drug A treats disease X. However, Drug B is contraindicated in disease Y.",
         [("Drug B", DRUG), ("disease Y", DIS)],
         "Drug B", "CONTRAINDICATED", "disease Y", True),
        ("Drug A treats disease X; Drug B is not indicated for disease Y.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Drug A treats disease X, but Drug B is not indicated for disease Y.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Avoid in pregnancy: Drug A treats disease X safely.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Drug A treats disease X, whereas Drug B is not indicated for disease Y.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Metformin is not associated with lactic acidosis.",
         [("Metformin", DRUG), ("lactic acidosis", DIS)],
         "Metformin", "ASSOCIATED_WITH", "lactic acidosis", True),
        ("Drug A interacts with Drug B. Drug C is not combined with Drug D.",
         [("Drug A", DRUG), ("Drug B", DRUG)],
         "Drug A", "INTERACTS_WITH", "Drug B", False),
        ("Drug A interacts with Drug B. Drug C is not combined with Drug D.",
         [("Drug C", DRUG), ("Drug D", DRUG)],
         "Drug C", "INTERACTS_WITH", "Drug D", True),
        ("Noted that Drug A treats disease X well.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Drug A manages disease X without difficulty.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Drug B is not indicated for disease Z, yet Drug A treats disease X.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Although Drug B is not recommended for disease Z, Drug A treats disease X.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Drug B never treats disease Z; nevertheless Drug A treats disease X.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "TREATS", "disease X", False),
        ("Drug A is not indicated for disease X and Drug B is indicated for disease Y.",
         [("Drug A", DRUG), ("disease X", DIS)],
         "Drug A", "INDICATED_FOR", "disease X", True),
    ]
    assert len(cases) >= 20
    for text, specs, head, relation, tail, expect in cases:
        _check(text, specs, head, relation, tail, expect)
