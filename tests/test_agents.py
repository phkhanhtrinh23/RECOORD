
import pytest

from recoord.agents import Rule, build_agent, parse_rules
from recoord.datasets import load
from recoord.federation import Usage
from recoord.llm import Reply, _parse
from recoord.model import Cell


class StubModel:
    def __init__(self, values, confidence=0.77):
        self.values = values
        self.confidence = confidence
        self.prompts = []

    def repair(self, rule, columns, rows, answers, instruction=""):
        self.prompts.append({"rule": rule, "columns": list(columns), "rows": list(rows),
                             "answers": answers, "instruction": instruction})
        return Reply(list(self.values), self.confidence, Usage(1, 120, 8))


JUDGEMENT = Rule(name="match-brand", target="brand", sources=("title", "brand"),
                 when="always", score=0.6, role="matcher")


def catalog_view():
    dataset = load("catalog", seed=0, limit=20)
    return dataset.dirty


def test_llm_agent_turns_a_reply_into_a_proposal():
    view = catalog_view()
    model = StubModel(["BrandZ"])
    agent = build_agent("llm", "pool", [JUDGEMENT], client=model)
    proposals = agent.propose(view, "pool")
    assert proposals
    first = proposals[0]
    assert first.patch[Cell(first.writes[0].cell.row, "brand")] == "BrandZ"
    assert first.score == pytest.approx(0.77)
    assert not first.replayable


def test_llm_agent_sees_the_evidence_answers():
    model = StubModel(["BrandZ"])
    build_agent("llm", "pool", [JUDGEMENT], client=model).propose(catalog_view(), "pool")
    sent = model.prompts[0]
    assert sent["rule"] == "match-brand"
    assert sent["columns"] == ["brand"]
    assert sent["answers"] and sent["answers"][0]


def test_llm_agent_reports_token_usage():
    model = StubModel(["BrandZ"])
    agent = build_agent("llm", "pool", [JUDGEMENT], client=model)
    agent.propose(catalog_view(), "pool")
    assert agent.usage.input_tokens > 0 and agent.usage.output_tokens > 0


def test_an_empty_reply_proposes_nothing():
    agent = build_agent("llm", "pool", [JUDGEMENT], client=StubModel([]))
    assert agent.propose(catalog_view(), "pool") == []


def test_a_regeneration_asks_the_model_again():
    model = StubModel(["BrandZ"])
    agent = build_agent("llm", "pool", [JUDGEMENT], client=model)
    proposal = agent.propose(catalog_view(), "pool")[0]
    asked, view = len(model.prompts), catalog_view()
    again = agent.regenerate(proposal, view, view.evidence(proposal.queries).answers)
    assert again is not None
    assert len(model.prompts) == asked + 1
    assert again.lineage == proposal.lineage


def test_reply_parsing_tolerates_surrounding_text():
    payload = _parse('sure, here:\n{"values": ["x"], "confidence": 0.5}\nthanks')
    assert payload == {"values": ["x"], "confidence": 0.5}
    assert _parse("no json here") == {"values": [], "confidence": 0.0}


def test_the_rule_library_drops_judgements_and_run_time_rules():
    rules = parse_rules([
        {"name": "p", "target": "a", "program": "copy", "sources": ["b"]},
        {"name": "q", "target": "a", "program": "copy", "sources": ["b"], "invented": True},
        {"name": "j", "target": "a", "sources": ["b"]},
    ])
    agent = build_agent("rules", "c", rules)
    assert [r.name for r in agent.rules] == ["p"]


def test_offline_and_llm_agents_emit_the_same_shape():
    view = catalog_view()
    offline = build_agent("offline", "pool", [JUDGEMENT], reference=view, accuracy=0.0)
    llm = build_agent("llm", "pool", [JUDGEMENT], client=StubModel(["BrandZ"]))
    a = {p.id: p for p in offline.propose(view, "pool")}
    b = {p.id: p for p in llm.propose(view, "pool")}
    shared = set(a) & set(b)
    assert shared
    for pid in shared:
        assert a[pid].queries[0].sql == b[pid].queries[0].sql
        assert a[pid].write_cells == b[pid].write_cells
        assert a[pid].replayable == b[pid].replayable == False


def test_an_unknown_agent_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown agent kind"):
        build_agent("psychic", "c", [])
