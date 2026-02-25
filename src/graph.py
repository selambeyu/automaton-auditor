"""Partial StateGraph: detectives in parallel (fan-out) and EvidenceAggregator (fan-in)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.state import AgentState
from src.nodes.detectives import (
    doc_analyst_node,
    evidence_aggregator_node,
    repo_investigator_node,
)


def _load_rubric(rubric_path: str | Path | None = None) -> list:
    path = Path(rubric_path or __file__).resolve().parent.parent / "rubric.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("dimensions", [])


def _entry_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure rubric_dimensions are set from rubric.json if not provided."""
    if state.get("rubric_dimensions"):
        return {}
    return {"rubric_dimensions": _load_rubric()}


def build_detective_graph():
    """
    Build the partial graph: entry -> [RepoInvestigator, DocAnalyst] (parallel)
    -> EvidenceAggregator -> END.
    """
    builder = StateGraph(AgentState)

    builder.add_node("entry", _entry_node)
    builder.add_node("repo_investigator", repo_investigator_node)
    builder.add_node("doc_analyst", doc_analyst_node)
    builder.add_node("evidence_aggregator", evidence_aggregator_node)

    builder.set_entry_point("entry")

    # Fan-out: entry -> both detectives
    builder.add_edge("entry", "repo_investigator")
    builder.add_edge("entry", "doc_analyst")

    # Fan-in: both detectives -> aggregator
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")

    builder.add_edge("evidence_aggregator", END)

    return builder.compile()


def run_audit(repo_url: str, pdf_path: str, rubric_path: str | None = None) -> Dict[str, Any]:
    """
    Run the detective graph and return final state. Optional rubric_path.
    """
    graph = build_detective_graph()
    initial: Dict[str, Any] = {
        "repo_url": repo_url,
        "pdf_path": pdf_path,
        "rubric_dimensions": _load_rubric(rubric_path),
        "evidences": {},
        "opinions": [],
    }
    result = graph.invoke(initial)
    return result
