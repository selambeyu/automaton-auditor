"""Partial StateGraph: detectives in parallel (fan-out) and EvidenceAggregator (fan-in).

Conditional routing: skip RepoInvestigator when repo_url is missing; skip DocAnalyst
when pdf_path is missing. Fatal errors during detective runs are recorded in
detector_fatal_errors and handled by the aggregator. After aggregation, a
judicial layer placeholder runs before END (Judges + Chief Justice will attach here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

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


def _route_repo(state: Dict[str, Any]) -> Literal["run", "skip"]:
    """Route to RepoInvestigator only when repo_url is present and non-empty."""
    repo_url = (state.get("repo_url") or "").strip()
    return "run" if repo_url else "skip"


def _route_doc(state: Dict[str, Any]) -> Literal["run", "skip"]:
    """Route to DocAnalyst only when pdf_path is present and non-empty."""
    pdf_path = (state.get("pdf_path") or "").strip()
    return "run" if pdf_path else "skip"


def _skip_repo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Short-circuit: no repo_url; record fatal error and empty evidence for repo branch."""
    return {
        "evidences": {"repo_investigator": []},
        "detector_fatal_errors": {"repo_investigator": "repo_url missing or empty; skipping RepoInvestigator."},
    }


def _skip_doc_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Short-circuit: no pdf_path; record fatal error and empty evidence for doc branch."""
    return {
        "evidences": {"doc_analyst": []},
        "detector_fatal_errors": {"doc_analyst": "pdf_path missing or empty; skipping DocAnalyst."},
    }


def _judicial_placeholder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for the judicial layer. Runs after EvidenceAggregator; state is ready
    for Judges (Prosecutor, Defense, TechLead) and Chief Justice synthesis.
    No-op for interim: optional placeholder message in state for documentation.
    """
    # Leave state unchanged; judicial layer will consume state["evidences"] and write opinions + final_report
    return {}


def build_detective_graph():
    """
    Build the partial graph with conditional routing and judicial placeholder:

    - Entry -> [repo_gate, doc_gate] (fan-out)
    - repo_gate: conditional -> RepoInvestigator (if repo_url) else skip_repo
    - doc_gate: conditional -> DocAnalyst (if pdf_path) else skip_doc
    - All four branches -> EvidenceAggregator (fan-in)
    - EvidenceAggregator -> judicial_placeholder -> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("entry", _entry_node)
    builder.add_node("repo_gate", lambda s: {})  # pass-through for routing only
    builder.add_node("doc_gate", lambda s: {})
    builder.add_node("skip_repo", _skip_repo_node)
    builder.add_node("skip_doc", _skip_doc_node)
    builder.add_node("repo_investigator", repo_investigator_node)
    builder.add_node("doc_analyst", doc_analyst_node)
    builder.add_node("evidence_aggregator", evidence_aggregator_node)
    builder.add_node("judicial_placeholder", _judicial_placeholder_node)

    builder.set_entry_point("entry")

    # Fan-out: entry -> both gates
    builder.add_edge("entry", "repo_gate")
    builder.add_edge("entry", "doc_gate")

    # Conditional edges: skip detectives when input is missing (short-circuit on fatal config)
    builder.add_conditional_edges("repo_gate", _route_repo, {"run": "repo_investigator", "skip": "skip_repo"})
    builder.add_conditional_edges("doc_gate", _route_doc, {"run": "doc_analyst", "skip": "skip_doc"})

    # Fan-in: all branches -> aggregator
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("skip_repo", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")
    builder.add_edge("skip_doc", "evidence_aggregator")

    # After aggregation: judicial layer placeholder, then END
    builder.add_edge("evidence_aggregator", "judicial_placeholder")
    builder.add_edge("judicial_placeholder", END)

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
        "detector_fatal_errors": {},
    }
    result = graph.invoke(initial)
    return result
