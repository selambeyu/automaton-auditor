"""StateGraph: detectives in parallel (fan-out), EvidenceAggregator (fan-in),
then Judges in parallel (fan-out), Chief Justice (fan-in), END.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

# Load .env before any LangChain/LangGraph import so LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY are set
# (LangSmith requires these to be in os.environ before the first langchain import)
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.graph import END, StateGraph

from src.state import AgentState
from src.nodes.detectives import (
    doc_analyst_node,
    evidence_aggregator_node,
    repo_investigator_node,
    vision_inspector_node,
)
from src.nodes.judges import defense_node, prosecutor_node, tech_lead_node
from src.nodes.justice import chief_justice_node


def _load_rubric(rubric_path: str | Path | None = None) -> list:
    """Load rubric dimensions only (backward compatible)."""
    full = _load_rubric_full(rubric_path)
    return full.get("dimensions", [])


def _load_rubric_full(rubric_path: str | Path | None = None) -> Dict[str, Any]:
    """Load full rubric (dimensions + synthesis_rules) from path. Dynamic: change path to use a different rubric."""
    path = Path(rubric_path or __file__).resolve().parent.parent / "rubric.json"
    if not path.exists():
        return {"dimensions": [], "synthesis_rules": {}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "dimensions": data.get("dimensions", []),
        "synthesis_rules": data.get("synthesis_rules", {}),
    }


def _context_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context Builder: load rubric via json and distribute into agent context.
    Iterates through the dimensions array and places them (and synthesis_rules)
    into state so detectives receive dimensions filtered by target_artifact
    and Chief Justice receives synthesis_rules. Allows updating the
    "Constitution" centrally via rubric.json without redeploying agent code.
    """
    if state.get("rubric_dimensions"):
        return {}
    full = _load_rubric_full(None)
    dimensions = full.get("dimensions", [])
    synthesis_rules = full.get("synthesis_rules", {})
    return {
        "rubric_dimensions": dimensions,
        "rubric_synthesis_rules": synthesis_rules,
    }


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


def _route_vision(state: Dict[str, Any]) -> Literal["run", "skip"]:
    """Route to VisionInspector only when pdf_path is present (same as DocAnalyst)."""
    pdf_path = (state.get("pdf_path") or "").strip()
    return "run" if pdf_path else "skip"


def _skip_vision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Short-circuit: no pdf_path; empty evidence for vision branch."""
    return {
        "evidences": {"vision_inspector": []},
        "detector_fatal_errors": {"vision_inspector": "pdf_path missing or empty; skipping VisionInspector."},
    }


def build_detective_graph():
    """
    Build the full graph: detectives (fan-out/fan-in) -> judges (fan-out/fan-in) -> chief_justice -> END.

    - Context Builder -> [repo_gate, doc_gate, vision_gate] (fan-out)
    - repo_gate: conditional -> RepoInvestigator or skip_repo
    - doc_gate: conditional -> DocAnalyst or skip_doc
    - vision_gate: conditional -> VisionInspector or skip_vision
    - All branches -> EvidenceAggregator (fan-in)
    - EvidenceAggregator -> [prosecutor, defense, tech_lead] (fan-out)
    - prosecutor, defense, tech_lead -> chief_justice (fan-in)
    - chief_justice -> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("context_builder", _context_builder_node)
    builder.add_node("repo_gate", lambda s: {})
    builder.add_node("doc_gate", lambda s: {})
    builder.add_node("vision_gate", lambda s: {})
    builder.add_node("skip_repo", _skip_repo_node)
    builder.add_node("skip_doc", _skip_doc_node)
    builder.add_node("skip_vision", _skip_vision_node)
    builder.add_node("repo_investigator", repo_investigator_node)
    builder.add_node("doc_analyst", doc_analyst_node)
    builder.add_node("vision_inspector", vision_inspector_node)
    builder.add_node("evidence_aggregator", evidence_aggregator_node)
    builder.add_node("prosecutor", prosecutor_node)
    builder.add_node("defense", defense_node)
    builder.add_node("tech_lead", tech_lead_node)
    builder.add_node("chief_justice", chief_justice_node)

    builder.set_entry_point("context_builder")

    builder.add_edge("context_builder", "repo_gate")
    builder.add_edge("context_builder", "doc_gate")
    builder.add_edge("context_builder", "vision_gate")

    builder.add_conditional_edges("repo_gate", _route_repo, {"run": "repo_investigator", "skip": "skip_repo"})
    builder.add_conditional_edges("doc_gate", _route_doc, {"run": "doc_analyst", "skip": "skip_doc"})
    builder.add_conditional_edges("vision_gate", _route_vision, {"run": "vision_inspector", "skip": "skip_vision"})

    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("skip_repo", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")
    builder.add_edge("skip_doc", "evidence_aggregator")
    builder.add_edge("vision_inspector", "evidence_aggregator")
    builder.add_edge("skip_vision", "evidence_aggregator")

    # Fan-out: aggregator -> all three judges
    builder.add_edge("evidence_aggregator", "prosecutor")
    builder.add_edge("evidence_aggregator", "defense")
    builder.add_edge("evidence_aggregator", "tech_lead")

    # Fan-in: all judges -> chief_justice -> END
    builder.add_edge("prosecutor", "chief_justice")
    builder.add_edge("defense", "chief_justice")
    builder.add_edge("tech_lead", "chief_justice")
    builder.add_edge("chief_justice", END)

    return builder.compile()


def run_audit(
    repo_url: str,
    pdf_path: str,
    rubric_path: str | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    """
    Run the detective graph and return final state. Optional rubric_path.
    If run_id is provided (or generated), it is used for LangSmith tracing so you can
    link to the trace. The run_id is attached to the result as _run_id.
    """
    import uuid

    graph = build_detective_graph()
    full_rubric = _load_rubric_full(rubric_path)
    initial: Dict[str, Any] = {
        "repo_url": repo_url,
        "pdf_path": pdf_path,
        "rubric_dimensions": full_rubric["dimensions"],
        "rubric_synthesis_rules": full_rubric["synthesis_rules"],
        "evidences": {},
        "opinions": [],
        "detector_fatal_errors": {},
    }

    rid = run_id or str(uuid.uuid4())
    config: Dict[str, Any] = {"run_id": rid}
    result = graph.invoke(initial, config=config)
    result["_run_id"] = rid
    return result
