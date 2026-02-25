"""Detective layer: RepoInvestigator and DocAnalyst nodes. Output structured Evidence only."""

from __future__ import annotations

from typing import Any, Dict, List

from src.state import Evidence
from src.tools.doc_tools import ingest_pdf, query_document
from src.tools.repo_tools import (
    analyze_graph_structure,
    clone_repo_sandboxed,
    extract_git_history,
)


def repo_investigator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clone repo, run git history and AST graph analysis. Produce Evidence per
    rubric dimension with target_artifact github_repo. No opinions.
    """
    repo_url = state.get("repo_url") or ""
    rubric_dimensions: List[Dict[str, Any]] = state.get("rubric_dimensions") or []
    repo_dims = [d for d in rubric_dimensions if d.get("target_artifact") == "github_repo"]
    evidences: List[Evidence] = []

    try:
        with clone_repo_sandboxed(repo_url) as path:
            commits = extract_git_history(path)
            graph_struct = analyze_graph_structure(path)

            # Git forensic analysis
            dim = next((d for d in repo_dims if d.get("id") == "git_forensic_analysis"), None)
            if dim:
                found = len(commits) > 3
                content = "\n".join(f"{c.hash} {c.message}" for c in commits[:20])
                if len(commits) > 20:
                    content += "\n..."
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "Git Forensic Analysis"),
                        found=found,
                        content=content or None,
                        location=str(path),
                        rationale=f"Found {len(commits)} commits. Success pattern: >3 with progression.",
                        confidence=0.9 if found else 0.3,
                    )
                )

            # State management rigor: check for src/state.py
            state_file = path / "src" / "state.py"
            has_state = state_file.exists()
            dim = next((d for d in repo_dims if d.get("id") == "state_management_rigor"), None)
            if dim:
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "State Management Rigor"),
                        found=has_state,
                        content=state_file.read_text(encoding="utf-8", errors="replace")[:1500]
                        if has_state
                        else None,
                        location="src/state.py",
                        rationale="File exists and was read" if has_state else "src/state.py not found",
                        confidence=0.95 if has_state else 0.0,
                    )
                )

            # Graph orchestration
            dim = next((d for d in repo_dims if d.get("id") == "graph_orchestration"), None)
            if dim:
                found = graph_struct.has_state_graph and (
                    graph_struct.add_edge_calls >= 2 or graph_struct.add_conditional_edges_calls >= 1
                )
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "Graph Orchestration"),
                        found=found,
                        content=graph_struct.snippet[:2000] if graph_struct.snippet else None,
                        location="src/graph.py",
                        rationale=f"StateGraph={graph_struct.has_state_graph}, add_edge={graph_struct.add_edge_calls}, add_conditional_edges={graph_struct.add_conditional_edges_calls}, nodes={graph_struct.node_names}",
                        confidence=0.8 if found else 0.2,
                    )
                )

            # Safe tool engineering: check tools use tempfile/subprocess
            tools_dir = path / "src" / "tools"
            tools_py = (tools_dir / "repo_tools.py").exists() if tools_dir.exists() else False
            dim = next((d for d in repo_dims if d.get("id") == "safe_tool_engineering"), None)
            if dim and tools_py:
                rt = (path / "src" / "tools" / "repo_tools.py").read_text(
                    encoding="utf-8", errors="replace"
                )
                has_tempfile = "tempfile" in rt or "TemporaryDirectory" in rt
                has_subprocess = "subprocess" in rt and "os.system" not in rt
                found = has_tempfile and has_subprocess
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "Safe Tool Engineering"),
                        found=found,
                        content=rt[:1500] if found else None,
                        location="src/tools/repo_tools.py",
                        rationale=f"tempfile/TemporaryDirectory={has_tempfile}, subprocess without os.system={has_subprocess}",
                        confidence=0.85 if found else 0.2,
                    )
                )
    except Exception as e:
        evidences.append(
            Evidence(
                goal="RepoInvestigator",
                found=False,
                content=None,
                location=repo_url,
                rationale=str(e),
                confidence=0.0,
            )
        )

    return {"evidences": {"repo_investigator": evidences}}


def doc_analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest PDF and query for theoretical depth / report accuracy dimensions.
    Produce Evidence only; no opinions.
    """
    pdf_path = state.get("pdf_path") or ""
    rubric_dimensions: List[Dict[str, Any]] = state.get("rubric_dimensions") or []
    pdf_dims = [d for d in rubric_dimensions if d.get("target_artifact") == "pdf_report"]
    evidences: List[Evidence] = []

    chunks = ingest_pdf(pdf_path)
    if not chunks:
        evidences.append(
            Evidence(
                goal="PDF ingestion",
                found=False,
                content=None,
                location=pdf_path,
                rationale="Could not ingest PDF or file missing",
                confidence=0.0,
            )
        )
        return {"evidences": {"doc_analyst": evidences}}

    # Theoretical depth: search for key terms
    keywords = [
        "Dialectical Synthesis",
        "Fan-In",
        "Fan-Out",
        "Metacognition",
        "State Synchronization",
    ]
    snippet = query_document(
        chunks,
        "Dialectical Synthesis Fan-In Fan-Out Metacognition State Synchronization",
        max_chars=3000,
        keywords=keywords,
    )
    dim = next((d for d in pdf_dims if d.get("id") == "theoretical_depth"), None)
    if dim:
        found = bool(snippet)
        evidences.append(
            Evidence(
                goal=dim.get("name", "Theoretical Depth"),
                found=found,
                content=snippet or None,
                location=pdf_path,
                rationale="Searched for orchestration and metacognition terms in report.",
                confidence=0.7 if found else 0.2,
            )
        )

    # Report accuracy: we would cross-ref with RepoInvestigator; for interim we note paths in text
    dim = next((d for d in pdf_dims if d.get("id") == "report_accuracy"), None)
    if dim:
        evidences.append(
            Evidence(
                goal=dim.get("name", "Report Accuracy"),
                found=True,
                content=snippet[:2000] if snippet else "No content",
                location=pdf_path,
                rationale="DocAnalyst extracted report content; cross-reference with repo evidence is done at aggregation or judicial layer.",
                confidence=0.6,
            )
        )

    if not evidences:
        evidences.append(
            Evidence(
                goal="DocAnalyst",
                found=True,
                content=snippet[:1500] if snippet else None,
                location=pdf_path,
                rationale="PDF ingested; no matching rubric dimensions for pdf_report.",
                confidence=0.5,
            )
        )
    return {"evidences": {"doc_analyst": evidences}}


def evidence_aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fan-in: run after all detectives. Normalize/validate evidences and leave
    state ready for future Judges. No new evidence; just pass through or enrich.
    """
    evidences = state.get("evidences") or {}
    # Optional: flatten or validate; for interim we just ensure it's a dict
    return {}
