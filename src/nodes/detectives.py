"""Detective layer: RepoInvestigator, DocAnalyst, and VisionInspector nodes. Output structured Evidence only.

Optional LLM use (fact extraction only, no opinions): set AUDITOR_DETECTIVE_LLM=1 to use
get_detective_llm() (or get_llm()) in DocAnalyst for classifying theoretical depth.
VisionInspector uses get_detective_llm() when DETECTIVE_PROVIDER=gemini, else get_vision_llm().
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List

from src.state import Evidence
from src.tools.doc_tools import (
    extract_file_paths_from_chunks,
    extract_images_from_pdf,
    ingest_pdf,
    query_document,
)
from src.tools.repo_tools import (
    analyze_graph_structure,
    analyze_state_structure,
    clone_repo_sandboxed,
    extract_git_history,
    list_src_files,
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

            # Git forensic analysis (rubric: commit messages and timestamps)
            dim = next((d for d in repo_dims if d.get("id") == "git_forensic_analysis"), None)
            if dim:
                found = len(commits) > 3
                lines = [
                    f"{c.hash} {c.message}" + (f" {c.timestamp}" if c.timestamp else "")
                    for c in commits[:20]
                ]
                content = "\n".join(lines)
                if len(commits) > 20:
                    content += "\n..."
                fi = (dim.get("forensic_instruction") or "")[:280]
                rationale = f"Found {len(commits)} commits. Success pattern: >3 with progression."
                if fi:
                    rationale = f"Forensic instruction: {fi}... {rationale}"
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "Git Forensic Analysis"),
                        found=found,
                        content=content or None,
                        location=str(path),
                        rationale=rationale,
                        confidence=0.9 if found else 0.3,
                    )
                )

            # State management rigor (rubric: AST for BaseModel, TypedDict, Evidence, JudicialOpinion, reducers)
            state_struct = analyze_state_structure(path)
            state_file = path / "src" / "state.py"
            has_state = state_file.exists()
            rigor_ok = (
                has_state
                and (state_struct.has_base_model or state_struct.has_typed_dict)
                and state_struct.has_evidence_class
                and state_struct.has_judicial_opinion_class
                and state_struct.has_operator_add
                and state_struct.has_operator_ior
            )
            dim = next((d for d in repo_dims if d.get("id") == "state_management_rigor"), None)
            if dim:
                fi = (dim.get("forensic_instruction") or "")[:280]
                rationale = f"AST: BaseModel={state_struct.has_base_model}, TypedDict={state_struct.has_typed_dict}, Evidence={state_struct.has_evidence_class}, JudicialOpinion={state_struct.has_judicial_opinion_class}, operator.add={state_struct.has_operator_add}, operator.ior={state_struct.has_operator_ior}"
                if fi:
                    rationale = f"Forensic instruction: {fi}... {rationale}"
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "State Management Rigor"),
                        found=rigor_ok,
                        content=state_struct.snippet if has_state else None,
                        location="src/state.py",
                        rationale=rationale,
                        confidence=0.95 if rigor_ok else (0.4 if has_state else 0.0),
                    )
                )

            # Repo file list for cross-reference (Report Accuracy)
            src_files = list_src_files(path)
            evidences.append(
                Evidence(
                    goal="Repo file list",
                    found=len(src_files) > 0,
                    content="\n".join(src_files) if src_files else None,
                    location="src/",
                    rationale=f"Files under src/: {len(src_files)}",
                    confidence=0.9 if src_files else 0.0,
                )
            )

            # Graph orchestration
            dim = next((d for d in repo_dims if d.get("id") == "graph_orchestration"), None)
            if dim:
                found = graph_struct.has_state_graph and (
                    graph_struct.add_edge_calls >= 2 or graph_struct.add_conditional_edges_calls >= 1
                )
                fi = (dim.get("forensic_instruction") or "")[:280]
                rationale = f"StateGraph={graph_struct.has_state_graph}, add_edge={graph_struct.add_edge_calls}, add_conditional_edges={graph_struct.add_conditional_edges_calls}, nodes={graph_struct.node_names}"
                if fi:
                    rationale = f"Forensic instruction: {fi}... {rationale}"
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "Graph Orchestration"),
                        found=found,
                        content=graph_struct.snippet[:2000] if graph_struct.snippet else None,
                        location="src/graph.py",
                        rationale=rationale,
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
                fi = (dim.get("forensic_instruction") or "")[:280]
                rationale = f"tempfile/TemporaryDirectory={has_tempfile}, subprocess without os.system={has_subprocess}"
                if fi:
                    rationale = f"Forensic instruction: {fi}... {rationale}"
                evidences.append(
                    Evidence(
                        goal=dim.get("name", "Safe Tool Engineering"),
                        found=found,
                        content=rt[:1500] if found else None,
                        location="src/tools/repo_tools.py",
                        rationale=rationale,
                        confidence=0.85 if found else 0.2,
                    )
                )

            # Structured output enforcement (rubric: judges.py .with_structured_output / .bind_tools + JudicialOpinion)
            judges_file = path / "src" / "nodes" / "judges.py"
            dim = next((d for d in repo_dims if d.get("id") == "structured_output_enforcement"), None)
            if dim:
                if judges_file.exists():
                    jc = judges_file.read_text(encoding="utf-8", errors="replace")
                    has_structured = (
                        "with_structured_output" in jc or "bind_tools" in jc
                    ) and "JudicialOpinion" in jc
                    fi = (dim.get("forensic_instruction") or "")[:280]
                    rationale = f"with_structured_output/bind_tools and JudicialOpinion present: {has_structured}"
                    if fi:
                        rationale = f"Forensic instruction: {fi}... {rationale}"
                    evidences.append(
                        Evidence(
                            goal=dim.get("name", "Structured Output Enforcement"),
                            found=has_structured,
                            content=jc[:1500] if has_structured else jc[:800],
                            location="src/nodes/judges.py",
                            rationale=rationale,
                            confidence=0.85 if has_structured else 0.3,
                        )
                    )
                else:
                    fi = (dim.get("forensic_instruction") or "")[:200]
                    rationale = "File not present (interim; judges not required yet)."
                    if fi:
                        rationale = f"Forensic instruction: {fi}... {rationale}"
                    evidences.append(
                        Evidence(
                            goal=dim.get("name", "Structured Output Enforcement"),
                            found=False,
                            content=None,
                            location="src/nodes/judges.py",
                            rationale=rationale,
                            confidence=0.0,
                        )
                    )

            # Chief Justice synthesis engine: check deterministic rules + Markdown report output
            cj_file = path / "src" / "nodes" / "justice.py"
            dim = next((d for d in repo_dims if d.get("id") == "chief_justice_synthesis"), None)
            if dim:
                if cj_file.exists():
                    cj = cj_file.read_text(encoding="utf-8", errors="replace")
                    has_deterministic = "def _resolve_score" in cj and ("if " in cj or "elif " in cj)
                    has_security_rule = "Rule of Security" in cj or "security" in cj.lower()
                    has_evidence_rule = "Rule of Evidence" in cj or "fact supremacy" in cj.lower() or "_get_evidence_for_criterion" in cj
                    has_functionality_rule = "Rule of Functionality" in cj or ("criterion_id == \"graph_orchestration\"" in cj)
                    has_variance = "variance > 2" in cj or "score variance" in cj.lower() or "variance" in cj.lower()
                    writes_markdown = "_report_to_markdown" in cj and "_write_report_markdown" in cj and "write_text" in cj

                    found = bool(
                        has_deterministic
                        and has_security_rule
                        and has_evidence_rule
                        and has_functionality_rule
                        and has_variance
                        and writes_markdown
                    )
                    fi = (dim.get("forensic_instruction") or "")[:280]
                    rationale = (
                        "justice.py checks: "
                        f"deterministic_rules={has_deterministic}, "
                        f"security_rule={has_security_rule}, "
                        f"evidence_rule={has_evidence_rule}, "
                        f"functionality_rule={has_functionality_rule}, "
                        f"variance_dissent={has_variance}, "
                        f"markdown_output={writes_markdown}"
                    )
                    if fi:
                        rationale = f"Forensic instruction: {fi}... {rationale}"
                    evidences.append(
                        Evidence(
                            goal=dim.get("name", "Chief Justice Synthesis Engine"),
                            found=found,
                            content=cj[:2000],
                            location="src/nodes/justice.py",
                            rationale=rationale,
                            confidence=0.9 if found else 0.35,
                        )
                    )
                else:
                    fi = (dim.get("forensic_instruction") or "")[:200]
                    rationale = "File not present (Chief Justice node missing)."
                    if fi:
                        rationale = f"Forensic instruction: {fi}... {rationale}"
                    evidences.append(
                        Evidence(
                            goal=dim.get("name", "Chief Justice Synthesis Engine"),
                            found=False,
                            content=None,
                            location="src/nodes/justice.py",
                            rationale=rationale,
                            confidence=0.0,
                        )
                    )

            # Dynamic rubric: for any repo dimension we don't have specific logic for, add one generic evidence
            # so judges and Chief Justice can still score it when you add new dimensions to rubric.json
            goals_done = {e.goal for e in evidences}
            for dim in repo_dims:
                name = dim.get("name") or dim.get("id") or ""
                if name and name not in goals_done:
                    evidences.append(
                        Evidence(
                            goal=name,
                            found=False,
                            content=f"Repo cloned; no specific collector for dimension '{name}'. forensic_instruction: {dim.get('forensic_instruction', '')[:200]}...",
                            location=str(path),
                            rationale="Generic fallback for rubric dimension without dedicated collector.",
                            confidence=0.3,
                        )
                    )
    except Exception as e:
        err_msg = str(e)
        evidences.append(
            Evidence(
                goal="RepoInvestigator",
                found=False,
                content=None,
                location=repo_url,
                rationale=err_msg,
                confidence=0.0,
            )
        )
        return {
            "evidences": {"repo_investigator": evidences},
            "detector_fatal_errors": {"repo_investigator": err_msg},
        }

    return {"evidences": {"repo_investigator": evidences}}


def _llm_classify_theoretical_depth(snippet: str, max_chars: int = 2500) -> str | None:
    """
    Use the shared LLM for fact-only classification: does the report text contain
    substantive explanation of orchestration concepts or only keyword mention?
    Returns a short factual summary string, or None on error. No opinions or scoring.
    """
    if not (snippet and snippet.strip()):
        return None
    text = snippet[:max_chars] if len(snippet) > max_chars else snippet
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.llm import get_detective_llm

        llm = get_detective_llm()
        system = (
            "You are a forensic document analyst. You only extract facts. Do not give opinions or scores. "
            "Classify whether the given report excerpt contains: (1) SUBSTANTIVE_EXPLANATION — the text explains "
            "how concepts like Dialectical Synthesis, Fan-In/Fan-Out, or Metacognition are implemented or used; "
            "or (2) KEYWORD_ONLY — the terms appear but without real explanation. "
            "Reply with exactly one line: SUBSTANTIVE_EXPLANATION or KEYWORD_ONLY, then a space, then a short "
            "quote from the text (max 200 chars) that supports your classification. No other commentary."
        )
        msg = HumanMessage(
            content=f"Report excerpt:\n\n{text}\n\nClassify (one line: SUBSTANTIVE_EXPLANATION or KEYWORD_ONLY + short quote):"
        )
        response = llm.invoke([SystemMessage(content=system), msg])
        out = response.content if hasattr(response, "content") else str(response)
        return (out or "").strip() or None
    except Exception:
        return None


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
        return {
            "evidences": {"doc_analyst": evidences},
            "detector_fatal_errors": {"doc_analyst": "Could not ingest PDF or file missing"},
        }

    # Theoretical depth: search for key terms
    keywords = [
        "Dialectical Synthesis",
        "Fan-In",
        "Fan-Out",
        "Fan-In / Fan-Out",
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
        rationale = "Searched for orchestration and metacognition terms in report."
        fi = (dim.get("forensic_instruction") or "")[:300]
        if fi:
            rationale = f"Forensic instruction: {fi}... {rationale}"
        content = snippet or None
        confidence = 0.7 if found else 0.2
        # Optional: use LLM for factual classification (substantive vs keyword-only)
        if os.environ.get("AUDITOR_DETECTIVE_LLM", "").strip().lower() in ("1", "true", "yes") and snippet:
            llm_class = _llm_classify_theoretical_depth(snippet)
            if llm_class:
                rationale = f"{rationale} LLM classification: {llm_class}"
                if "SUBSTANTIVE" in llm_class.upper():
                    confidence = 0.85
                elif "KEYWORD_ONLY" in llm_class.upper():
                    confidence = 0.4
        evidences.append(
            Evidence(
                goal=dim.get("name", "Theoretical Depth"),
                found=found,
                content=content,
                location=pdf_path,
                rationale=rationale,
                confidence=confidence,
            )
        )

    # Report accuracy: extract file paths mentioned in report for cross-ref in EvidenceAggregator
    paths_mentioned = extract_file_paths_from_chunks(chunks)
    dim = next((d for d in pdf_dims if d.get("id") == "report_accuracy"), None)
    if dim:
        paths_content = "Paths mentioned in report:\n" + "\n".join(paths_mentioned) if paths_mentioned else "No paths extracted."
        fi = (dim.get("forensic_instruction") or "")[:300]
        rationale = f"Extracted {len(paths_mentioned)} path(s) from report; cross-reference in EvidenceAggregator."
        if fi:
            rationale = f"Forensic instruction: {fi}... {rationale}"
        evidences.append(
            Evidence(
                goal=dim.get("name", "Report Accuracy"),
                found=True,
                content=paths_content + "\n\n" + (snippet[:1500] if snippet else ""),
                location=pdf_path,
                rationale=rationale,
                confidence=0.6,
            )
        )

    # Dynamic rubric: for any pdf_report dimension we don't have specific logic for, add one generic evidence
    goals_done = {e.goal for e in evidences}
    for dim in pdf_dims:
        name = dim.get("name") or dim.get("id") or ""
        if name and name not in goals_done:
            evidences.append(
                Evidence(
                    goal=name,
                    found=bool(snippet),
                    content=(snippet[:1500] if snippet else None) or f"PDF ingested. forensic_instruction: {dim.get('forensic_instruction', '')[:200]}...",
                    location=pdf_path,
                    rationale="Generic fallback for rubric dimension without dedicated collector.",
                    confidence=0.4,
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


def vision_inspector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagram Detective: extract images from the PDF and optionally run vision model
    for "Architectural Diagram Analysis" (rubric: swarm_visual). Execution optional:
    set AUDITOR_RUN_VISION=1 to run; otherwise returns Evidence with found=False.
    """
    pdf_path = (state.get("pdf_path") or "").strip()
    rubric_dimensions: List[Dict[str, Any]] = state.get("rubric_dimensions") or []
    dim = next(
        (d for d in rubric_dimensions if d.get("target_artifact") == "pdf_images"),
        None,
    )
    goal_name = dim.get("name", "Architectural Diagram Analysis") if dim else "Architectural Diagram Analysis"
    evidences: List[Evidence] = []

    if not pdf_path:
        evidences.append(
            Evidence(
                goal=goal_name,
                found=False,
                content=None,
                location="",
                rationale="pdf_path missing; VisionInspector skipped.",
                confidence=0.0,
            )
        )
        return {"evidences": {"vision_inspector": evidences}}

    images = extract_images_from_pdf(pdf_path)
    run_vision = os.environ.get("AUDITOR_RUN_VISION", "0").strip().lower() in ("1", "true", "yes")

    if not images or not run_vision:
        evidences.append(
            Evidence(
                goal=goal_name,
                found=False,
                content=None,
                location=pdf_path,
                rationale="Vision inspection skipped (AUDITOR_RUN_VISION not set) or no images extracted from PDF."
                if not run_vision
                else "No images extracted from PDF.",
                confidence=0.0,
            )
        )
        return {"evidences": {"vision_inspector": evidences}}

    # Call vision-capable model: multi-model stack uses get_detective_llm() (Gemini), else get_vision_llm()
    try:
        from langchain_core.messages import HumanMessage

        from src.llm import get_detective_llm, get_detective_provider, get_vision_llm, get_vision_provider
    except ImportError:
        evidences.append(
            Evidence(
                goal=goal_name,
                found=False,
                content=None,
                location=pdf_path,
                rationale="Vision model not available (import error).",
                confidence=0.0,
            )
        )
        return {"evidences": {"vision_inspector": evidences}}

    if get_detective_provider() == "gemini":
        llm = get_detective_llm()
    else:
        llm = get_vision_llm()
    if llm is None:
        evidences.append(
            Evidence(
                goal=goal_name,
                found=False,
                content=None,
                location=pdf_path,
                rationale="Vision model not configured (set API key or OLLAMA_VISION_MODEL).",
                confidence=0.0,
            )
        )
        return {"evidences": {"vision_inspector": evidences}}

    # Send forensic_instruction to VisionInspector so it evaluates against the rubric
    forensic_instruction = (dim.get("forensic_instruction") or "").strip() if dim else ""
    prompt = (
        "Classify this diagram: Is it an accurate LangGraph State Machine diagram, a sequence diagram, or generic flowchart? "
        "Does it show parallel split: Detectives in parallel -> Evidence Aggregation -> Judges in parallel -> Chief Justice? "
        "Reply in one short paragraph: type (StateGraph / sequence / generic), then whether parallel flow is shown (yes/no)."
    )
    if forensic_instruction:
        prompt = f"Rubric forensic instruction for this dimension:\n{forensic_instruction}\n\nYour task: {prompt}"
    img = images[0]
    img_bytes = img.get("bytes") or b""
    fmt = (img.get("format") or "png").lower()
    mime = "image/png" if fmt == "png" else "image/jpeg" if fmt in ("jpg", "jpeg") else "image/png"
    b64 = base64.b64encode(img_bytes).decode("utf-8") if img_bytes else ""

    if not b64:
        evidences.append(
            Evidence(
                goal=goal_name,
                found=False,
                content=None,
                location=pdf_path,
                rationale="Image bytes empty.",
                confidence=0.0,
            )
        )
        return {"evidences": {"vision_inspector": evidences}}

    # Gemini expects {"type": "image", "base64", "mime_type"}; others use image_url with data URL
    vision_provider = get_detective_provider() or get_vision_provider()
    if vision_provider == "gemini":
        image_part = {"type": "image", "base64": b64, "mime_type": mime}
    else:
        image_part = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    try:
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                image_part,
            ],
        )
        response = llm.invoke([msg])
        # Gemini can return content as list of blocks; prefer .text when available
        content = getattr(response, "text", None) or response.content
        if isinstance(content, list):
            parts = (b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            content = (" ".join(parts).strip()) or str(response)
        else:
            content = content or str(response)
        # Heuristic: if response suggests StateGraph and parallel, found=True
        content_lower = (content or "").lower()
        found = (
            ("stategraph" in content_lower or "parallel" in content_lower)
            and ("generic" not in content_lower or "parallel" in content_lower)
        )
        evidences.append(
            Evidence(
                goal=goal_name,
                found=found,
                content=content[:2000] if content else None,
                location=pdf_path,
                rationale="Vision model classified diagram.",
                confidence=0.7 if found else 0.3,
            )
        )
    except Exception as e:
        evidences.append(
            Evidence(
                goal=goal_name,
                found=False,
                content=None,
                location=pdf_path,
                rationale=f"Vision model error: {e!s}"[:200],
                confidence=0.0,
            )
        )

    return {"evidences": {"vision_inspector": evidences}}


def _get_goal(item: Any) -> str:
    return item.get("goal", "") if isinstance(item, dict) else getattr(item, "goal", "")


def _get_content(item: Any) -> str | None:
    return item.get("content") if isinstance(item, dict) else getattr(item, "content", None)


def _get_confidence(item: Any) -> float:
    return item.get("confidence", 0.0) if isinstance(item, dict) else getattr(item, "confidence", 0.0)


def evidence_aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fan-in: run after all detectives. Reconciles and normalizes evidence across
    detectives; cross-references report paths with repo file list (rubric:
    Verified Paths vs Hallucinated Paths). Handles detector_fatal_errors for
    partial runs. Leaves state ready for the judicial layer (Judges + Chief Justice).
    """
    evidences = state.get("evidences") or {}
    detector_fatal_errors = state.get("detector_fatal_errors") or {}
    agg_evidences: List[Evidence] = []

    # --- 1. Collect all evidence items and normalize confidence to [0, 1] ---
    all_items: List[Evidence] = []
    for source, items in evidences.items():
        for item in (items or []):
            if isinstance(item, dict):
                item = Evidence(**item)
            conf = _get_confidence(item)
            normalized_conf = max(0.0, min(1.0, float(conf)))
            if normalized_conf != conf and isinstance(item, Evidence):
                item = item.model_copy(update={"confidence": normalized_conf})
            elif isinstance(item, dict):
                item = Evidence(**(item | {"confidence": normalized_conf}))
            all_items.append(item)

    # --- 2. Reconcile by dimension/goal: one evidence per goal, merged from all detectives ---
    by_goal: Dict[str, List[Evidence]] = {}
    for item in all_items:
        goal = _get_goal(item)
        if goal not in by_goal:
            by_goal[goal] = []
        by_goal[goal].append(item)

    for goal, items in by_goal.items():
        if len(items) == 1:
            agg_evidences.append(items[0])
            continue
        # Multiple detectives contributed to this goal: take max confidence, merge rationale
        best = max(items, key=lambda x: _get_confidence(x))
        rationales = [getattr(i, "rationale", i.get("rationale", "")) for i in items]
        merged_rationale = " | ".join(rationales) if len(rationales) <= 2 else f"Reconciled from {len(items)} sources: {rationales[0][:200]}..."
        if isinstance(best, Evidence):
            agg_evidences.append(
                best.model_copy(update={"rationale": merged_rationale, "location": "aggregated"})
            )
        else:
            agg_evidences.append(
                Evidence(
                    goal=_get_goal(best),
                    found=best.get("found", False) if isinstance(best, dict) else getattr(best, "found", False),
                    content=_get_content(best),
                    location="aggregated",
                    rationale=merged_rationale,
                    confidence=_get_confidence(best),
                )
            )

    # --- 3. Report Accuracy cross-reference (Verified vs Hallucinated Paths) ---
    repo_list: List[str] = []
    for item in evidences.get("repo_investigator") or []:
        if _get_goal(item) == "Repo file list":
            content = _get_content(item)
            if content:
                repo_list = [line.strip() for line in content.splitlines() if line.strip()]
            break

    paths_mentioned: List[str] = []
    for item in evidences.get("doc_analyst") or []:
        if _get_goal(item) == "Report Accuracy":
            content = _get_content(item)
            if content and "Paths mentioned in report:" in content:
                block = content.split("Paths mentioned in report:")[1].split("\n\n")[0]
                paths_mentioned = [line.strip() for line in block.splitlines() if line.strip()]
            break

    if repo_list or paths_mentioned:
        repo_set = set(repo_list)
        verified = [p for p in paths_mentioned if p in repo_set]
        hallucinated = [p for p in paths_mentioned if p not in repo_set]
        content = (
            "Verified Paths:\n" + ("\n".join(verified) if verified else "(none)")
            + "\n\nHallucinated Paths:\n"
            + ("\n".join(hallucinated) if hallucinated else "(none)")
        )
        agg_evidences.append(
            Evidence(
                goal="Report Accuracy (Cross-Reference)",
                found=len(hallucinated) == 0,
                content=content,
                location="aggregated",
                rationale=f"Verified={len(verified)}, Hallucinated={len(hallucinated)}",
                confidence=0.9 if len(hallucinated) == 0 else 0.3,
            )
        )

    # --- 4. If any detector had a fatal error, add aggregated note for judicial layer ---
    if detector_fatal_errors:
        err_summary = "; ".join(f"{k}: {v[:100]}" for k, v in detector_fatal_errors.items())
        agg_evidences.append(
            Evidence(
                goal="Aggregation (Partial Run)",
                found=False,
                content=err_summary,
                location="aggregated",
                rationale=f"Detector(s) failed; partial evidence only. {err_summary}",
                confidence=0.0,
            )
        )

    if agg_evidences:
        return {"evidences": {"evidence_aggregator": agg_evidences}}
    return {}
