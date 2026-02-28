"""Judicial layer: Prosecutor, Defense, and Tech Lead nodes. Output structured JudicialOpinion only."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_judicial_llm, get_llm
from src.state import Evidence, JudicialOpinion


def _collect_evidence_for_dimension(
    state: Dict[str, Any],
    dimension: Dict[str, Any],
) -> List[Evidence]:
    """Gather evidence items relevant to this rubric dimension (match by goal or dimension name/id)."""
    evidences = state.get("evidences") or {}
    dim_id = dimension.get("id", "")
    dim_name = dimension.get("name", "")
    out: List[Evidence] = []

    def goal_matches(goal: str) -> bool:
        if goal == dim_name:
            return True
        # Map common evidence goal names to dimension names/ids
        if dim_id == "report_accuracy" and "Report Accuracy" in goal:
            return True
        if dim_id == "theoretical_depth" and "Theoretical" in goal:
            return True
        if dim_id == "swarm_visual" and ("Diagram" in goal or "Visual" in goal):
            return True
        return False

    for _source, items in evidences.items():
        for item in items or []:
            if isinstance(item, dict):
                item = Evidence(**item)
            goal = getattr(item, "goal", "")
            if goal_matches(goal):
                out.append(item)
    return out


def _evidence_context(evidence_list: List[Evidence], max_content_len: int = 1500) -> str:
    """Build a short context string from evidence for the LLM."""
    parts = []
    for e in evidence_list:
        content = (e.content or "")[:max_content_len]
        if len((e.content or "")) > max_content_len:
            content += "..."
        parts.append(
            f"[{e.goal}] found={e.found} confidence={e.confidence}\n"
            f"rationale: {e.rationale}\n"
            f"content: {content or '(none)'}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No evidence collected for this criterion."


# Persona system prompts (distinct per challenge rubric)
PROSECUTOR_SYSTEM = """You are the Prosecutor in a Digital Courtroom. Your philosophy: "Trust No One. Assume Vibe Coding."
Your job is to scrutinize the evidence for gaps, security flaws, and laziness.
- If the rubric success pattern is NOT met, argue for a low score (1 or 2).
- If you see linear flow instead of parallel orchestration, charge "Orchestration Fraud" and score 1 for architecture.
- If Judge nodes would return freeform text without Pydantic validation, charge "Hallucination Liability" and score at most 2.
- Cite specific evidence in your argument. Output ONLY a valid JudicialOpinion: judge="Prosecutor", criterion_id, score (1-5), argument, cited_evidence (list of short strings)."""

DEFENSE_SYSTEM = """You are the Defense Attorney in a Digital Courtroom. Your philosophy: "Reward Effort and Intent. Look for the Spirit of the Law."
Your job is to highlight creative workarounds, deep thought, and effort even when implementation is imperfect.
- If the code shows partial compliance or good intent (e.g. AST parsing present but graph has a bug), argue for a higher score (3-4).
- If Git history shows iterative development, credit "Engineering Process" and argue for a better score.
- If Judge personas are distinct and dialectical even if synthesis is not fully deterministic, argue for partial credit (3-4).
- Cite specific evidence. Output ONLY a valid JudicialOpinion: judge="Defense", criterion_id, score (1-5), argument, cited_evidence (list of short strings)."""

TECH_LEAD_SYSTEM = """You are the Tech Lead in a Digital Courtroom. Your philosophy: "Does it actually work? Is it maintainable?"
Your job is to evaluate architectural soundness, code cleanliness, and practical viability. You are the tie-breaker.
- Ignore "vibe" and "struggle." Focus on artifacts: Are reducers (operator.add, operator.ior) actually used? Are tool calls safe?
- If the Prosecutor says 1 (security flaw) and Defense says 5 (great effort), assess technical debt and give a realistic score (1, 3, or 5).
- Provide technical remediation in your argument when score is low.
- Output ONLY a valid JudicialOpinion: judge="TechLead", criterion_id, score (1-5), argument, cited_evidence (list of short strings)."""


def _format_judicial_logic(synthesis_rules: Dict[str, Any]) -> str:
    """Format rubric synthesis_rules as judicial_logic for judge system prompts (facts over opinions)."""
    if not synthesis_rules:
        return ""
    lines = ["Court rules (Chief Justice will apply these; weigh facts over opinions):"]
    for key, value in (synthesis_rules or {}).items():
        if isinstance(value, str) and value.strip():
            lines.append(f"- {key}: {value.strip()}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _opinion_for_dimension(
    judge_name: Literal["Prosecutor", "Defense", "TechLead"],
    dimension: Dict[str, Any],
    evidence_context: str,
    system_prompt: str,
    judicial_logic: str = "",
    max_retries: int = 2,
) -> JudicialOpinion:
    """Call LLM once for this dimension; return JudicialOpinion. Retry on parse failure."""
    dim_id = dimension.get("id", "unknown")
    dim_name = dimension.get("name", "Unknown")
    success = dimension.get("success_pattern", "")
    failure = dimension.get("failure_pattern", "")

    full_system = system_prompt
    if judicial_logic:
        full_system = f"{system_prompt}\n\n{judicial_logic}"

    user_content = (
        f"Criterion: {dim_name} (id={dim_id})\n"
        f"Success pattern: {success}\n"
        f"Failure pattern: {failure}\n\n"
        f"Evidence:\n{evidence_context}\n\n"
        f"Produce a single JudicialOpinion: judge={repr(judge_name)}, criterion_id={repr(dim_id)}, score (1-5), argument, cited_evidence (list)."
    )

    try:
        llm = get_judicial_llm()
    except Exception:
        llm = get_llm()
    llm = llm.with_structured_output(JudicialOpinion)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            opinion = llm.invoke(
                [
                    SystemMessage(content=full_system),
                    HumanMessage(content=user_content),
                ]
            )
            if isinstance(opinion, JudicialOpinion):
                # Ensure judge field is set correctly
                return opinion.model_copy(update={"judge": judge_name, "criterion_id": dim_id})
            return JudicialOpinion(
                judge=judge_name,
                criterion_id=dim_id,
                score=3,
                argument=str(opinion)[:500] if opinion else "No response",
                cited_evidence=[],
            )
        except Exception as e:
            last_error = e
            continue

    return JudicialOpinion(
        judge=judge_name,
        criterion_id=dim_id,
        score=3,
        argument=f"Parse/LLM error after retries: {last_error!s}"[:500],
        cited_evidence=[],
    )


def prosecutor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prosecutor: one JudicialOpinion per rubric dimension. Returns {"opinions": [...]}."""
    dimensions: List[Dict[str, Any]] = state.get("rubric_dimensions") or []
    judicial_logic = _format_judicial_logic(state.get("rubric_synthesis_rules") or {})
    opinions: List[JudicialOpinion] = []
    for dim in dimensions:
        evidence_list = _collect_evidence_for_dimension(state, dim)
        context = _evidence_context(evidence_list)
        opinion = _opinion_for_dimension(
            "Prosecutor",
            dim,
            context,
            PROSECUTOR_SYSTEM,
            judicial_logic=judicial_logic,
        )
        opinions.append(opinion)
    return {"opinions": opinions}


def defense_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Defense: one JudicialOpinion per rubric dimension. Returns {"opinions": [...]}."""
    dimensions: List[Dict[str, Any]] = state.get("rubric_dimensions") or []
    judicial_logic = _format_judicial_logic(state.get("rubric_synthesis_rules") or {})
    opinions: List[JudicialOpinion] = []
    for dim in dimensions:
        evidence_list = _collect_evidence_for_dimension(state, dim)
        context = _evidence_context(evidence_list)
        opinion = _opinion_for_dimension(
            "Defense",
            dim,
            context,
            DEFENSE_SYSTEM,
            judicial_logic=judicial_logic,
        )
        opinions.append(opinion)
    return {"opinions": opinions}


def tech_lead_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Tech Lead: one JudicialOpinion per rubric dimension. Returns {"opinions": [...]}."""
    dimensions: List[Dict[str, Any]] = state.get("rubric_dimensions") or []
    judicial_logic = _format_judicial_logic(state.get("rubric_synthesis_rules") or {})
    opinions: List[JudicialOpinion] = []
    for dim in dimensions:
        evidence_list = _collect_evidence_for_dimension(state, dim)
        context = _evidence_context(evidence_list)
        opinion = _opinion_for_dimension(
            "TechLead",
            dim,
            context,
            TECH_LEAD_SYSTEM,
            judicial_logic=judicial_logic,
        )
        opinions.append(opinion)
    return {"opinions": opinions}
