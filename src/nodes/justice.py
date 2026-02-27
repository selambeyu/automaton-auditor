"""Chief Justice node: deterministic synthesis of judge opinions into an AuditReport and Markdown file."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.state import (
    AuditReport,
    CriterionResult,
    Evidence,
    JudicialOpinion,
)


def _load_rubric(rubric_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load rubric.json; return dict with dimensions and synthesis_rules."""
    path = rubric_path or Path(__file__).resolve().parent.parent.parent / "rubric.json"
    if not path.exists():
        return {"dimensions": [], "synthesis_rules": {}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "dimensions": data.get("dimensions", []),
        "synthesis_rules": data.get("synthesis_rules", {}),
    }


def _normalize_opinion(o: Any) -> JudicialOpinion:
    """Ensure we have a JudicialOpinion instance (state may have dicts)."""
    if isinstance(o, JudicialOpinion):
        return o
    if isinstance(o, dict):
        return JudicialOpinion(**o)
    raise TypeError(f"Expected JudicialOpinion or dict, got {type(o)}")


def _get_evidence_for_criterion(
    state: Dict[str, Any],
    criterion_id: str,
) -> List[Evidence]:
    """Collect evidence items whose goal matches this criterion (by dimension name or id)."""
    dimensions = state.get("rubric_dimensions") or []
    dim = next((d for d in dimensions if d.get("id") == criterion_id), None)
    if not dim:
        return []
    dim_name = dim.get("name", "")
    evidences = state.get("evidences") or {}
    out: List[Evidence] = []
    for _source, items in evidences.items():
        for item in items or []:
            if isinstance(item, dict):
                item = Evidence(**item)
            goal = getattr(item, "goal", "")
            if goal == dim_name or criterion_id in goal.lower().replace(" ", "_"):
                out.append(item)
    return out


def _resolve_score(
    criterion_id: str,
    opinions_by_judge: Dict[str, JudicialOpinion],
    state: Dict[str, Any],
    synthesis_rules: Dict[str, str],
) -> tuple[int, Optional[str], str]:
    """
    Apply hardcoded synthesis rules. Returns (final_score, dissent_summary, remediation).
    """
    prosecutor = opinions_by_judge.get("Prosecutor")
    defense = opinions_by_judge.get("Defense")
    tech_lead = opinions_by_judge.get("TechLead")

    scores = []
    if prosecutor:
        scores.append(prosecutor.score)
    if defense:
        scores.append(defense.score)
    if tech_lead:
        scores.append(tech_lead.score)

    if not scores:
        return 3, None, "No judge opinions available for this criterion."

    # Rule of Security: Prosecutor cites security flaw -> cap at 3
    if prosecutor and prosecutor.argument:
        arg_lower = prosecutor.argument.lower()
        if "os.system" in arg_lower or "shell injection" in arg_lower or "security" in arg_lower and ("unsanitized" in arg_lower or "raw " in arg_lower):
            dissent = f"Prosecutor identified a security concern; score capped at 3 (Rule of Security)."
            remediation = "Address security: use subprocess with proper sanitization, avoid os.system with user input."
            return min(3, max(scores)), dissent, remediation

    # Rule of Evidence: Defense claims metacognition / deep understanding but evidence says found=False
    evidence_list = _get_evidence_for_criterion(state, criterion_id)
    if defense and defense.argument:
        arg_lower = defense.argument.lower()
        if "metacognition" in arg_lower or "deep understanding" in arg_lower:
            if evidence_list and not any(getattr(e, "found", True) for e in evidence_list):
                dissent = "Defense argued for deep understanding but forensic evidence did not support it (Rule of Evidence)."
                remediation = "Ensure implementation is evidenced in code/report so Defense claims can be verified."
                # Prefer Prosecutor or Tech Lead score
                other = (prosecutor or tech_lead)
                score = other.score if other else 2
                return score, dissent, remediation

    # Rule of Functionality: for graph_orchestration, Tech Lead carries highest weight
    if criterion_id == "graph_orchestration" and tech_lead:
        # Tech Lead says modular/workable -> lean toward their score
        arg_lower = (tech_lead.argument or "").lower()
        if "modular" in arg_lower or "workable" in arg_lower or "sound" in arg_lower:
            dissent = None
            if scores and max(scores) - min(scores) > 2:
                dissent = "Prosecutor and Defense disagreed; Tech Lead assessment (architecture viability) given highest weight."
            return tech_lead.score, dissent, tech_lead.argument[:500] if tech_lead.argument else "See Tech Lead opinion."

    # Variance > 2: require dissent summary
    score_min, score_max = min(scores), max(scores)
    variance = score_max - score_min
    dissent_summary: Optional[str] = None
    if variance > 2:
        parts = []
        if prosecutor:
            parts.append(f"Prosecutor: {prosecutor.score} — {prosecutor.argument[:150]}...")
        if defense:
            parts.append(f"Defense: {defense.score} — {defense.argument[:150]}...")
        if tech_lead:
            parts.append(f"Tech Lead: {tech_lead.score} — {tech_lead.argument[:150]}...")
        dissent_summary = " | ".join(parts)

    # Default: use median / Tech Lead as tie-breaker when variance is high
    if tech_lead and variance > 2:
        final = tech_lead.score
    else:
        final = round(sum(scores) / len(scores)) if scores else 3
        final = max(1, min(5, final))

    remediation = "Review evidence and judge arguments; address gaps noted by Prosecutor and Tech Lead."
    if prosecutor and prosecutor.score <= 2 and prosecutor.argument:
        remediation = prosecutor.argument[:400]

    return final, dissent_summary, remediation


def _report_to_markdown(report: AuditReport) -> str:
    """Serialize AuditReport to Markdown: Executive Summary -> Criterion Breakdown -> Remediation Plan."""
    lines = [
        "# Audit Report",
        "",
        f"**Repository:** {report.repo_url}",
        f"**Overall Score:** {report.overall_score:.2f}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Criterion Breakdown",
        "",
    ]
    for c in report.criteria:
        lines.append(f"### {c.dimension_name} (`{c.dimension_id}`)")
        lines.append("")
        lines.append(f"- **Final Score:** {c.final_score}")
        if c.dissent_summary:
            lines.append(f"- **Dissent:** {c.dissent_summary}")
        lines.append("- **Remediation:** " + c.remediation.replace("\n", " "))
        lines.append("")
        for o in c.judge_opinions:
            lines.append(f"  - **{o.judge}** (score {o.score}): {o.argument[:200]}...")
        lines.append("")
    lines.append("## Remediation Plan")
    lines.append("")
    lines.append(report.remediation_plan)
    lines.append("")
    return "\n".join(lines)


def _write_report_markdown(report: AuditReport, out_path: Path) -> None:
    """Write AuditReport to a Markdown file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = _report_to_markdown(report)
    out_path.write_text(md, encoding="utf-8")


def chief_justice_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize judge opinions into an AuditReport using hardcoded rules.
    Writes the report to audit/report_<timestamp>.md.
    Returns {"final_report": AuditReport, "report_path": str}.
    """
    # Use rubric from state so the same dynamic repo/report/rubric is used end-to-end
    dimensions = state.get("rubric_dimensions") or []
    synthesis_rules = state.get("rubric_synthesis_rules")
    if synthesis_rules is None:
        synthesis_rules = _load_rubric(None).get("synthesis_rules", {})

    raw_opinions = state.get("opinions") or []
    opinions: List[JudicialOpinion] = []
    for o in raw_opinions:
        try:
            opinions.append(_normalize_opinion(o))
        except Exception:
            continue

    # Group by criterion_id
    by_criterion: Dict[str, List[JudicialOpinion]] = defaultdict(list)
    for o in opinions:
        by_criterion[o.criterion_id].append(o)

    criteria_results: List[CriterionResult] = []
    for dim in dimensions:
        dim_id = dim.get("id", "")
        dim_name = dim.get("name", "Unknown")
        judge_opinions = by_criterion.get(dim_id, [])
        by_judge = {o.judge: o for o in judge_opinions}

        final_score, dissent_summary, remediation = _resolve_score(
            dim_id,
            by_judge,
            state,
            synthesis_rules,
        )

        criteria_results.append(
            CriterionResult(
                dimension_id=dim_id,
                dimension_name=dim_name,
                final_score=final_score,
                judge_opinions=judge_opinions,
                dissent_summary=dissent_summary,
                remediation=remediation,
            )
        )

    overall = sum(c.final_score for c in criteria_results) / len(criteria_results) if criteria_results else 0.0
    repo_url = state.get("repo_url", "")

    executive_summary = (
        f"Audit of {repo_url}. "
        f"Overall score: {overall:.2f}/5 across {len(criteria_results)} criteria. "
        "See Criterion Breakdown for per-dimension scores and dissent where applicable."
    )
    remediation_plan = "\n\n".join(
        f"- **{c.dimension_name}:** {c.remediation}" for c in criteria_results
    )

    audit_report = AuditReport(
        repo_url=repo_url,
        executive_summary=executive_summary,
        overall_score=round(overall, 2),
        criteria=criteria_results,
        remediation_plan=remediation_plan,
    )

    # Write Markdown to audit/report_<timestamp>.md
    repo_root = Path(__file__).resolve().parent.parent.parent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = repo_root / "audit" / f"report_{timestamp}.md"
    _write_report_markdown(audit_report, out_path)

    return {"final_report": audit_report, "report_path": str(out_path)}
