"""Typed state and Pydantic models for the Automaton Auditor graph."""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --- Detective output ---


class Evidence(BaseModel):
    """Structured evidence produced by Detective nodes. No opinions."""

    goal: str = Field(description="The forensic goal this evidence addresses")
    found: bool = Field(description="Whether the artifact exists")
    content: Optional[str] = Field(default=None, description="Relevant snippet or content")
    location: str = Field(
        description="File path or commit hash",
    )
    rationale: str = Field(
        description="Rationale for confidence on this evidence",
    )
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")


# --- Judge output ---


class JudicialOpinion(BaseModel):
    """Single judge's opinion for one criterion."""

    judge: Literal["Prosecutor", "Defense", "TechLead"]
    criterion_id: str
    score: int = Field(ge=1, le=5)
    argument: str
    cited_evidence: List[str] = Field(default_factory=list)


# --- Chief Justice output ---


class CriterionResult(BaseModel):
    """Final result for one rubric dimension after synthesis."""

    dimension_id: str
    dimension_name: str
    final_score: int = Field(ge=1, le=5)
    judge_opinions: List[JudicialOpinion] = Field(default_factory=list)
    dissent_summary: Optional[str] = Field(
        default=None,
        description="Required when score variance > 2",
    )
    remediation: str = Field(
        description="Specific file-level instructions for improvement",
    )


class AuditReport(BaseModel):
    """Final audit report produced by Chief Justice."""

    repo_url: str
    executive_summary: str
    overall_score: float
    criteria: List[CriterionResult] = Field(default_factory=list)
    remediation_plan: str = ""


# --- Graph state ---


class AgentState(TypedDict, total=False):
    """LangGraph state. Use reducers for evidences and opinions."""

    repo_url: str
    pdf_path: str
    rubric_dimensions: List[Dict[str, Any]]
    evidences: Annotated[
        Dict[str, List[Evidence]],
        operator.ior,
    ]
    opinions: Annotated[
        List[JudicialOpinion],
        operator.add,
    ]
    final_report: Optional[AuditReport]
    # Fatal errors per detector (e.g. clone failure, PDF missing); used for routing and aggregation
    detector_fatal_errors: Annotated[
        Dict[str, str],
        operator.ior,
    ]
