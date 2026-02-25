"""Sandboxed repo operations and AST-based graph analysis. No os.system."""

from __future__ import annotations

import ast
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional


@dataclass
class CommitInfo:
    """Single commit from git log."""

    hash: str
    message: str
    timestamp: Optional[str] = None


@dataclass
class GraphStructure:
    """Result of AST analysis of the StateGraph."""

    has_state_graph: bool = False
    add_edge_calls: int = 0
    add_conditional_edges_calls: int = 0
    node_names: List[str] = field(default_factory=list)
    parallel_fan_out: bool = False
    has_sync_node: bool = False
    snippet: Optional[str] = None


@contextmanager
def clone_repo_sandboxed(repo_url: str) -> Generator[Path, None, None]:
    """
    Clone the repository into a temporary directory. Yields the clone path.
    Temp dir is removed when the context exits. Uses subprocess.run(); no os.system().
    """
    if not repo_url or not isinstance(repo_url, str):
        raise ValueError("repo_url must be a non-empty string")
    if not (
        repo_url.startswith("https://")
        or repo_url.startswith("http://")
        or repo_url.startswith("git@")
    ):
        raise ValueError("repo_url must be an https or git SSH URL")

    with tempfile.TemporaryDirectory(prefix="auditor_clone_") as tmp:
        dest = Path(tmp)
        result = subprocess.run(
            ["git", "clone", "--depth", "50", repo_url, str(dest)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr or result.stdout or "Unknown error"
            if "Authentication failed" in err or "Permission denied" in err:
                raise PermissionError(f"Git authentication failed: {err}")
            raise RuntimeError(f"git clone failed: {err}")
        subdirs = [d for d in dest.iterdir() if d.is_dir() and not d.name.startswith(".")]
        yield subdirs[0] if len(subdirs) == 1 else dest


@dataclass
class StateStructure:
    """Result of AST analysis of state definitions (state.py or in graph.py)."""

    has_base_model: bool = False
    has_typed_dict: bool = False
    has_evidence_class: bool = False
    has_judicial_opinion_class: bool = False
    has_operator_add: bool = False
    has_operator_ior: bool = False
    snippet: Optional[str] = None


def _analyze_state_ast(tree: ast.AST, source: str) -> StateStructure:
    """Walk AST to find BaseModel, TypedDict, Evidence, JudicialOpinion, operator.add, operator.ior."""
    out = StateStructure()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "Evidence":
                out.has_evidence_class = True
            if node.name == "JudicialOpinion":
                out.has_judicial_opinion_class = True
            for base in node.bases:
                name = None
                if isinstance(base, ast.Name):
                    name = base.id
                elif isinstance(base, ast.Attribute):
                    name = base.attr
                if name == "BaseModel":
                    out.has_base_model = True
                if name == "TypedDict":
                    out.has_typed_dict = True
    if "operator.ior" in source:
        out.has_operator_ior = True
    if "operator.add" in source:
        out.has_operator_add = True
    return out


def analyze_state_structure(repo_path: str | Path) -> StateStructure:
    """
    Use AST on src/state.py (or state in src/graph.py) to find BaseModel, TypedDict,
    Evidence, JudicialOpinion, and operator.add / operator.ior reducers.
    """
    path = Path(repo_path)
    state_file = path / "src" / "state.py"
    if not state_file.exists():
        return StateStructure()
    try:
        source = state_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return StateStructure()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return StateStructure()
    out = _analyze_state_ast(tree, source)
    out.snippet = source[:2000] if len(source) > 2000 else source
    return out


def list_src_files(repo_path: str | Path) -> List[str]:
    """Return relative paths of all files under src/ for cross-reference with report."""
    path = Path(repo_path)
    src_dir = path / "src"
    if not src_dir.is_dir():
        return []
    out: List[str] = []
    for f in src_dir.rglob("*"):
        if f.is_file():
            out.append(str(f.relative_to(path)))
    return sorted(out)


def extract_git_history(repo_path: str | Path) -> List[CommitInfo]:
    """
    Run git log --oneline --reverse in the given path. Return structured commits.
    """
    path = Path(repo_path)
    if not path.is_dir():
        return []
    result = subprocess.run(
        ["git", "log", "--oneline", "--reverse", "--format=%h %s %ci"],
        cwd=path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return []
    commits: List[CommitInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 2:
            h, msg = parts[0], parts[1]
            ts = parts[2] if len(parts) > 2 else None
            commits.append(CommitInfo(hash=h, message=msg, timestamp=ts))
    return commits


def _find_state_graph_and_edges(tree: ast.AST, source: str) -> GraphStructure:
    """Walk AST to find StateGraph usage and add_edge / add_conditional_edges."""
    out = GraphStructure()
    node_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = None
            if isinstance(f, ast.Attribute):
                name = f.attr
                if isinstance(f.value, ast.Name):
                    # e.g. builder.add_edge
                    pass
            if name == "add_edge":
                out.add_edge_calls += 1
            elif name == "add_conditional_edges":
                out.add_conditional_edges_calls += 1
            elif name == "StateGraph":
                out.has_state_graph = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_node" and node.args:
                if isinstance(node.args[0], ast.Constant):
                    node_names.add(str(node.args[0].value))
    out.node_names = sorted(node_names)
    # Heuristic: multiple add_edge from same "source" suggests fan-out; presence of
    # a node like evidence_aggregator or aggregator suggests fan-in
    out.parallel_fan_out = out.add_edge_calls >= 2
    sync_like = {"evidence_aggregator", "evidenceaggregator", "aggregator"}
    out.has_sync_node = any(n.lower().replace("_", "") in sync_like for n in out.node_names)
    return out


def analyze_graph_structure(repo_path: str | Path) -> GraphStructure:
    """
    Use Python AST to find StateGraph, add_edge, add_conditional_edges in src/graph.py.
    No regex for structure. Returns GraphStructure with snippet if file found.
    """
    path = Path(repo_path)
    graph_file = path / "src" / "graph.py"
    if not graph_file.exists():
        return GraphStructure()
    try:
        source = graph_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return GraphStructure()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return GraphStructure()
    out = _find_state_graph_and_edges(tree, source)
    # Capture a short snippet around StateGraph or first add_edge
    if out.has_state_graph or out.add_edge_calls:
        out.snippet = source[:2000] if len(source) > 2000 else source
    return out
