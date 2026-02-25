# Automaton Auditor – Week 2 Interim (Wednesday)

Hierarchical LangGraph auditor: Detective layer (RepoInvestigator, DocAnalyst) with fan-out/fan-in and EvidenceAggregator. Judges and Chief Justice are out of scope for this interim submission.

## Setup

- **Python:** 3.11+
- **Package manager:** [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies (creates venv and locks)
uv sync

# Copy env template and add your keys
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
```

## Install from repo root

From the repository root:

```bash
uv sync
```

Then run the detective graph (see below). If you run without `uv run`, set `PYTHONPATH=.` so that `src` is importable.

## Run the detective graph

Input: a **GitHub repository URL** and a **PDF report path** (e.g. your interim report).

**Interactive (prompts for input):**

```bash
uv run python main.py
```

You will be asked for the repo URL and the report PDF path.

**Or pass them as arguments:**

```bash
uv run python main.py "https://github.com/owner/repo" "./reports/interim_report.pdf"
```

Optional: append `--json` to print evidences as JSON.

**What to expect:** The graph runs RepoInvestigator and DocAnalyst in parallel.

## Testing the clone

**1. Clone + tools only (no full graph)** – checks that `git clone`, `git log`, and AST run correctly:

```bash
uv run python scripts/test_clone.py
# Or with your own repo:
uv run python scripts/test_clone.py "https://github.com/owner/repo"
```

You should see: `Cloned to (temp): ...`, commit count, AST summary, then `OK: Clone and tools work.` If you see `FAIL: ...`, often it’s network, git not in PATH, or (in some environments) temp-dir permissions.

**2. Full detective graph** – runs both detectives and prints evidence:

```bash
uv run python main.py "https://github.com/octocat/Hello-World" "./reports/report.pdf"
```

Use a real PDF path you have (e.g. `./reports/report.pdf` or `./reports/interim_report.pdf`). Check the output: `repo_investigator` should list several evidence items (Git Forensic Analysis, State Management Rigor, etc.). If the clone worked, you’ll see `found=True` for repo evidence and real content; if the clone failed, you’ll see one `RepoInvestigator: found=False` with the error in the rationale. RepoInvestigator clones the repo (in a temporary directory), runs `git log` and AST-based graph analysis, and emits structured `Evidence`. DocAnalyst ingests the PDF and queries it for theoretical depth / report accuracy, and emits `Evidence`. Both outputs are merged into state via reducers; EvidenceAggregator runs once after both, then the run finishes. The final state contains `evidences` (keyed by `repo_investigator` and `doc_analyst`) and can be used later by the Judicial layer.

## Project layout

- `src/state.py` – Pydantic models (Evidence, JudicialOpinion, …) and AgentState with reducers
- `src/tools/repo_tools.py` – Sandboxed `git clone`, `extract_git_history`, `analyze_graph_structure` (AST)
- `src/tools/doc_tools.py` – `ingest_pdf`, `query_document` (RAG-lite)
- `src/nodes/detectives.py` – RepoInvestigator, DocAnalyst, EvidenceAggregator nodes
- `src/graph.py` – StateGraph: entry → [RepoInvestigator, DocAnalyst] → EvidenceAggregator → END
- `rubric.json` – Machine-readable rubric (dimensions, synthesis_rules)
- `spec/` – Constitution (Spec Kit–style)
- `reports/` – Interim report (PDF or source for export)

## Observability

Set in `.env`:

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=automaton-auditor`

Then run the graph; traces will appear in LangSmith if configured.
