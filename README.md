# Automaton Auditor – Week 2 Interim (Wednesday)

Hierarchical LangGraph auditor: Detective layer (RepoInvestigator, DocAnalyst) with fan-out/fan-in and EvidenceAggregator. Judges and Chief Justice are out of scope for this interim submission.

## System prerequisites

- **Python:** 3.11 or newer
- **Git:** required for cloning target repos (used by RepoInvestigator)
- **Package manager:** [uv](https://docs.astral.sh/uv/) (install from the link if not present)

Optional: `OPENAI_API_KEY` in `.env` for DocAnalyst (PDF querying). Repo-only runs work without it.

## Setup

```bash
# Install dependencies (creates venv; uses committed lock file)
uv sync


# uv lock

# Copy env template and add your keys
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
```

The project keeps **`uv.lock`** in version control so all environments resolve the same dependency versions. After editing `pyproject.toml`, run `uv lock` and commit the updated `uv.lock` if you change dependencies.

## Run the full auditor workflow (detective graph)

The auditor runs a **detective graph** (current scope) and then passes control to a **judicial layer placeholder**. Future work will attach Judges (Prosecutor, Defense, TechLead) and Chief Justice after aggregation.

**Flow:**  
`entry → [repo_gate, doc_gate]` (fan-out) → conditional routing (skip RepoInvestigator if no `repo_url`, skip DocAnalyst if no `pdf_path`) → `RepoInvestigator` and/or `DocAnalyst` → `EvidenceAggregator` (fan-in) → **judicial placeholder** (where Judges + Chief Justice will run) → END.

**Input:** a **GitHub repository URL** and a **PDF report path** (e.g. your interim report).

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

**What to expect:** The graph runs RepoInvestigator and DocAnalyst in parallel (or skips one/both if URL or PDF path is missing). Evidence is aggregated and normalized; then the judicial placeholder runs. Output includes `evidences` and (if any) `detector_fatal_errors` for partial runs.

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

Use a real PDF path you have (e.g. `./reports/report.pdf` or `./reports/interim_report.pdf`). Check the output: `repo_investigator` should list several evidence items (Git Forensic Analysis, State Management Rigor, etc.). If the clone worked, you’ll see `found=True` for repo evidence and real content; if the clone failed, you’ll see one `RepoInvestigator: found=False` with the error in the rationale. RepoInvestigator clones the repo (in a temporary directory), runs `git log` and AST-based graph analysis, and emits structured `Evidence`. DocAnalyst ingests the PDF and queries it for theoretical depth / report accuracy, and emits `Evidence`. Both outputs are merged into state via reducers. **EvidenceAggregator** runs once after both detectives (or after skip nodes if inputs were missing): it normalizes confidence, reconciles evidence by goal across detectives, cross-references report paths with the repo file list, and records partial-run info if any detector failed. Then the **judicial placeholder** node runs; the full judicial layer (Judges + Chief Justice) will be attached there in a future release. The final state contains `evidences` (including `evidence_aggregator`) and optional `detector_fatal_errors`.

## Project layout

- `src/state.py` – Pydantic models (Evidence, JudicialOpinion, …) and AgentState with reducers
- `src/tools/repo_tools.py` – Sandboxed `git clone`, `extract_git_history`, `analyze_graph_structure` (AST)
- `src/tools/doc_tools.py` – `ingest_pdf`, `query_document` (RAG-lite)
- `src/nodes/detectives.py` – RepoInvestigator, DocAnalyst, EvidenceAggregator nodes
- `src/graph.py` – StateGraph: entry → [repo_gate, doc_gate] → conditional edges → detectives or skip nodes → EvidenceAggregator → judicial_placeholder → END
- `rubric.json` – Machine-readable rubric (dimensions, synthesis_rules)
- `spec/` – Constitution (Spec Kit–style)
- `reports/` – Interim report (PDF or source for export)

## Observability

Set in `.env`:

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=automaton-auditor`

Then run the graph; traces will appear in LangSmith if configured.
