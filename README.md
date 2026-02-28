# Automaton Auditor – Week 2

LangGraph-based auditor that evaluates a GitHub repository and an architectural PDF report against a machine-readable rubric. A **Context Builder** loads the rubric and distributes it; **Detectives** (RepoInvestigator, DocAnalyst, VisionInspector) collect evidence by artifact type; **Judges** (Prosecutor, Defense, Tech Lead) produce structured opinions; and the **Chief Justice** synthesizes a verdict with deterministic rules and writes a Markdown report.

## System prerequisites

- **Python:** 3.11 or newer
- **Git:** required for cloning target repos (RepoInvestigator)
- **Package manager:** [uv](https://docs.astral.sh/uv/) (install from the link if not present)

The **Judges** (and optional VisionInspector) need an LLM. You can use **OpenAI** (paid), **OpenRouter** (free tier), or **Ollama** (local, free). See **LLM providers** below.

## Setup

```bash
# Install dependencies (creates venv; uses committed lock file)
uv sync

# Copy env template and add your keys
cp .env.example .env
# Edit .env: set LLM_PROVIDER and the matching API key (or use Ollama with no key)
```

### LLM providers (no paid OpenAI required)

Set `LLM_PROVIDER` in `.env` to one of:

| Provider       | `.env` example | Notes |
|----------------|----------------|-------|
| **OpenRouter** | `LLM_PROVIDER=openrouter`<br>`OPENROUTER_API_KEY=sk-or-...`<br>`OPENROUTER_MODEL=openai/gpt-4o-mini` | Free tier at [openrouter.ai](https://openrouter.ai); one key for many models. |
| **Ollama**     | `LLM_PROVIDER=ollama`<br>`OLLAMA_MODEL=llama3.2` | Local, free. Run `ollama serve` and `ollama pull llama3.2`. For vision: `OLLAMA_VISION_MODEL=llava` and `ollama pull llava`. |
| **OpenAI**     | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=sk-...` | Default; paid. |

**Using Gemini for vision only:** Keep your main LLM (e.g. Ollama for judges) and use Google Gemini only for diagram analysis. Set `VISION_PROVIDER=gemini`, `GOOGLE_API_KEY=...`, and optionally `GOOGLE_VISION_MODEL=gemini-2.0-flash` in `.env`. With `AUDITOR_RUN_VISION=1`, VisionInspector calls Gemini to classify PDF diagrams (StateGraph vs sequence vs generic).

**Multi-model:** Use a different model per role. Set in `.env`; if unset, the app falls back to `LLM_PROVIDER` / `get_llm()` / `get_vision_llm()`.

| Role            | Env vars |
|-----------------|----------|
| Judicial bench  | `JUDICIAL_PROVIDER=groq`, `GROQ_API_KEY`, `GROQ_JUDICIAL_MODEL` (e.g. `llama-3.3-70b-versatile`) |
| Vision & PDF detective | `DETECTIVE_PROVIDER=gemini`, `GOOGLE_API_KEY`, `GOOGLE_DETECTIVE_MODEL` (e.g. `gemini-2.5-pro`) |
| Recorder        | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` |

See `.env.example` for all optional variables (model names, vision model, OpenRouter base URL, etc.).

### LLM in the Detective layer

Detectives collect facts only (no opinions). Optional LLM use:

- **DocAnalyst:** Set `AUDITOR_DETECTIVE_LLM=1`. DocAnalyst uses the detective LLM to classify whether the report has substantive explanation of terms (e.g. Dialectical Synthesis) or only keyword mention. Result is used for confidence and rationale; no scoring.
- **VisionInspector:** Set `AUDITOR_RUN_VISION=1` to run the vision model (Gemini via `DETECTIVE_PROVIDER=gemini`, else `get_vision_llm()`). Implementation is always present; execution is optional.
- **RepoInvestigator:** Tool-only (git, AST). No LLM required.

The project keeps **`uv.lock`** in version control so all environments resolve the same dependency versions. After editing `pyproject.toml`, run `uv lock` and commit the updated `uv.lock` if you change dependencies.

## Architecture and flow

The graph is defined in `src/graph.py`:

1. **Context Builder** – Loads `rubric.json` and distributes `dimensions` and `synthesis_rules` into state. Detectives receive only dimensions whose `target_artifact` matches their role; judges receive `judicial_logic` (synthesis rules) in their system prompts.

2. **Detectives (fan-out)** – Three branches, each gated by input:
   - **RepoInvestigator** – When `repo_url` is set: clone repo, run git history and AST analysis (state, graph, tools, judges). Produces `Evidence` for dimensions with `target_artifact: github_repo`. Uses each dimension’s `forensic_instruction` in evidence rationales.
   - **DocAnalyst** – When `pdf_path` is set: ingest PDF, query for theoretical depth and report accuracy. Produces `Evidence` for `target_artifact: pdf_report`. Uses `forensic_instruction` in rationales.
   - **VisionInspector** – When `pdf_path` is set and `AUDITOR_RUN_VISION=1`: extract images from PDF, optionally call vision model to classify diagrams. Produces `Evidence` for `target_artifact: pdf_images`. Uses `forensic_instruction` in the vision prompt.

3. **EvidenceAggregator (fan-in)** – Normalizes confidence, reconciles evidence by goal, cross-references report paths with repo file list (Report Accuracy), and records partial-run errors.

4. **Judges (fan-out)** – Prosecutor, Defense, and Tech Lead each produce one `JudicialOpinion` per rubric dimension via `.with_structured_output(JudicialOpinion)`. Each judge has a distinct system prompt; `synthesis_rules` from the rubric are appended as judicial logic (facts over opinions, security override, etc.).

5. **Chief Justice (fan-in)** – Applies hardcoded deterministic rules (security override, fact supremacy, dissent when variance > 2). Builds an `AuditReport`, serializes it to Markdown, and writes `audit/report_<timestamp>.md`. Returns `final_report` and `report_path` in state.

**Flow summary:**  
`context_builder` → `[repo_gate, doc_gate, vision_gate]` → conditional edges → RepoInvestigator / DocAnalyst / VisionInspector or skip nodes → `evidence_aggregator` → `[prosecutor, defense, tech_lead]` → `chief_justice` → END.

## Run the auditor

**Input:** A **GitHub repository URL** and a **PDF report path** (e.g. your architectural report).

**CLI (interactive – prompts for URL and PDF path):**

```bash
uv run python main.py
```

**CLI (with arguments):**

```bash
uv run python main.py "https://github.com/owner/repo" "./reports/report.pdf"
```

Optional: append `--json` to print evidences as JSON.

**Web UI:**

```bash
uv run python app.py
```

Then open **http://127.0.0.1:5000**. Submit repo URL and upload the report PDF; when the audit finishes you see the overall score, executive summary, and a link to the generated Markdown report.

**Output:** The run returns the final state with `final_report` (AuditReport), `report_path` (e.g. `audit/report_2026-02-28_08-30-17.md`), `evidences`, and `opinions`. The Markdown file contains Executive Summary, Criterion Breakdown (with dissent where applicable), and Remediation Plan.

## Rubric and targeting protocol

The rubric is in **`rubric.json`**: a machine-readable JSON spec with `dimensions` (each with `id`, `name`, `target_artifact`, `forensic_instruction`, `success_pattern`, `failure_pattern`) and `synthesis_rules` (security_override, fact_supremacy, functionality_weight, dissent_requirement, variance_re_evaluation).

- **Context Builder** – Iterates the `dimensions` array and distributes them (and `synthesis_rules`) into state so the “Constitution” can be updated centrally without redeploying code.
- **Detectives** – Only see dimensions where `target_artifact` matches their capability (`github_repo`, `pdf_report`, `pdf_images`). They receive and use `forensic_instruction` for those dimensions.
- **Judges** – Receive `synthesis_rules` as **judicial_logic** in their persona-specific system prompts so the verdict respects facts over opinions.
- **Chief Justice** – Uses `synthesis_rules` from state for deterministic conflict resolution.

To change behavior, edit `rubric.json` (add/remove/edit dimensions or synthesis rules); no code change or CLI flag is required.

## Project layout

| Path | Description |
|------|-------------|
| `src/state.py` | State definitions: `Evidence`, `JudicialOpinion`, `CriterionResult`, `AuditReport`, `AgentState` with reducers (`operator.add`, `operator.ior`). |
| `src/tools/repo_tools.py` | Forensic repo tools: sandboxed `clone_repo_sandboxed`, `extract_git_history`, `analyze_graph_structure`, `analyze_state_structure`, `list_src_files`. |
| `src/tools/doc_tools.py` | PDF tools: `ingest_pdf`, `query_document`, `extract_file_paths_from_chunks`, `extract_images_from_pdf`. |
| `src/nodes/detectives.py` | RepoInvestigator, DocAnalyst, VisionInspector, EvidenceAggregator. |
| `src/nodes/judges.py` | Prosecutor, Defense, Tech Lead with distinct system prompts and `.with_structured_output(JudicialOpinion)`; judicial_logic from rubric. |
| `src/nodes/justice.py` | ChiefJusticeNode: hardcoded synthesis rules, produces `AuditReport`, writes Markdown to `audit/report_<timestamp>.md`. |
| `src/graph.py` | StateGraph: context_builder → [repo_gate, doc_gate, vision_gate] → detectives or skip → evidence_aggregator → [prosecutor, defense, tech_lead] → chief_justice → END. |
| `src/llm.py` | LLM factory (OpenAI, OpenRouter, Ollama, Groq, Gemini) for judicial, detective, and vision. |
| `rubric.json` | Machine-readable rubric: dimensions and synthesis_rules. |
| `main.py` | CLI entrypoint: repo URL + PDF path, runs audit, prints score and report path. |
| `app.py` | Flask web UI: submit repo URL and upload PDF, view report. |
| `scripts/test_clone.py` | Smoke test: clone + repo tools only (no full graph). |
| `docs/testing.md` | Step-by-step testing (tools-only, full audit, smoke test). |
| `audit/` | Generated Markdown reports (`report_<timestamp>.md`). |
| `reports/` | Your PDF reports (e.g. interim or final). |

## Testing

See **[docs/testing.md](docs/testing.md)** for step-by-step testing. Summary:

**1. Clone + tools only (no full graph):**

```bash
uv run python scripts/test_clone.py
# Or with your own repo:
uv run python scripts/test_clone.py "https://github.com/owner/repo"
```

**2. Full audit (detectives → judges → Chief Justice → report):**

```bash
uv run python main.py "https://github.com/octocat/Hello-World" "./reports/report.pdf"
```

Use a real PDF path. Check output for evidence counts, overall score, and `Report saved to: audit/report_<timestamp>.md`.

## Observability and LangSmith trace link

Set in `.env`:

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=automaton-auditor`
- `LANGCHAIN_API_KEY=lsv2_pt_...`

Then run the audit (e.g. `uv run python main.py <repo_url> <pdf_path>`). When the run finishes, the script prints a **LangSmith trace URL** — use this link for your submission (full reasoning loop: detectives collecting evidence, judges arguing, Chief Justice synthesizing the verdict). If the URL cannot be resolved, the run ID is printed so you can search for it in your LangSmith project.
