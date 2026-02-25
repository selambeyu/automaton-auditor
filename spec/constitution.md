# Automaton Auditor – Project Constitution

These principles govern all implementation decisions for the Week 2 Automaton Auditor (Interim and Final).

## 1. State and types

- **Pydantic and TypedDict only.** Shared graph state is defined with Pydantic models and `TypedDict`. Plain Python dicts are not used for `AgentState` or for structured outputs (Evidence, JudicialOpinion, AuditReport).
- **Reducers for shared writes.** Fields written by multiple nodes use LangGraph reducers so updates merge instead of overwrite: `Annotated[Dict[str, List[Evidence]], operator.ior]` for evidences, `Annotated[List[JudicialOpinion], operator.add]` for opinions.

## 2. Forensic mindset (Detective layer)

- **Detectives collect facts only.** They do not score, grade, or opine. Output is structured `Evidence` (goal, found, content, location, rationale, confidence).
- **Evidence over opinion.** Facts from Detectives (repo, PDF, diagrams) are the source of truth. Judicial interpretation must cite evidence; claims without evidence are overruled.

## 3. Sandboxing and safety

- **Git and tools are sandboxed.** Repository clone and git commands run inside `tempfile.TemporaryDirectory()`. The live working directory is never used for cloned repos.
- **No raw shell.** Use `subprocess.run()` with captured stdout/stderr and return-code checks. Do not use `os.system()` for clone or git operations. Handle authentication and command failures gracefully.

## 4. Structure over string matching

- **AST for code structure.** Graph structure, state definitions, and tool usage are inferred via Python’s `ast` module (or equivalent parsing). Regex alone is not used to determine StateGraph topology, Pydantic usage, or reducer presence.
- **Rubric as config.** The machine-readable rubric (`rubric.json`) is loaded at runtime. Detectives and Judges receive instructions filtered by `target_artifact` and use `forensic_instruction` / synthesis rules from the rubric.

## 5. Interim scope (Wednesday)

- **Detective layer + partial graph only.** Implement RepoInvestigator and DocAnalyst; wire them in parallel with an EvidenceAggregator (fan-out/fan-in). Judges and Chief Justice are out of scope for the interim submission.
- **Deliverables match the checklist.** File paths and deliverables follow the Wednesday submission requirements exactly (`src/state.py`, `src/tools/repo_tools.py`, etc.).
