#!/usr/bin/env python3
"""
Run the full auditor graph: detectives -> judges -> Chief Justice -> Markdown report.
Repo URL and report path are from CLI; rubric is loaded from rubric.json at runtime
(change the file content to add/edit dimensions and the next run adapts).
Usage: uv run python main.py
   or:  uv run python main.py <repo_url> <pdf_path>
   or:  uv run python main.py <repo_url> <pdf_path> --json
"""

import json
import sys
from pathlib import Path

# Load .env from project root so LLM_PROVIDER and API keys are set before any imports that read them
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Ensure repo root is on path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import run_audit

def _print_langsmith_trace_url(run_id: str) -> None:
    """Print the LangSmith trace URL for this run (for submission: link to full reasoning loop)."""
    import os

    try:
        from langsmith import Client

        client = Client(api_key=os.environ.get("LANGCHAIN_API_KEY"))
        project_name = os.environ.get("LANGCHAIN_PROJECT") or "default"
        # get_run expects a UUID; run_id is already a string from uuid.uuid4()
        run = client.read_run(run_id)
        url = client.get_run_url(run=run, project_name=project_name)
        print(f"\n--- LangSmith trace (submit this link) ---")
        print(url)
        print("---")
    except Exception as e:
        # Trace may still be visible in the project; print run_id so user can search
        print(f"\nLangSmith trace run_id (search in your project): {run_id}", file=sys.stderr)
        print(f"(Could not resolve URL: {e})", file=sys.stderr)


def main() -> None:
    if len(sys.argv) >= 3:
        repo_url = sys.argv[1].strip()
        pdf_path = sys.argv[2].strip()
    else:
        print("Enter the GitHub repository URL and path to the PDF report.\n")
        repo_url = input("Repo URL (e.g. https://github.com/owner/repo): ").strip()
        if not repo_url:
            print("Error: Repo URL is required.", file=sys.stderr)
            sys.exit(1)
        pdf_path = input("Report PDF path (e.g. ./reports/report.pdf): ").strip()
        if not pdf_path:
            print("Error: Report path is required.", file=sys.stderr)
            sys.exit(1)
    if not Path(pdf_path).exists():
        print(f"Warning: PDF path does not exist: {pdf_path}", file=sys.stderr)

    # LangSmith tracing: must have LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY set in .env
    import os
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes"):
        if not os.environ.get("LANGCHAIN_API_KEY", "").strip():
            print("Note: LANGCHAIN_TRACING_V2 is set but LANGCHAIN_API_KEY is missing. Traces will not appear in LangSmith.", file=sys.stderr)
        else:
            print("LangSmith tracing enabled (LANGCHAIN_TRACING_V2=true).")
    print("Running full audit (detectives -> judges -> Chief Justice)...")
    result = run_audit(repo_url, pdf_path)

    # LangSmith trace link (for submission: "link to your langsmith trace showing the full reasoning loop")
    run_id = result.get("_run_id")
    if run_id and os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes") and os.environ.get("LANGCHAIN_API_KEY", "").strip():
        _print_langsmith_trace_url(run_id)

    # Final report: summary and path
    final_report = result.get("final_report")
    if final_report is not None:
        report = final_report
        if hasattr(report, "overall_score"):
            overall = report.overall_score
        elif isinstance(report, dict):
            overall = report.get("overall_score", 0)
        else:
            overall = 0
        print(f"\n--- Audit complete ---")
        print(f"Overall score: {overall}")
        report_path = result.get("report_path")
        if report_path:
            print(f"Report saved to: {report_path}")
        else:
            print("(Report was generated; see state['final_report'].)")
    else:
        print("\nNo final_report in state (judicial layer may have been skipped or failed).")

    evidences = result.get("evidences") or {}
    print("\n--- Evidence summary ---")
    for source, items in evidences.items():
        print(f"  {source}: {len(items)} item(s)")
    print("State keys:", list(result.keys()))

    if "--json" in sys.argv:
        def _serialize(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "model_dump_json"):
                return json.loads(obj.model_dump_json())
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(x) for x in obj]
            return obj
        print(json.dumps(_serialize(evidences), indent=2, default=str))


if __name__ == "__main__":
    main()
