#!/usr/bin/env python3
"""
Run the detective graph against a target repo URL and PDF report path.
Prompts for repo_url and report path if not given as arguments.
Usage: uv run python main.py
   or:  uv run python main.py <repo_url> <pdf_path>
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import run_audit


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
    print("Running detective graph (RepoInvestigator + DocAnalyst in parallel)...")
    result = run_audit(repo_url, pdf_path)
    evidences = result.get("evidences") or {}
    print("\n--- Evidence summary ---")
    for source, items in evidences.items():
        print(f"\n{source}: {len(items)} evidence item(s)")
        for e in items:
            goal = e.get("goal", "?") if isinstance(e, dict) else getattr(e, "goal", "?")
            found = e.get("found") if isinstance(e, dict) else getattr(e, "found", None)
            conf = e.get("confidence") if isinstance(e, dict) else getattr(e, "confidence", None)
            rationale = e.get("rationale") if isinstance(e, dict) else getattr(e, "rationale", "")
            line = f"  - {goal}: found={found} confidence={conf}"
            if rationale and not found:
                line += f" | {rationale[:120]}"
            print(line)
    print("\nFull state keys:", list(result.keys()))
    # Optional: dump evidences as JSON
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
