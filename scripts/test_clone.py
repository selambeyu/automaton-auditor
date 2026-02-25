#!/usr/bin/env python3
"""
Test that sandboxed clone, git log, and AST analysis work.
Run from repo root: uv run python scripts/test_clone.py [repo_url]
Default URL: a tiny public repo (octocat/Hello-World).
"""

import sys
from pathlib import Path

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.repo_tools import (
    clone_repo_sandboxed,
    extract_git_history,
    analyze_graph_structure,
)


def main() -> int:
    repo_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/octocat/Hello-World"
    print(f"Testing clone: {repo_url}\n")

    try:
        with clone_repo_sandboxed(repo_url) as path:
            print(f"  Cloned to (temp): {path}")
            print(f"  Exists: {path.exists()}")
            commits = extract_git_history(path)
            print(f"  Git log: {len(commits)} commit(s)")
            for c in commits[:5]:
                print(f"    - {c.hash} {c.message}")
            if len(commits) > 5:
                print(f"    ... and {len(commits) - 5} more")
            graph = analyze_graph_structure(path)
            print(f"  AST: StateGraph={graph.has_state_graph}, add_edge={graph.add_edge_calls}, nodes={graph.node_names[:5] or []}")
        print("\n  Temp dir cleaned up after context exit.")
        print("OK: Clone and tools work.")
        return 0
    except Exception as e:
        print(f"\nFAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
