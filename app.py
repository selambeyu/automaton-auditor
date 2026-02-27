#!/usr/bin/env python3
"""
Web UI for the Automaton Auditor: submit repo URL + PDF report, run audit, view results.
Run: uv run python app.py   or: uv run flask --app app run
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Ensure project root is on path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# Where to save uploaded PDFs (temporary; path passed to run_audit)
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
AUDIT_DIR = Path(__file__).resolve().parent / "audit"
UPLOADS_DIR.mkdir(exist_ok=True)
AUDIT_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/audit", methods=["POST"])
def run_audit_api():
    """Accept repo_url and PDF file; run audit; return report path and summary."""
    repo_url = (request.form.get("repo_url") or "").strip()
    if not repo_url:
        return jsonify({"ok": False, "error": "Repo URL is required."}), 400

    pdf_file = request.files.get("pdf")
    if not pdf_file or pdf_file.filename == "":
        return jsonify({"ok": False, "error": "PDF file is required."}), 400

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "File must be a PDF."}), 400

    # Save upload to a unique path so run_audit can read it
    safe_name = f"{uuid.uuid4().hex}.pdf"
    pdf_path = UPLOADS_DIR / safe_name
    try:
        pdf_file.save(str(pdf_path))
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to save upload: {e}"}), 500

    try:
        from src.graph import run_audit

        result = run_audit(repo_url, str(pdf_path))
    except Exception as e:
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception:
                pass
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception:
                pass

    final_report = result.get("final_report")
    report_path = result.get("report_path")
    report_filename = Path(report_path).name if report_path else None

    # Optional: resolve LangSmith trace URL (mirrors main.py logic)
    trace_url = None
    run_id = result.get("_run_id")
    if run_id and os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes"):
        api_key = (os.environ.get("LANGCHAIN_API_KEY") or "").strip()
        if api_key:
            try:
                from langsmith import Client

                client = Client(api_key=api_key)
                project_name = os.environ.get("LANGCHAIN_PROJECT") or "default"
                run = client.read_run(run_id)
                trace_url = client.get_run_url(run=run, project_name=project_name)
            except Exception:
                trace_url = None

    overall_score = None
    executive_summary = None
    if final_report is not None:
        if hasattr(final_report, "overall_score"):
            overall_score = final_report.overall_score
            executive_summary = getattr(final_report, "executive_summary", None) or ""
        elif isinstance(final_report, dict):
            overall_score = final_report.get("overall_score")
            executive_summary = final_report.get("executive_summary") or ""

    payload = {
        "ok": True,
        "report_path": report_path,
        "report_filename": report_filename,
        "report_url": f"/reports/{report_filename}" if report_filename else None,
        "overall_score": overall_score,
        "executive_summary": executive_summary or "",
        "run_id": run_id,
        "trace_url": trace_url,
    }
    return jsonify(payload)


@app.route("/reports/<path:filename>")
def serve_report(filename):
    """Serve a generated audit report (Markdown) from the audit/ directory."""
    try:
        path = (AUDIT_DIR / filename).resolve()
        audit_resolved = AUDIT_DIR.resolve()
        if not path.is_file() or path.parent != audit_resolved:
            return "Report not found.", 404
        return send_from_directory(AUDIT_DIR, filename, as_attachment=False)
    except Exception:
        return "Report not found.", 404


def main():
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")


if __name__ == "__main__":
    main()
