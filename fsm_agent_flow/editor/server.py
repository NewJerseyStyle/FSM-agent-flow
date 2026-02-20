"""Stdlib-only HTTP server for the visual workflow editor."""

from __future__ import annotations

import json
import mimetypes
import os
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from ..schema import validate_workflow_json, workflow_to_python

STATIC_DIR = Path(__file__).parent / "static"

# Templates for new workflows
TEMPLATES = [
    {
        "name": "Simple Pipeline",
        "data": {
            "version": "2.0",
            "objective": "Simple two-step pipeline",
            "states": {
                "step1": {
                    "objective": "First step",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 3,
                    "is_initial": True,
                    "is_final": False,
                    "execute_module": None,
                },
                "step2": {
                    "objective": "Second step",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 3,
                    "is_initial": False,
                    "is_final": True,
                    "execute_module": None,
                },
            },
            "transitions": {"step1": "step2", "step2": None},
            "graph_layout": {"step1": [150, 250], "step2": [500, 250]},
        },
    },
    {
        "name": "Research & Write",
        "data": {
            "version": "2.0",
            "objective": "Research a topic and write a report",
            "states": {
                "research": {
                    "objective": "Gather information on the topic",
                    "key_results": [
                        {
                            "name": "has_content",
                            "description": "At least 200 chars",
                            "check": "len(str(output)) >= 200",
                        }
                    ],
                    "tools": ["search_web"],
                    "max_retries": 2,
                    "is_initial": True,
                    "is_final": False,
                    "execute_module": None,
                },
                "writing": {
                    "objective": "Write a structured report",
                    "key_results": [
                        {
                            "name": "has_sections",
                            "description": "Has clear sections with headings",
                            "check": "str(output).count('#') >= 2",
                        }
                    ],
                    "tools": [],
                    "max_retries": 3,
                    "is_initial": False,
                    "is_final": True,
                    "execute_module": None,
                },
            },
            "transitions": {"research": "writing", "writing": None},
            "graph_layout": {"research": [150, 250], "writing": [500, 250]},
        },
    },
    {
        "name": "OODA Loop",
        "data": {
            "version": "2.0",
            "objective": "OODA decision loop",
            "states": {
                "observe": {
                    "objective": "Gather and observe relevant information",
                    "key_results": [
                        {
                            "name": "observations",
                            "description": "Collected observations",
                            "check": "len(str(output)) > 50",
                        }
                    ],
                    "tools": [],
                    "max_retries": 2,
                    "is_initial": True,
                    "is_final": False,
                    "execute_module": None,
                },
                "orient": {
                    "objective": "Analyze observations and form understanding",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 2,
                    "is_initial": False,
                    "is_final": False,
                    "execute_module": None,
                },
                "decide": {
                    "objective": "Make a decision based on analysis",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 2,
                    "is_initial": False,
                    "is_final": False,
                    "execute_module": None,
                },
                "act": {
                    "objective": "Execute the decision",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 2,
                    "is_initial": False,
                    "is_final": True,
                    "execute_module": None,
                },
            },
            "transitions": {
                "observe": "orient",
                "orient": "decide",
                "decide": "act",
                "act": None,
            },
            "graph_layout": {
                "observe": [100, 250],
                "orient": [350, 250],
                "decide": [600, 250],
                "act": [850, 250],
            },
        },
    },
]


class EditorHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the workflow editor."""

    def __init__(self, *args, initial_file: str | None = None, **kwargs):
        self.initial_file = initial_file
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet logging — only show errors."""
        if args and isinstance(args[0], str) and args[0].startswith("4"):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._serve_file(STATIC_DIR / "index.html", "text/html")
        elif self.path.startswith("/static/"):
            rel_path = self.path[len("/static/"):]
            # Strip query string
            if "?" in rel_path:
                rel_path = rel_path.split("?")[0]
            file_path = STATIC_DIR / rel_path
            if file_path.is_file() and STATIC_DIR in file_path.resolve().parents:
                content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                self._serve_file(file_path, content_type)
            else:
                self.send_error(404)
        elif self.path.startswith("/api/load"):
            self._handle_load()
        elif self.path == "/api/templates":
            self._send_json(TEMPLATES)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self._read_body()
        if body is None:
            return

        if self.path == "/api/save":
            self._handle_save(body)
        elif self.path == "/api/export-python":
            self._handle_export_python(body)
        elif self.path == "/api/validate":
            self._handle_validate(body)
        else:
            self.send_error(404)

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json({"error": str(e)}, status=400)
            return None

    def _handle_load(self) -> None:
        # Parse query string for path
        path = None
        if "?" in self.path:
            query = self.path.split("?", 1)[1]
            for param in query.split("&"):
                if param.startswith("path="):
                    path = param[5:]
                    break

        if not path:
            # Return empty workflow
            self._send_json({
                "version": "2.0",
                "objective": "",
                "states": {},
                "transitions": {},
                "graph_layout": {},
            })
            return

        try:
            resolved = Path(path).resolve()
            with open(resolved, "r") as f:
                data = json.load(f)
            self._send_json(data)
        except FileNotFoundError:
            self._send_json({"error": f"File not found: {path}"}, status=404)
        except json.JSONDecodeError as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, status=400)

    def _handle_save(self, body: dict) -> None:
        path = body.get("path")
        workflow = body.get("workflow")
        if not path or not workflow:
            self._send_json({"error": "Missing 'path' or 'workflow'"}, status=400)
            return

        try:
            resolved = Path(path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved, "w") as f:
                json.dump(workflow, f, indent=2)
            self._send_json({"ok": True, "path": str(resolved)})
        except OSError as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_export_python(self, body: dict) -> None:
        workflow = body.get("workflow")
        if not workflow:
            self._send_json({"error": "Missing 'workflow'"}, status=400)
            return

        try:
            code = workflow_to_python(workflow)
            self._send_json({"code": code})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_validate(self, body: dict) -> None:
        workflow = body.get("workflow")
        if not workflow:
            self._send_json({"error": "Missing 'workflow'"}, status=400)
            return

        errors = validate_workflow_json(workflow)
        self._send_json({"errors": errors, "valid": len(errors) == 0})

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8742, initial_file: str | None = None) -> None:
    """Start the editor HTTP server."""
    handler = partial(EditorHandler, initial_file=initial_file)
    server = HTTPServer(("127.0.0.1", port), handler)
    print(f"Editor server running on http://127.0.0.1:{port}")
    server.serve_forever()
