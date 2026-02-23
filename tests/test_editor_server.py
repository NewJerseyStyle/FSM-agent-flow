"""Tests for the editor HTTP server endpoints."""

from __future__ import annotations

import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fsm_agent_flow.editor.server import EditorHandler, STATIC_DIR, TEMPLATES


class MockWfile(BytesIO):
    """Mock wfile that captures written data."""
    pass


class FakeRequest:
    """Fake socket request for handler init."""
    def makefile(self, *args, **kwargs):
        return BytesIO()


def make_handler(method: str, path: str, body: dict | None = None) -> EditorHandler:
    """Create a handler with a mocked request for testing."""
    handler = EditorHandler.__new__(EditorHandler)
    handler.initial_file = None
    handler.path = path
    handler.command = method

    # Mock headers
    handler.headers = {}
    handler.wfile = BytesIO()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = MagicMock()
    handler.request_version = "HTTP/1.1"
    handler.close_connection = True

    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        handler.rfile = BytesIO(raw)
        handler.headers = {"Content-Length": str(len(raw))}
    else:
        handler.rfile = BytesIO(b"")
        handler.headers = {}

    # Capture response
    handler._response_code = None
    handler._response_headers = {}
    handler._response_body = b""

    original_send_response = handler.send_response.__func__ if hasattr(handler.send_response, '__func__') else None

    def mock_send_response(code, message=None):
        handler._response_code = code

    def mock_send_header(key, value):
        handler._response_headers[key] = value

    def mock_end_headers():
        pass

    def mock_send_error(code, message=None, explain=None):
        handler._response_code = code

    handler.send_response = mock_send_response
    handler.send_header = mock_send_header
    handler.end_headers = mock_end_headers
    handler.send_error = mock_send_error
    handler.log_message = lambda *a: None

    return handler


def get_response_json(handler: EditorHandler) -> dict:
    """Extract JSON from handler's wfile."""
    handler.wfile.seek(0)
    raw = handler.wfile.read()
    if raw:
        return json.loads(raw)
    return {}


# ── GET endpoint tests ──────────────────────────────────────────────────


class TestGetEndpoints:
    def test_serve_index(self):
        handler = make_handler("GET", "/")
        handler.do_GET()
        assert handler._response_code == 200

    def test_serve_static_file(self):
        handler = make_handler("GET", "/static/editor.js")
        handler.do_GET()
        assert handler._response_code == 200

    def test_static_404(self):
        handler = make_handler("GET", "/static/nonexistent.js")
        handler.do_GET()
        assert handler._response_code == 404

    def test_templates(self):
        handler = make_handler("GET", "/api/templates")
        handler.do_GET()
        assert handler._response_code == 200
        data = get_response_json(handler)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "name" in data[0]
        assert "data" in data[0]

    def test_load_empty(self):
        handler = make_handler("GET", "/api/load")
        handler.do_GET()
        assert handler._response_code == 200
        data = get_response_json(handler)
        assert data["version"] == "2.0"
        assert data["states"] == {}

    def test_load_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"version": "2.0", "objective": "test", "states": {}, "transitions": {}}, f)
            f.flush()
            path = f.name

        try:
            handler = make_handler("GET", f"/api/load?path={path}")
            handler.do_GET()
            assert handler._response_code == 200
            data = get_response_json(handler)
            assert data["objective"] == "test"
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        handler = make_handler("GET", "/api/load?path=/nonexistent/file.json")
        handler.do_GET()
        assert handler._response_code == 404


# ── POST endpoint tests ────────────────────────────────────────────────


class TestPostEndpoints:
    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_save.json")
            workflow = {
                "version": "2.0",
                "objective": "saved",
                "states": {},
                "transitions": {},
            }
            handler = make_handler("POST", "/api/save", {"path": path, "workflow": workflow})
            handler.do_POST()
            assert handler._response_code == 200
            data = get_response_json(handler)
            assert data["ok"] is True

            # Verify file was written
            with open(path) as f:
                saved = json.load(f)
            assert saved["objective"] == "saved"

    def test_save_missing_params(self):
        handler = make_handler("POST", "/api/save", {"path": None, "workflow": None})
        handler.do_POST()
        assert handler._response_code == 400

    def test_export_python(self):
        workflow = {
            "version": "2.0",
            "objective": "Export test",
            "states": {
                "s1": {
                    "objective": "Do something",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 3,
                    "is_initial": True,
                    "is_final": True,
                    "execute_module": None,
                },
            },
            "transitions": {"s1": None},
        }
        handler = make_handler("POST", "/api/export-python", {"workflow": workflow})
        handler.do_POST()
        assert handler._response_code == 200
        data = get_response_json(handler)
        assert "code" in data
        assert "from fsm_agent_flow import" in data["code"]
        # Verify it's valid Python
        compile(data["code"], "test.py", "exec")

    def test_validate_valid(self):
        workflow = {
            "states": {
                "a": {"objective": "Do A", "is_initial": True, "is_final": True},
            },
            "transitions": {"a": None},
        }
        handler = make_handler("POST", "/api/validate", {"workflow": workflow})
        handler.do_POST()
        assert handler._response_code == 200
        data = get_response_json(handler)
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_invalid(self):
        workflow = {
            "states": {
                "a": {"objective": "", "is_initial": False, "is_final": False},
            },
            "transitions": {},
        }
        handler = make_handler("POST", "/api/validate", {"workflow": workflow})
        handler.do_POST()
        assert handler._response_code == 200
        data = get_response_json(handler)
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_404_on_unknown_path(self):
        handler = make_handler("POST", "/api/unknown", {})
        handler.do_POST()
        assert handler._response_code == 404


# ── Templates integrity ────────────────────────────────────────────────


class TestTemplates:
    def test_all_templates_are_valid(self):
        for template in TEMPLATES:
            assert "name" in template
            assert "data" in template
            data = template["data"]
            assert "states" in data
            assert "transitions" in data

            errors = []
            from fsm_agent_flow.schema import validate_workflow_json
            errors = validate_workflow_json(data)
            assert errors == [], f"Template '{template['name']}' has errors: {errors}"
