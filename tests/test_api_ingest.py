import io
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
import pymupdf

from app.main import app
from app.config import Settings
from app.errors import AppError
from app.ocr import DocText, Line
from app.schemas import Extraction, FieldValue


def make_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    b = doc.tobytes()
    doc.close()
    return b


def test_openapi_schema_has_binary_file_array():
    schema = app.openapi()
    ingest_body = schema["components"]["schemas"]["Body_ingest_ingest_post"]
    files_prop = ingest_body["properties"]["files"]
    assert files_prop["type"] == "array"
    assert files_prop["items"]["format"] == "binary"
    assert files_prop["items"]["contentMediaType"] == "application/octet-stream"

    extract_body = schema["components"]["schemas"]["Body_extract_extract_post"]
    file_prop = extract_body["properties"]["file"]
    assert file_prop["format"] == "binary"
    assert file_prop["contentMediaType"] == "application/octet-stream"


def test_ingest_requires_api_key(monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)
    # mock settings on app.state
    settings = Settings(_env_file=None, api_key="secret-key")
    app.state.settings = settings

    pdf_data = make_pdf_bytes("INVOICE\nInvoice No: 123\nTotal: 10.00\nDate: 2026-01-01\nVendor: Acme")
    res = client.post(
        "/ingest",
        files=[("files", ("invoice1.pdf", pdf_data, "application/pdf"))],
    )
    assert res.status_code == 401
    assert "Missing or wrong API key" in res.json()["error"]


def test_ingest_multiple_files_batch(monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)
    settings = Settings(_env_file=None, api_key="secret-key")
    app.state.settings = settings

    mock_pipeline = MagicMock()
    mock_pipeline.ingest.side_effect = [
        {
            "request_id": "req-1",
            "doc_id": "doc-001",
            "filename": "invoice1.pdf",
            "status": "pending",
            "document_type": "invoice",
            "data": {"fields": {"invoice_number": {"value": "INV-1"}}},
            "validation_warnings": [],
            "meta": {"processing_time_ms": 100},
            "error": None,
        },
        {
            "request_id": "req-1",
            "doc_id": "doc-002",
            "filename": "invoice2.pdf",
            "status": "pending",
            "document_type": "invoice",
            "data": {"fields": {"invoice_number": {"value": "INV-2"}}},
            "validation_warnings": [],
            "meta": {"processing_time_ms": 120},
            "error": None,
        },
    ]
    app.state.pipeline = mock_pipeline

    pdf1 = make_pdf_bytes("Invoice 1")
    pdf2 = make_pdf_bytes("Invoice 2")

    res = client.post(
        "/ingest",
        headers={"X-API-Key": "secret-key"},
        files=[
            ("files", ("invoice1.pdf", pdf1, "application/pdf")),
            ("files", ("invoice2.pdf", pdf2, "application/pdf")),
        ],
    )
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] == "invoice1.pdf"
    assert data["results"][0]["doc_id"] == "doc-001"
    assert data["results"][1]["filename"] == "invoice2.pdf"
    assert data["results"][1]["doc_id"] == "doc-002"
    assert mock_pipeline.ingest.call_count == 2


def test_ingest_batch_partial_failure_does_not_fail_whole_batch(monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)
    settings = Settings(_env_file=None, api_key="secret-key")
    app.state.settings = settings

    mock_pipeline = MagicMock()
    mock_pipeline.ingest.side_effect = [
        {
            "request_id": "req-1",
            "doc_id": "doc-001",
            "filename": "invoice1.pdf",
            "status": "pending",
            "document_type": "invoice",
            "data": {"fields": {}},
            "validation_warnings": [],
            "meta": {},
            "error": None,
        },
        AppError(400, "Corrupt or unreadable PDF"),
    ]
    app.state.pipeline = mock_pipeline

    pdf1 = make_pdf_bytes("Invoice 1")
    bad_pdf = b"not a valid pdf data"

    res = client.post(
        "/ingest",
        headers={"X-API-Key": "secret-key"},
        files=[
            ("files", ("good.pdf", pdf1, "application/pdf")),
            ("files", ("bad.pdf", bad_pdf, "application/pdf")),
        ],
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["doc_id"] == "doc-001"
    assert data["results"][0]["status"] == "pending"
    assert data["results"][1]["status"] == "error"
    assert "Corrupt or unreadable PDF" in data["results"][1]["error"]
    assert data["results"][1]["http_status"] == 400
