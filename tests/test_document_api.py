from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import documents
from app.main import app
from app.services.retrieval_pipeline import RetrievalResponse
from app.services.semantic_search import SearchResult
from app.services.text_chunker import TextChunk


client = TestClient(app)


def fake_retrieval_response() -> RetrievalResponse:
    """Create a predictable retrieval result for API tests."""

    chunk = TextChunk(
        chunk_id="lecture.pdf-page-2-chunk-1",
        text=(
            "Responsible artificial intelligence "
            "considers fairness."
        ),
        source="lecture.pdf",
        page_number=2,
        chunk_index=1,
    )

    return RetrievalResponse(
        query="What is responsible AI?",
        source="lecture.pdf",
        page_count=2,
        chunk_count=2,
        results=[
            SearchResult(
                chunk=chunk,
                score=0.91,
            )
        ],
    )


def test_search_document_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_retrieve_from_pdf(
        file_path: str | Path,
        query: str,
        top_k: int,
    ) -> RetrievalResponse:
        path = Path(file_path)

        assert path.exists()
        assert path.name == "lecture.pdf"
        assert query == "What is responsible AI?"
        assert top_k == 1

        return fake_retrieval_response()

    monkeypatch.setattr(
        documents,
        "retrieve_from_pdf",
        fake_retrieve_from_pdf,
    )

    response = client.post(
        "/documents/search",
        data={
            "query": "What is responsible AI?",
            "top_k": "1",
        },
        files={
            "file": (
                "lecture.pdf",
                b"temporary PDF content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["source"] == "lecture.pdf"
    assert body["page_count"] == 2
    assert body["chunk_count"] == 2

    assert len(body["results"]) == 1
    assert body["results"][0]["page_number"] == 2
    assert body["results"][0]["score"] == pytest.approx(
        0.91
    )


def test_non_pdf_file_is_rejected() -> None:
    response = client.post(
        "/documents/search",
        data={
            "query": "What does this document say?",
            "top_k": "3",
        },
        files={
            "file": (
                "notes.txt",
                b"Some ordinary text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only PDF files are supported"
    )


def test_blank_query_is_rejected() -> None:
    response = client.post(
        "/documents/search",
        data={
            "query": "   ",
            "top_k": "3",
        },
        files={
            "file": (
                "lecture.pdf",
                b"temporary PDF content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Query cannot be empty"
    )


def test_invalid_top_k_is_rejected() -> None:
    response = client.post(
        "/documents/search",
        data={
            "query": "A valid question",
            "top_k": "0",
        },
        files={
            "file": (
                "lecture.pdf",
                b"temporary PDF content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "top_k must be between 1 and 10"
    )