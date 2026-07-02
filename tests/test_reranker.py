import numpy as np
import pytest

from app.services import reranker
from app.services.text_chunker import TextChunk


class FakeRerankerModel:
    """Predictable reranker used for offline tests."""

    def predict(
        self,
        pairs: list[tuple[str, str]],
        **kwargs: object,
    ) -> np.ndarray:
        assert len(pairs) == 2

        return np.array(
            [0.91, 0.32],
            dtype=np.float32,
        )


def create_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id="page-4-sentence-1",
            text="Its backend framework is FastAPI.",
            source="evaluation.pdf",
            page_number=4,
            chunk_index=1,
        ),
        TextChunk(
            chunk_id="page-4-sentence-2",
            text="Its web interface uses Streamlit.",
            source="evaluation.pdf",
            page_number=4,
            chunk_index=2,
        ),
    ]


def test_reranker_orders_passages_by_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: FakeRerankerModel(),
    )

    results = reranker.rerank_chunks(
        query="Which backend framework does the project use?",
        chunks=create_chunks(),
        top_k=2,
    )

    assert len(results) == 2
    assert "FastAPI" in results[0].chunk.text
    assert "Streamlit" in results[1].chunk.text

    assert results[0].score > results[1].score
    assert 0.0 <= results[0].score <= 1.0
    assert 0.0 <= results[1].score <= 1.0


def test_reranker_respects_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: FakeRerankerModel(),
    )

    results = reranker.rerank_chunks(
        query="Which backend framework is used?",
        chunks=create_chunks(),
        top_k=1,
    )

    assert len(results) == 1


def test_empty_chunks_return_empty_results() -> None:
    assert reranker.rerank_chunks(
        query="A valid question",
        chunks=[],
        top_k=3,
    ) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        reranker.rerank_chunks(
            query="A valid question",
            chunks=[],
            top_k=top_k,
        )


def test_backend_terms_favor_backend_sentence() -> None:
    backend_score = reranker.lexical_overlap_score(
        query="Which backend framework does the project use?",
        passage="Its backend framework is FastAPI.",
    )

    interface_score = reranker.lexical_overlap_score(
        query="Which backend framework does the project use?",
        passage="Its web interface uses Streamlit.",
    )

    assert backend_score > interface_score


def test_interface_terms_favor_interface_sentence() -> None:
    interface_score = reranker.lexical_overlap_score(
        query="Which framework is used for the web interface?",
        passage="Its web interface uses Streamlit.",
    )

    backend_score = reranker.lexical_overlap_score(
        query="Which framework is used for the web interface?",
        passage="Its backend framework is FastAPI.",
    )

    assert interface_score > backend_score


class MisleadingRerankerModel:
    """A model that incorrectly prefers the Streamlit sentence."""

    def predict(
        self,
        pairs: list[tuple[str, str]],
        **kwargs: object,
    ) -> np.ndarray:
        return np.array(
            [0.10, 0.90],
            dtype=np.float32,
        )


def test_hybrid_ranking_corrects_backend_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        TextChunk(
            chunk_id="backend",
            text="Its backend framework is FastAPI.",
            source="evaluation.pdf",
            page_number=4,
            chunk_index=1,
        ),
        TextChunk(
            chunk_id="interface",
            text="Its web interface uses Streamlit.",
            source="evaluation.pdf",
            page_number=4,
            chunk_index=2,
        ),
    ]

    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: MisleadingRerankerModel(),
    )

    results = reranker.rerank_chunks(
        query="Which backend framework does the project use?",
        chunks=chunks,
        top_k=2,
    )

    assert results[0].chunk.text == (
        "Its backend framework is FastAPI."
    )