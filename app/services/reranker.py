import re
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.services.semantic_search import SearchResult
from app.services.text_chunker import TextChunk


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

LEXICAL_WEIGHT = 0.80
SEMANTIC_WEIGHT = 1.0 - LEXICAL_WEIGHT

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "is",
    "of",
    "the",
    "this",
    "to",
    "use",
    "used",
    "uses",
    "what",
    "which",
}


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """Load and cache the passage reranking model."""

    return CrossEncoder(
        MODEL_NAME,
        activation_fn=torch.nn.Sigmoid(),
    )


def tokenize_for_overlap(text: str) -> set[str]:
    """Convert text into meaningful lowercase tokens."""

    tokens = set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )

    return {
        token
        for token in tokens
        if token not in STOP_WORDS
    }


def lexical_overlap_score(
    query: str,
    passage: str,
) -> float:
    """Measure how many meaningful query terms occur in a passage."""

    query_tokens = tokenize_for_overlap(query)
    passage_tokens = tokenize_for_overlap(passage)

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(passage_tokens)

    return len(overlap) / len(query_tokens)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize reranker scores to the range zero to one."""

    if len(scores) == 1:
        return np.ones_like(scores, dtype=np.float32)

    minimum = float(scores.min())
    maximum = float(scores.max())

    if np.isclose(minimum, maximum):
        return np.full_like(
            scores,
            0.5,
            dtype=np.float32,
        )

    return (
        (scores - minimum)
        / (maximum - minimum)
    ).astype(np.float32)


def rerank_chunks(
    query: str,
    chunks: list[TextChunk],
    top_k: int,
) -> list[SearchResult]:
    """
    Rank chunks using semantic relevance and lexical overlap.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    model = get_reranker_model()

    pairs = [
        (cleaned_query, chunk.text)
        for chunk in chunks
    ]

    raw_semantic_scores = np.asarray(
        model.predict(
            pairs,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    ).reshape(-1)

    semantic_scores = normalize_scores(
        raw_semantic_scores
    )

    lexical_scores = np.asarray(
        [
            lexical_overlap_score(
                query=cleaned_query,
                passage=chunk.text,
            )
            for chunk in chunks
        ],
        dtype=np.float32,
    )

    hybrid_scores = (
        LEXICAL_WEIGHT * lexical_scores
        + SEMANTIC_WEIGHT * semantic_scores
    )

    ranked_indices = np.argsort(
        hybrid_scores
    )[::-1]

    selected_indices = ranked_indices[
        : min(top_k, len(chunks))
    ]

    return [
        SearchResult(
            chunk=chunks[index],
            score=float(hybrid_scores[index]),
        )
        for index in selected_indices
    ]