from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)


MODEL_NAME = (
    "distilbert/distilbert-base-cased-distilled-squad"
)


@dataclass(frozen=True)
class ExtractedAnswer:
    """An answer span extracted directly from source text."""

    text: str
    confidence: float
    start_character: int
    end_character: int


@lru_cache(maxsize=1)
def get_qa_components() -> tuple[Any, Any]:
    """Load and cache the tokenizer and question-answering model."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    )
    model.eval()

    return tokenizer, model


def select_best_span(
    context: str,
    start_probabilities: Sequence[float],
    end_probabilities: Sequence[float],
    offsets: Sequence[tuple[int, int]],
    sequence_ids: Sequence[int | None],
    max_answer_tokens: int = 40,
) -> ExtractedAnswer:
    """Select the highest-scoring valid answer span."""

    if max_answer_tokens <= 0:
        raise ValueError(
            "max_answer_tokens must be greater than zero"
        )

    best_start: int | None = None
    best_end: int | None = None
    best_score = -1.0

    context_positions = [
        index
        for index, (sequence_id, offset) in enumerate(
            zip(sequence_ids, offsets)
        )
        if sequence_id == 1 and offset[1] > offset[0]
    ]

    for start_index in context_positions:
        final_end = min(
            start_index + max_answer_tokens,
            len(offsets),
        )

        for end_index in range(start_index, final_end):
            if sequence_ids[end_index] != 1:
                continue

            if offsets[end_index][1] <= offsets[end_index][0]:
                continue

            score = (
                float(start_probabilities[start_index])
                * float(end_probabilities[end_index])
            )

            if score > best_score:
                best_start = start_index
                best_end = end_index
                best_score = score

    if best_start is None or best_end is None:
        return ExtractedAnswer(
            text="",
            confidence=0.0,
            start_character=0,
            end_character=0,
        )

    start_character = offsets[best_start][0]
    end_character = offsets[best_end][1]

    answer_text = context[
        start_character:end_character
    ].strip()

    return ExtractedAnswer(
        text=answer_text,
        confidence=best_score,
        start_character=start_character,
        end_character=end_character,
    )


def extract_answer(
    question: str,
    context: str,
    max_answer_tokens: int = 40,
) -> ExtractedAnswer:
    """Extract an answer to a question from supplied context."""

    cleaned_question = question.strip()
    cleaned_context = context.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty")

    if not cleaned_context:
        raise ValueError("Context cannot be empty")

    if max_answer_tokens <= 0:
        raise ValueError(
            "max_answer_tokens must be greater than zero"
        )

    tokenizer, model = get_qa_components()

    encoded = tokenizer(
        cleaned_question,
        cleaned_context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True,
    )

    offsets = [
        (int(start), int(end))
        for start, end in encoded[
            "offset_mapping"
        ][0].tolist()
    ]

    sequence_ids = encoded.sequence_ids(0)

    model_inputs = {
        name: value
        for name, value in encoded.items()
        if name != "offset_mapping"
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    start_probabilities = torch.softmax(
        outputs.start_logits,
        dim=-1,
    )[0].tolist()

    end_probabilities = torch.softmax(
        outputs.end_logits,
        dim=-1,
    )[0].tolist()

    return select_best_span(
        context=cleaned_context,
        start_probabilities=start_probabilities,
        end_probabilities=end_probabilities,
        offsets=offsets,
        sequence_ids=sequence_ids,
        max_answer_tokens=max_answer_tokens,
    )