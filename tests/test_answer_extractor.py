import pytest

from app.services.answer_extractor import (
    extract_answer,
    select_best_span,
)


def test_select_best_span_extracts_context_answer() -> None:
    context = "The answer is 2."

    offsets = [
        (0, 0),
        (0, 4),
        (0, 0),
        (0, 3),
        (4, 10),
        (11, 13),
        (14, 15),
        (15, 16),
        (0, 0),
    ]

    sequence_ids = [
        None,
        0,
        None,
        1,
        1,
        1,
        1,
        1,
        None,
    ]

    start_probabilities = [0.0] * len(offsets)
    end_probabilities = [0.0] * len(offsets)

    start_probabilities[6] = 0.9
    end_probabilities[6] = 0.8

    result = select_best_span(
        context=context,
        start_probabilities=start_probabilities,
        end_probabilities=end_probabilities,
        offsets=offsets,
        sequence_ids=sequence_ids,
    )

    assert result.text == "2"
    assert result.confidence == pytest.approx(0.72)
    assert result.start_character == 14
    assert result.end_character == 15


def test_question_tokens_are_not_used_as_answers() -> None:
    context = "The answer is 2."

    offsets = [
        (0, 0),
        (0, 4),
        (0, 0),
        (0, 3),
        (4, 10),
        (11, 13),
        (14, 15),
        (0, 0),
    ]

    sequence_ids = [
        None,
        0,
        None,
        1,
        1,
        1,
        1,
        None,
    ]

    start_probabilities = [
        0.0,
        0.99,
        0.0,
        0.0,
        0.0,
        0.0,
        0.7,
        0.0,
    ]

    end_probabilities = [
        0.0,
        0.99,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6,
        0.0,
    ]

    result = select_best_span(
        context=context,
        start_probabilities=start_probabilities,
        end_probabilities=end_probabilities,
        offsets=offsets,
        sequence_ids=sequence_ids,
    )

    assert result.text == "2"


@pytest.mark.parametrize(
    ("question", "context", "expected_message"),
    [
        ("   ", "Some context.", "Question cannot be empty"),
        ("A question?", "   ", "Context cannot be empty"),
    ],
)
def test_blank_input_is_rejected(
    question: str,
    context: str,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        extract_answer(
            question=question,
            context=context,
        )


def test_invalid_max_answer_length_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_answer_tokens must be greater than zero",
    ):
        select_best_span(
            context="Some context.",
            start_probabilities=[],
            end_probabilities=[],
            offsets=[],
            sequence_ids=[],
            max_answer_tokens=0,
        )