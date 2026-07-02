import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.answer_pipeline import answer_from_pdf
from app.services.retrieval_pipeline import retrieve_from_pdf


EVALUATION_DIRECTORY = PROJECT_ROOT / "evaluation"
QUESTIONS_PATH = EVALUATION_DIRECTORY / "questions.json"
GENERATED_DIRECTORY = EVALUATION_DIRECTORY / "generated"
PDF_PATH = GENERATED_DIRECTORY / "evaluation_document.pdf"
JSON_RESULTS_PATH = EVALUATION_DIRECTORY / "results.json"
MARKDOWN_RESULTS_PATH = EVALUATION_DIRECTORY / "results.md"


DOCUMENT_PAGES = [
    (
        "The academic library opens at 8:00 AM on weekdays. "
        "The library closes at 9:00 PM."
    ),
    (
        "Students receive fifteen vacation days each year. "
        "Remote work is permitted on Tuesdays and Thursdays."
    ),
    (
        "Responsible artificial intelligence requires fairness, "
        "transparency, and human oversight."
    ),
    (
    "The Academic Research Assistant is built with Python. "
    "Its backend framework is FastAPI. "
    "Its web interface uses Streamlit."
    ),
]


def normalize_text(text: str) -> str:
    """Normalize text before comparing predicted and expected answers."""

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def create_evaluation_pdf(file_path: Path) -> None:
    """Create a controlled four-page PDF for repeatable evaluation."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    document = canvas.Canvas(str(file_path))

    for page_text in DOCUMENT_PAGES:
        text_object = document.beginText(72, 720)
        text_object.setLeading(18)

        for line in textwrap.wrap(page_text, width=75):
            text_object.textLine(line)

        document.drawText(text_object)
        document.showPage()

    document.save()


def load_questions() -> list[dict[str, Any]]:
    """Load the evaluation questions."""

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def answer_contains_expected(
    predicted_answer: str,
    expected_text: str,
) -> bool:
    """Check whether an expected answer appears in the prediction."""

    normalized_prediction = normalize_text(predicted_answer)
    normalized_expected = normalize_text(expected_text)

    return normalized_expected in normalized_prediction


def evaluate_question(
    item: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate retrieval, answering, citation, or refusal behaviour."""

    question = item["question"]
    should_answer = item["should_answer"]
    expected_text = item["expected_text"]
    expected_page = item["expected_page"]

    retrieval = retrieve_from_pdf(
        file_path=PDF_PATH,
        query=question,
        top_k=3,
    )

    retrieved_pages = [
        result.chunk.page_number
        for result in retrieval.results
    ]

    answer = answer_from_pdf(
        file_path=PDF_PATH,
        question=question,
        top_k=3,
    )

    retrieval_hit = None
    answer_correct = None
    citation_correct = None
    refusal_correct = None

    if should_answer:
        retrieval_hit = expected_page in retrieved_pages

        answer_correct = (
            answer.answered
            and answer_contains_expected(
                answer.answer,
                expected_text,
            )
        )

        citation_correct = (
            answer.citation is not None
            and answer.citation.page_number == expected_page
        )

        passed = bool(
            retrieval_hit
            and answer_correct
            and citation_correct
        )
    else:
        refusal_correct = not answer.answered
        passed = refusal_correct

    return {
        "id": item["id"],
        "question": question,
        "should_answer": should_answer,
        "expected_text": expected_text,
        "expected_page": expected_page,
        "retrieved_pages": retrieved_pages,
        "retrieval_hit": retrieval_hit,
        "predicted_answer": answer.answer,
        "answered": answer.answered,
        "answer_correct": answer_correct,
        "citation_page": (
            answer.citation.page_number
            if answer.citation is not None
            else None
        ),
        "citation_correct": citation_correct,
        "refusal_correct": refusal_correct,
        "answer_confidence": answer.answer_confidence,
        "retrieval_score": answer.retrieval_score,
        "passed": passed,
    }


def safe_rate(correct: int, total: int) -> float:
    """Return a rate without dividing by zero."""

    return correct / total if total else 0.0


def calculate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Calculate aggregate evaluation metrics."""

    answerable = [
        result
        for result in results
        if result["should_answer"]
    ]

    unsupported = [
        result
        for result in results
        if not result["should_answer"]
    ]

    retrieval_correct = sum(
        result["retrieval_hit"] is True
        for result in answerable
    )

    answer_correct = sum(
        result["answer_correct"] is True
        for result in answerable
    )

    citation_correct = sum(
        result["citation_correct"] is True
        for result in answerable
    )

    refusal_correct = sum(
        result["refusal_correct"] is True
        for result in unsupported
    )

    overall_correct = sum(
        result["passed"] is True
        for result in results
    )

    return {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "unsupported_questions": len(unsupported),
        "retrieval_hit_at_3": safe_rate(
            retrieval_correct,
            len(answerable),
        ),
        "answer_accuracy": safe_rate(
            answer_correct,
            len(answerable),
        ),
        "citation_accuracy": safe_rate(
            citation_correct,
            len(answerable),
        ),
        "refusal_accuracy": safe_rate(
            refusal_correct,
            len(unsupported),
        ),
        "overall_success_rate": safe_rate(
            overall_correct,
            len(results),
        ),
    }


def escape_markdown(value: object) -> str:
    """Make a value safe for a Markdown table."""

    if value is None:
        return "—"

    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_report(
    metrics: dict[str, float | int],
    results: list[dict[str, Any]],
) -> None:
    """Write a human-readable evaluation report."""

    lines = [
        "# Evaluation Results",
        "",
        "This evaluation uses a controlled four-page synthetic PDF.",
        "",
        "## Summary",
        "",
        (
            f"- Total questions: "
            f"{metrics['total_questions']}"
        ),
        (
            f"- Retrieval Hit@3: "
            f"{metrics['retrieval_hit_at_3']:.1%}"
        ),
        (
            f"- Answer accuracy: "
            f"{metrics['answer_accuracy']:.1%}"
        ),
        (
            f"- Citation accuracy: "
            f"{metrics['citation_accuracy']:.1%}"
        ),
        (
            f"- Unsupported-question refusal accuracy: "
            f"{metrics['refusal_accuracy']:.1%}"
        ),
        (
            f"- Overall success rate: "
            f"{metrics['overall_success_rate']:.1%}"
        ),
        "",
        "## Detailed Results",
        "",
        (
            "| ID | Expected | Predicted | Retrieved pages | "
            "Citation page | Passed |"
        ),
        "|---|---|---|---|---|---|",
    ]

    for result in results:
        lines.append(
            "| "
            f"{escape_markdown(result['id'])} | "
            f"{escape_markdown(result['expected_text'])} | "
            f"{escape_markdown(result['predicted_answer'])} | "
            f"{escape_markdown(result['retrieved_pages'])} | "
            f"{escape_markdown(result['citation_page'])} | "
            f"{'Yes' if result['passed'] else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The scores measure performance on a small controlled "
                "evaluation set and should not be interpreted as general "
                "real-world accuracy."
            ),
            "",
            (
                "The evaluation is intended to identify retrieval, "
                "answer-extraction, citation, and refusal weaknesses "
                "that require further testing."
            ),
        ]
    )

    MARKDOWN_RESULTS_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete evaluation."""

    print("Creating evaluation PDF...")
    create_evaluation_pdf(PDF_PATH)

    questions = load_questions()
    results: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        print(
            f"[{index}/{len(questions)}] "
            f"{item['question']}"
        )

        result = evaluate_question(item)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {status}: "
            f"{result['predicted_answer']}"
        )

    metrics = calculate_metrics(results)

    output = {
        "metrics": metrics,
        "results": results,
    }

    JSON_RESULTS_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(metrics, results)

    print()
    print("Evaluation complete.")
    print(
        "Retrieval Hit@3:",
        f"{metrics['retrieval_hit_at_3']:.1%}",
    )
    print(
        "Answer accuracy:",
        f"{metrics['answer_accuracy']:.1%}",
    )
    print(
        "Citation accuracy:",
        f"{metrics['citation_accuracy']:.1%}",
    )
    print(
        "Refusal accuracy:",
        f"{metrics['refusal_accuracy']:.1%}",
    )
    print(
        "Overall success:",
        f"{metrics['overall_success_rate']:.1%}",
    )
    print()
    print(f"Markdown report: {MARKDOWN_RESULTS_PATH}")
    print(f"JSON report: {JSON_RESULTS_PATH}")


if __name__ == "__main__":
    main()