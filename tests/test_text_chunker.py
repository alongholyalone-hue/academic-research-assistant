import pytest

from app.services.pdf_extractor import ExtractedPage
from app.services.text_chunker import TextChunk, chunk_pages


def test_chunk_pages_creates_overlapping_chunks() -> None:
    page = ExtractedPage(
        page_number=1,
        source="sample.pdf",
        text="one two three four five six seven eight nine ten",
    )

    chunks = chunk_pages(
        pages=[page],
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) == 3

    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "four five six seven"
    assert chunks[2].text == "seven eight nine ten"


def test_chunk_metadata_is_preserved() -> None:
    page = ExtractedPage(
        page_number=5,
        source="lecture.pdf",
        text="Artificial intelligence uses computational models.",
    )

    chunks = chunk_pages(
        pages=[page],
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], TextChunk)

    assert chunks[0].source == "lecture.pdf"
    assert chunks[0].page_number == 5
    assert chunks[0].chunk_index == 1
    assert chunks[0].chunk_id == "lecture.pdf-page-5-chunk-1"


def test_multiple_pages_remain_separate() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            source="reading.pdf",
            text="Machine learning identifies patterns in data.",
        ),
        ExtractedPage(
            page_number=2,
            source="reading.pdf",
            text="Responsible AI considers fairness and transparency.",
        ),
    ]

    chunks = chunk_pages(
        pages=pages,
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].source == "reading.pdf"
    assert chunks[1].source == "reading.pdf"


def test_empty_pages_are_skipped() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            source="empty.pdf",
            text="",
        ),
        ExtractedPage(
            page_number=2,
            source="empty.pdf",
            text="This page contains text.",
        ),
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_settings_raise_error(
    chunk_size: int,
    overlap: int,
) -> None:
    page = ExtractedPage(
        page_number=1,
        source="sample.pdf",
        text="Some sample text.",
    )

    with pytest.raises(ValueError):
        chunk_pages(
            pages=[page],
            chunk_size=chunk_size,
            overlap=overlap,
        )
