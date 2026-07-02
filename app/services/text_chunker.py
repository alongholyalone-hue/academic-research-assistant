from dataclasses import dataclass

from app.services.pdf_extractor import ExtractedPage


@dataclass(frozen=True)
class TextChunk:
    """A searchable section of text with citation metadata."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 120,
    overlap: int = 30,
) -> list[TextChunk]:
    """
    Divide extracted PDF pages into overlapping word-based chunks.

    Args:
        pages: Pages returned by the PDF extraction service.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words repeated between neighbouring chunks.

    Returns:
        A list of chunks containing text and citation metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []

    for page in pages:
        words = page.text.split()

        if not words:
            continue

        start = 0
        chunk_index = 1

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append(
                TextChunk(
                    chunk_id=(
                        f"{page.source}-page-{page.page_number}"
                        f"-chunk-{chunk_index}"
                    ),
                    text=chunk_text,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )

            if end == len(words):
                break

            start = end - overlap
            chunk_index += 1

    return chunks
