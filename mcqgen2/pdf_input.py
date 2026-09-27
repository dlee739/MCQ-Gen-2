from __future__ import annotations

import pymupdf


MIN_EXTRACTED_TEXT_CHARACTERS = 20


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract every PDF page into one page-marked text source in memory."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("The uploaded PDF could not be opened for text extraction.") from exc

    pages: list[str] = []
    extracted_characters = 0
    try:
        for page_number, page in enumerate(document, start=1):
            text = (page.get_text("text") or "").strip()
            extracted_characters += len("".join(text.split()))
            pages.append(f"[Page {page_number}]\n{text}".rstrip())
    finally:
        document.close()

    if extracted_characters < MIN_EXTRACTED_TEXT_CHARACTERS:
        raise ValueError(
            "The PDF contains too little extractable text. Choose 'Send original PDF' "
            "to include its page images instead."
        )
    return "\n\n".join(pages)
