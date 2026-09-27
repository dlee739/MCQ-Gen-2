from __future__ import annotations

import pymupdf
import pytest

from mcqgen2.pdf_input import extract_pdf_text


def make_pdf(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    contents = document.tobytes()
    document.close()
    return contents


def test_extract_pdf_text_keeps_all_pages_and_markers() -> None:
    extracted = extract_pdf_text(make_pdf("First page source text.", "Second page."))

    assert "[Page 1]\nFirst page source text." in extracted
    assert "[Page 2]\nSecond page." in extracted


def test_extract_pdf_text_rejects_image_only_or_empty_pdf() -> None:
    with pytest.raises(ValueError, match="too little extractable text"):
        extract_pdf_text(make_pdf(""))


def test_extract_pdf_text_rejects_unreadable_pdf() -> None:
    with pytest.raises(ValueError, match="could not be opened"):
        extract_pdf_text(b"%PDF-1.7\nnot actually a PDF")
