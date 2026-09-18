"""Document loading, cleaning and ingestion."""

from __future__ import annotations

import json

import pytest

from ai_studio.core.errors import IngestionError
from ai_studio.data import ingestion
from ai_studio.data.loaders import load_document
from ai_studio.data.preprocessing import CleaningOptions, clean_text


def test_text_loader_reads_plain_text(tmp_docs):
    path = tmp_docs / "note.txt"
    path.write_text("Hello world.\n\nSecond paragraph.")
    document = load_document(path)
    assert "Hello world" in document.text
    assert document.doc_type == "txt"
    assert document.word_count == 4


def test_markdown_loader_uses_heading_as_title(tmp_docs):
    path = tmp_docs / "doc.md"
    path.write_text("# Real Title\n\nBody text here.")
    document = load_document(path)
    assert document.title == "Real Title"
    assert document.meta["heading_count"] == 1


def test_json_loader_counts_records_and_malformed_lines(tmp_docs):
    path = tmp_docs / "data.jsonl"
    path.write_text('{"text": "one"}\n{"text": "two"}\nnot json\n')
    document = load_document(path)
    assert document.meta["records"] == 2
    assert document.meta["malformed_lines"] == 1


def test_csv_loader_extracts_columns(tmp_docs):
    path = tmp_docs / "table.csv"
    path.write_text("question,answer\nWhat is 2+2?,4\nCapital of France?,Paris\n")
    document = load_document(path)
    assert document.meta["columns"] == ["question", "answer"]
    assert "Paris" in document.text


def test_unsupported_extension_is_rejected(tmp_docs):
    path = tmp_docs / "weights.bin"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(IngestionError):
        load_document(path)


def test_cleaning_removes_page_numbers_and_joins_wrapped_lines():
    raw = "Chapter One\n\n1\n\nThis line was bro-\nken by the extractor and continues\nhere.\n\n2\n"
    result = clean_text(raw, CleaningOptions.for_type("pdf"))
    assert "broken by the extractor" in result.text
    assert result.report.page_numbers_removed >= 2
    assert "\n1\n" not in result.text


def test_cleaning_preserves_code_blocks():
    raw = "Text.\n\n```python\nx  =  1    # spaced\n```\n"
    result = clean_text(raw, CleaningOptions(preserve_code_blocks=True))
    assert "x  =  1    # spaced" in result.text


def test_cleaning_removes_duplicate_paragraphs():
    paragraph = "This is a long enough paragraph to be considered for deduplication by the cleaner."
    result = clean_text(f"{paragraph}\n\n{paragraph}\n\nUnique tail paragraph that is also long enough.")
    assert result.report.duplicate_paragraphs_removed == 1


def test_import_keeps_original_and_writes_cleaned_separately(tmp_docs):
    path = tmp_docs / "source.md"
    original = "# Title\n\nBody.\n\nBody.\n"
    path.write_text(original)
    record = ingestion.import_file(path)

    from pathlib import Path

    assert Path(record["original_path"]).read_text() == original  # untouched
    assert Path(record["cleaned_path"]).exists()
    assert record["cleaned_path"] != record["original_path"]


def test_import_text_and_delete(tmp_docs):
    record = ingestion.import_text("Some pasted article content.", title="Pasted")
    assert record["source"] == "paste"
    assert ingestion.document_text(record["id"]).strip()
    ingestion.delete_document(record["id"])
    from ai_studio.core.errors import NotFoundError

    with pytest.raises(NotFoundError):
        ingestion.document_text(record["id"])


def test_folder_import_survives_a_bad_file(tmp_docs):
    (tmp_docs / "good.txt").write_text("Readable content here.")
    (tmp_docs / "bad.json").write_text("{not valid json")
    outcome = ingestion.import_folder(tmp_docs, recursive=False)
    assert len(outcome.imported) >= 1
    assert any("bad.json" in name for name, _ in outcome.failed)
