from fragile_compassion.sft_document_generation.process_book import (
    chunk_text_by_words,
    clean_markdown_text,
)


def test_remove_running_headers_and_footers():
    raw_text = "Paragraph one.\n\n24 FUNDAMENTAL NUTRITION\n\nParagraph two."
    cleaned = clean_markdown_text(raw_text)
    assert "FUNDAMENTAL NUTRITION" not in cleaned
    assert "Paragraph one.\n\nParagraph two." in cleaned


def test_rejoin_hyphenated_line_wraps():
    raw_text = "Fat digestion involves lipolytic enzymes and bi-\nological processes."
    cleaned = clean_markdown_text(raw_text)
    assert "and biological processes." in cleaned


def test_strip_ocr_graph_axis_clutter():
    raw_text = "Figure 1.2 Data\n\n1.00 0.90 0.80 0.70 0.60 0.50 0.40 OM NDF\n\nThis shows organic matter digestibility."
    cleaned = clean_markdown_text(raw_text)
    assert "1.00 0.90 0.80" not in cleaned
    assert "This shows organic matter digestibility." in cleaned


def test_strip_integer_axis_ticks():
    raw_text = "Figure 2\n\n0 5 10 15 20\n\nWeight gain by week."
    cleaned = clean_markdown_text(raw_text)
    assert cleaned == "Figure 2\n\nWeight gain by week."


def test_keeps_wrapped_prose_line_of_short_words_and_numbers():
    # Regression: the axis-tick rule once deleted any line of 3+ short tokens, so a
    # wrapped line like "at 32 to 34" vanished along with the parameter it carried.
    raw_text = "Keep brooding temperature\nat 32 to 34\ndegrees for the first week."
    cleaned = clean_markdown_text(raw_text)
    assert cleaned == "Keep brooding temperature at 32 to 34 degrees for the first week."


def test_keeps_markdown_table_rows_of_numbers():
    raw_text = "| Age | Temp |\n|---|---|\n| 1 | 32 |\n| 7 | 29 |"
    cleaned = clean_markdown_text(raw_text)
    assert cleaned == raw_text


def test_unwraps_mid_sentence_newlines():
    raw_text = "The duodenal epithelium contains\nspecific cells that secrete\npeptides."
    cleaned = clean_markdown_text(raw_text)
    assert cleaned == "The duodenal epithelium contains specific cells that secrete peptides."


def test_preserves_markdown_headers_and_lists():
    raw_text = "### Duodenum\n* Bullet 1\n* Bullet 2\n\nProse text here."
    cleaned = clean_markdown_text(raw_text)
    assert "### Duodenum" in cleaned
    assert "* Bullet 1" in cleaned
    assert "* Bullet 2" in cleaned

def test_chunk_text_by_words_preserves_paragraphs():
    raw_text = "Header 1\n\nFirst paragraph with some content.\n\nSecond paragraph with more text."
    chunks = chunk_text_by_words(raw_text, chunk_size=100)

    assert len(chunks) == 1
    assert "\n\n" in chunks[0]
    assert chunks[0] == raw_text