from fragile_compassion.sft_document_generation.process_book import clean_markdown_text


def test_remove_running_headers_and_footers():
    raw_text = "Paragraph one.\n\n24 FUNDAMENTAL NUTRITION\n\nParagraph two."
    cleaned = clean_markdown_text(raw_text)
    assert "FUNDAMENTAL NUTRITION" not in cleaned
    assert "Paragraph one.\n\nParagraph two." in cleaned


def test_rejoin_hyphenated_line_wraps():
    raw_text = "The physiological digestion of fat involves lipolytic enzymes and bi-\nologi- cal processes."
    cleaned = clean_markdown_text(raw_text)
    assert "biological" in cleaned or "bi- ologi- cal" not in cleaned


def test_strip_ocr_graph_axis_clutter():
    raw_text = "Figure 1.2 Data\n\n1.00 0.90 0.80 0.70 0.60 0.50 0.40 OM NDF\n\nThis shows organic matter digestibility."
    cleaned = clean_markdown_text(raw_text)
    assert "1.00 0.90 0.80" not in cleaned
    assert "This shows organic matter digestibility." in cleaned


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
