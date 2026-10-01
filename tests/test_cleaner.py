from fragile_compassion.sft_document_generation.process_book import (
    calculate_data_density_score,
    chunk_text_by_words,
    clean_markdown_text,
    is_data_dense_chunk,
    is_quality_content_chunk,
)


def test_calculate_data_density_score_markdown_tables():
    raw_text = "| Parameter | Value |\n|---|---|\n| Rumen pH | 6.5 |\n| Propionate | 25% |"
    score = calculate_data_density_score(raw_text)
    # 4 table rows * 2 + 2 numbers ("6.5", "25%") = 10
    assert score >= 8


def test_calculate_data_density_score_numerical_expressions():
    raw_text = (
        "Crude protein content was 18.5% with 2.5 kg/day intake and 350 mg "
        "supplementation across 12 test subjects."
    )
    score = calculate_data_density_score(raw_text)
    # Matches 4 quantitative terms: 18.5%, 2.5 kg, 350 mg, 12
    assert score >= 4


def test_calculate_data_density_score_caps_bare_numbers():
    # Test that 30 bare integers hit the 10-item cap (10 * 0.5 = 5 points)
    raw_text = " ".join([str(i) for i in range(30)])
    score = calculate_data_density_score(raw_text)
    assert score == 5


def test_calculate_data_density_score_caps_runaway_numerical_counts():
    # Test that 30 quantitative matches hit the 15-item cap (15 * 2.0 = 30 points)
    raw_text = " ".join([f"{i} mg" for i in range(30)])
    score = calculate_data_density_score(raw_text)
    assert score == 30


def test_is_data_dense_chunk_exceeds_threshold():
    raw_text = (
        "| Nutrient | Level |\n"
        "|---|---|\n"
        "| Calcium | 0.8% |\n"
        "| Phosphorus | 0.4% |\n"
        "Includes 12.5 mg vitamin D3 per 100 kg body weight."
    )
    assert is_data_dense_chunk(raw_text, min_score=5) is True


def test_is_data_dense_chunk_fails_sparse_prose():
    raw_text = "The general consensus remains unchanged despite additional research in this field."
    assert is_data_dense_chunk(raw_text, min_score=5) is False


def test_is_quality_content_chunk_accepts_valid_body_chunk():
    # Generates a valid chunk exceeding the 200-word minimum length check
    words = ["digestive", "enzyme", "secretion", "in", "ruminant", "species"] * 35
    raw_text = " ".join(words)
    assert is_quality_content_chunk(raw_text) is True


def test_is_quality_content_chunk_rejects_short_chunks():
    raw_text = "This chunk is far too short to pass the 200-word minimum threshold."
    assert is_quality_content_chunk(raw_text) is False


def test_is_quality_content_chunk_rejects_front_matter():
    words = ["publication", "data", "for", "academic", "purposes"] * 45
    raw_text = f"ISBN 978-0-123456-78-9\nLibrary of Congress Cataloging\n" + " ".join(words)
    assert is_quality_content_chunk(raw_text) is False


def test_is_quality_content_chunk_rejects_table_of_contents():
    toc_lines = [f"Chapter {i} ................................. {i * 10}" for i in range(10)]
    words = ["content", "overview"] * 100
    raw_text = "\n".join(toc_lines) + "\n\n" + " ".join(words)
    assert is_quality_content_chunk(raw_text) is False


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


def test_chunk_text_by_words_merges_small_trailing_tail():
    # Verify that a 150-word tail paragraph is merged into the previous chunk
    body_paragraph = " ".join(["word"] * 1400)
    tail_paragraph = " ".join(["tail"] * 150)
    full_text = f"{body_paragraph}\n\n{tail_paragraph}"

    chunks = chunk_text_by_words(full_text, chunk_size=1000, min_tail_words=300)

    # Should merge into 1 single chunk rather than creating a 150-word chunk that gets dropped
    assert len(chunks) == 1
    assert "tail" in chunks[0]


def test_calculate_data_density_score_differentiates_units_from_bare_years():
    bare_years_text = "In 2018, 2019, 2020, 2021, and 2022 we published reports."
    units_text = "Added 15.5 mg/kg with 25% yield across 3 trials."

    years_score = calculate_data_density_score(bare_years_text)
    units_score = calculate_data_density_score(units_text)

    # Bare years yield low score (5 * 0.5 = 2.5 -> 2), units yield high score (2 * 2.0 = 4)
    assert units_score > years_score
