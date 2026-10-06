import pytest

from app.llm import parse_json
from app.ocr import Fragment, Line, build_lines, parse_rendered, render_lines, sniff_type


def test_build_lines_groups_rows_and_keeps_wide_gaps():
    frags = [
        Fragment("INV-204", 300, 11, 350, 21, 0.9),
        Fragment("Invoice", 10, 10, 60, 20, 0.9),
        Fragment("No:", 64, 10, 84, 20, 0.9),
        Fragment("Total", 10, 40, 45, 50, 0.5),
    ]
    lines = build_lines(frags)
    assert [text for text, _ in lines] == ["Invoice No:   INV-204", "Total"]
    assert lines[0][1] == pytest.approx(0.9)
    assert lines[1][1] == pytest.approx(0.5)


def test_build_lines_empty():
    assert build_lines([Fragment("  ", 0, 0, 1, 1)]) == []


def test_render_and_parse_round_trip():
    lines = [Line("L1", 1, "Acme Ltd", 1.0), Line("L2", 2, "Total: 120.00", 1.0)]
    text = render_lines(lines)
    assert text == "--- PAGE 1 ---\nL1: Acme Ltd\n--- PAGE 2 ---\nL2: Total: 120.00"
    assert parse_rendered(text) == lines


def test_sniff_type():
    assert sniff_type(b"%PDF-1.7 ...") == "pdf"
    assert sniff_type(b"\xff\xd8\xff\xe0....") == "jpg"
    assert sniff_type(b"\x89PNG\r\n\x1a\n....") == "png"
    assert sniff_type(b"II*\x00....") == "tiff"
    assert sniff_type(b"MZ executable") is None


def test_parse_json_with_fences_and_trailing_comma():
    assert parse_json('```json\n{"a": 1,}\n```') == {"a": 1}


def test_parse_json_with_stray_text():
    assert parse_json('Here you go: {"a": {"b": 2}} Hope that helps!') == {"a": {"b": 2}}


def test_parse_json_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json("I could not read the invoice.")
