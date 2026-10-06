from datetime import date

import pytest

from app.labels import (
    find_candidates, normalize_label, parse_date, text_matches_pattern, to_number, value_pattern,
)


@pytest.mark.parametrize("raw, expected", [
    ("Bill No. :", "bill no"),
    ("P.O. Number", "po number"),
    ("Inv #", "inv #"),
    ("  VAT   Reg. No.", "vat reg no"),
    ("VAT @ 20%", "vat 20%"),
    ("Sub-Total", "sub total"),
])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("03/04/2026", date(2026, 4, 3)),  # day first
    ("12.03.26", date(2026, 3, 12)),
    ("3rd April 2026", date(2026, 4, 3)),
    ("12-Mar-2026", date(2026, 3, 12)),
    ("March 12, 2026", date(2026, 3, 12)),
    ("2026-03-12", date(2026, 3, 12)),
])
def test_parse_date(raw, expected):
    assert parse_date(raw)[0] == expected


def test_parse_date_rejects_text():
    assert parse_date("INV-204") is None


@pytest.mark.parametrize("raw, expected", [
    ("£1,234.50", 1234.5), ("20%", 20.0), ("(12.00)", -12.0), (15, 15.0), ("abc", None), ("INV-204", None),
])
def test_to_number(raw, expected):
    assert to_number(raw) == expected


def test_value_pattern_never_keeps_the_value():
    assert value_pattern("INV-204") == "INV-###"
    assert value_pattern("12/03/2026", "date") == "DD/MM/YYYY"
    assert value_pattern(1180.0, "number") == "NUMBER"
    assert value_pattern("Acme Trading Ltd") == "TEXT"


def test_text_matches_pattern():
    assert text_matches_pattern("INV-1001", "INV-###")
    assert not text_matches_pattern("PO 55", "INV-###")
    assert text_matches_pattern("12 March 2026", "DATE")
    assert not text_matches_pattern("INV-204", "DATE")
    assert text_matches_pattern("anything", None)


def test_find_candidates():
    lines = [
        ("L1", "Invoice No: INV-204   Date: 12/03/2026"),
        ("L2", "Description   Qty   Unit Price   Amount"),
        ("L3", "Widget   2   50.00   100.00"),
        ("L4", "Total Due £120.00"),
    ]
    found = {(c.line_id, c.text, c.kind) for c in find_candidates(lines)}
    assert ("L1", "Invoice No", "explicit") in found
    assert ("L1", "Date", "explicit") in found
    assert {("L2", t, "table") for t in ("Description", "Qty", "Unit Price", "Amount")} <= found
    assert ("L4", "Total Due", "prefix") in found
    assert not any(kind == "table" for line_id, _, kind in found if line_id != "L2")


def test_candidate_value_text():
    cands = find_candidates([("L1", "Invoice No: INV-204"), ("L2", "Due Date"), ("L3", "30/04/2026")])
    by_text = {c.text: c for c in cands}
    assert by_text["Invoice No"].value_text == "INV-204"
    assert by_text["Due Date"].value_text == "30/04/2026"  # label above its value
