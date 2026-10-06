from datetime import date

from app.ocr import Line
from app.schemas import Extraction
from app.validation import (
    check_iban, check_uk_vat_number, needs_review, normalize, run_checks, score_confidence,
)

TODAY = date(2026, 10, 4)

LINES = [
    Line("L1", 1, "Acme Trading Ltd", 1.0),
    Line("L2", 1, "Invoice No: INV-204   Date: 12/03/2026", 1.0),
    Line("L3", 1, "Widget   2   50.00   100.00", 1.0),
    Line("L4", 1, "Subtotal   100.00", 1.0),
    Line("L5", 1, "VAT @ 20%   20.00", 1.0),
    Line("L6", 1, "Total Due   £120.00", 1.0),
]


def make(**overrides) -> Extraction:
    fields = {
        "vendor_name": {"value": "Acme Trading Ltd", "line": "L1"},
        "invoice_number": {"value": "INV-204", "line": "L2", "label": "Invoice No"},
        "invoice_date": {"value": "12/03/2026", "line": "L2", "label": "Date"},
        "subtotal": {"value": "100.00", "line": "L4"},
        "vat_rate": {"value": "20%", "line": "L5"},
        "vat_amount": {"value": 20, "line": "L5"},
        "total_amount": {"value": "£120.00", "line": "L6"},
    }
    fields.update(overrides)
    return Extraction.model_validate({
        "fields": fields,
        "line_items": [{"line": "L3", "description": "Widget", "quantity": 2, "unit_price": "50.00",
                        "line_total": 100}],
    })


def check(ex: Extraction):
    issues = normalize(ex) + run_checks(ex, LINES, check_labels=True, today=TODAY)
    score_confidence(ex, LINES, issues)
    return issues


def test_clean_invoice_passes_everything():
    ex = make()
    assert check(ex) == []
    assert ex.fields["invoice_date"].value == "2026-03-12"
    assert ex.fields["total_amount"].value == 120.0
    assert all(fv.confidence == "high" for fv in ex.fields.values())
    assert not needs_review(ex, [])


def test_wrong_total_is_flagged_not_fixed():
    ex = make(total_amount={"value": 125, "line": "L6"})
    issues = check(ex)
    assert any(i.kind == "math" and "total_amount" in i.fields for i in issues)
    assert ex.fields["total_amount"].value == 125.0
    assert needs_review(ex, issues)


def test_vat_amount_must_match_rate():
    issues = check(make(vat_amount={"value": 19, "line": "L5"}))
    assert any(i.kind == "math" and "vat_rate" in i.fields for i in issues)


def test_non_uk_vat_rate():
    issues = check(make(vat_rate={"value": 18, "line": "L5"}))
    assert any(i.kind == "format" and i.fields == ["vat_rate"] for i in issues)


def test_line_item_math():
    ex = make()
    ex.line_items[0].quantity = 3
    assert any("line item 1" in i.message for i in check(ex))


def test_due_date_before_invoice_date():
    issues = check(make(due_date={"value": "2026-03-01", "line": "L2"}))
    assert any(i.kind == "date" and "due_date" in i.fields for i in issues)


def test_value_not_on_cited_line_gets_low_confidence():
    ex = make(invoice_number={"value": "INV-999", "line": "L2"})
    issues = check(ex)
    assert any(i.kind == "trace" and i.fields == ["invoice_number"] for i in issues)
    assert ex.fields["invoice_number"].confidence == "low"
    assert needs_review(ex, issues)


def test_invented_label_is_flagged():
    issues = check(make(invoice_number={"value": "INV-204", "line": "L2", "label": "Bill No."}))
    assert any(i.kind == "label" for i in issues)


def test_missing_required_field():
    ex = make()
    del ex.fields["invoice_number"]
    issues = check(ex)
    assert any(i.kind == "missing" and i.fields == ["invoice_number"] for i in issues)
    assert needs_review(ex, issues)


def test_low_ocr_confidence_lowers_field_confidence():
    lines = [Line(ln.id, ln.page, ln.text, 0.4 if ln.id == "L6" else 1.0) for ln in LINES]
    ex = make()
    issues = normalize(ex) + run_checks(ex, lines, today=TODAY)
    score_confidence(ex, lines, issues)
    assert ex.fields["total_amount"].confidence == "low"
    assert needs_review(ex, issues)


def test_schema_accepts_messy_model_output():
    ex = Extraction.model_validate({
        "invoice_number": "INV-204",                      # fields at the top level, bare value
        "total_amount": {"value": 120, "line": 6},        # line as a number
        "made_up_field": {"value": "x"},                  # unknown field dropped
        "due_date": {"value": None},                      # empty dropped
        "line_items": "none",
    })
    assert set(ex.fields) == {"invoice_number", "total_amount"}
    assert ex.fields["total_amount"].line == "L6"
    assert ex.line_items == []


def test_uk_vat_number():
    assert check_uk_vat_number("GB123456782") is None
    assert check_uk_vat_number("GB 123 4567 82") is None
    assert check_uk_vat_number("123456782") is None
    assert "checksum" in check_uk_vat_number("GB123456789")
    assert "format" in check_uk_vat_number("GB12345")
    assert check_uk_vat_number("DE123456789") is None  # other countries are not checked


def test_iban():
    assert check_iban("GB82 WEST 1234 5698 7654 32") is None
    assert "checksum" in check_iban("GB82 WEST 1234 5698 7654 33")
    assert "format" in check_iban("12345")
