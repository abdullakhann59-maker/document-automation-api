from app.config import Settings
from app.hints import build_hints
from app.label_store import LabelStore
from app.storage import Storage

INVOICE = [
    ("L1", "Invoice No: INV-204   Date: 12/03/2026"),
    ("L2", "Description   Qty   Unit Price   Total"),
    ("L3", "Widget   2   50.00   100.00"),
    ("L4", "VAT   20.00"),
    ("L5", "Total   120.00"),
    ("L6", "Ref Code: X1"),
]


def test_exact_lookup_ignores_punctuation(store):
    recs, via = store.lookup("Invoice No. :")
    assert via == "exact"
    assert [r.field for r in recs] == ["invoice_number"]


def test_fuzzy_lookup_catches_ocr_typo(store):
    recs, via = store.lookup("lnvoice No")
    assert via == "fuzzy"
    assert recs[0].field == "invoice_number"


def test_unknown_label(store):
    assert store.lookup("Ref Code") == ([], None)


def test_confirm_increments_and_adds(store):
    items = [
        {"label": "Invoice No", "field": "invoice_number", "section": "header", "pattern": "INV-###"},
        {"label": "Bill No.", "field": "invoice_number", "section": "header", "pattern": "INV-###"},
    ]
    assert store.add_labels(items, source="confirmed", vendor="GB123456782") == (1, 1)
    existing = store.lookup("Invoice No")[0][0]
    assert (existing.count, existing.pattern, existing.vendors) == (1, "INV-###", ["GB123456782"])
    assert store.lookup("bill no")[0][0].count == 1


def test_library_survives_reload(store):
    store.add_labels([{"label": "Bill No.", "field": "invoice_number", "section": "header", "pattern": None}],
                     source="confirmed")
    store.load()
    assert store.lookup("Bill No")[0][0].field == "invoice_number"


def test_min_label_count_filters_confirmed_but_not_seed(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, min_label_count=2)
    s = LabelStore(Storage(tmp_path / "t.db"), settings)
    s.add_labels([{"label": "Total", "field": "total_amount", "section": "totals", "pattern": None}],
                 source="seed", increment=False)
    s.add_labels([{"label": "Bill No", "field": "invoice_number", "section": "header", "pattern": None}],
                 source="confirmed")
    assert s.lookup("Total")[0]
    assert s.lookup("Bill No") == ([], None)


def test_delete(store):
    rec = store.lookup("Invoice No")[0][0]
    assert store.delete(rec.id)
    assert store.lookup("Invoice No") == ([], None)
    assert not store.delete(rec.id)


def test_hints_resolve_same_label_by_section(store):
    hints = build_hints(INVOICE, store)
    assert hints.fields_at("L5") == {"total_amount"}       # "Total" after the table is the grand total
    assert hints.fields_at("L4") == {"vat_amount"}
    table = {h.label: h.options[0].field for h in hints.hints if h.kind == "table"}
    assert table == {"Description": "description", "Qty": "quantity",
                     "Unit Price": "unit_price", "Total": "line_total"}
    assert "invoice_number" in hints.fields_at("L1")


def test_hints_pass_both_options_when_unclear(store):
    hints = build_hints(INVOICE, store)
    date_hint = next(h for h in hints.hints if h.label == "Date")
    assert {r.field for r in date_hint.options} == {"invoice_date", "tax_point_date"}


def test_hints_report_unmatched_and_stats(store):
    hints = build_hints(INVOICE, store)
    assert hints.unmatched == ["Ref Code"]
    stats = hints.stats()
    assert stats["matched"] == 8 and stats["from_dictionary"] == 8 and stats["from_vector_search"] == 0


def test_hint_map_text(store):
    text = build_hints(INVOICE, store).render()
    assert text.startswith("LABEL HINTS")
    assert 'L1  "Invoice No" → invoice_number (seed)' in text
    assert "L2  table header → Description=description, Qty=quantity, Unit Price=unit_price, Total=line_total" in text


def test_table_header_without_wide_gaps(store):
    hints = build_hints([("L1", "Description Qty Unit Price Total"), ("L2", "Total 120.00")], store)
    assert [h.label for h in hints.hints if h.kind == "table"] == ["Description", "Qty", "Unit Price", "Total"]
    assert hints.fields_at("L2") == {"total_amount"}


def test_sources_filter_for_evaluation(store):
    store.add_labels([{"label": "Ref Code", "field": "po_number", "section": "header", "pattern": None}],
                     source="confirmed")
    assert build_hints(INVOICE, store).fields_at("L6") == {"po_number"}
    assert build_hints(INVOICE, store, sources={"seed"}).fields_at("L6") == set()
