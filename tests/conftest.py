import pytest

from app.config import Settings
from app.label_store import LabelStore
from app.storage import Storage

SEED = [
    ("Invoice No", "invoice_number", "header", None),
    ("Date", "invoice_date", "header", "DATE"),
    ("Date", "tax_point_date", "header", "DATE"),
    ("VAT", "vat_amount", "totals", "NUMBER"),
    ("Total", "total_amount", "totals", "NUMBER"),
    ("Description", "description", "table", None),
    ("Qty", "quantity", "table", "NUMBER"),
    ("Unit Price", "unit_price", "table", "NUMBER"),
    ("Total", "line_total", "table", "NUMBER"),
]


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def store(tmp_path, settings):
    """A label store with a few seed labels and no vector layer."""
    s = LabelStore(Storage(tmp_path / "test.db"), settings)
    s.add_labels(
        [{"label": label, "field": f, "section": sec, "pattern": p} for label, f, sec, p in SEED],
        source="seed", increment=False,
    )
    return s
