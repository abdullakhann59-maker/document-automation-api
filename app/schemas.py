import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# name -> (type, description). Types: "str", "number", "date".
FIELDS: dict[str, tuple[str, str]] = {
    "vendor_name": ("str", "business that issued the invoice (seller)"),
    "vendor_address": ("str", "seller's postal address"),
    "vendor_vat_number": ("str", "seller's VAT registration number, e.g. GB123456789"),
    "vendor_company_number": ("str", "seller's Companies House registration number"),
    "vendor_email": ("str", "seller's email address"),
    "vendor_phone": ("str", "seller's phone number"),
    "buyer_name": ("str", "customer being invoiced"),
    "buyer_address": ("str", "customer's billing address"),
    "buyer_vat_number": ("str", "customer's VAT number"),
    "delivery_address": ("str", "ship-to / delivery address if different"),
    "invoice_number": ("str", "the invoice's unique number"),
    "invoice_date": ("date", "date the invoice was issued"),
    "tax_point_date": ("date", "tax point / date of supply, if shown separately"),
    "due_date": ("date", "date payment is due"),
    "po_number": ("str", "customer's purchase order number or reference"),
    "payment_terms": ("str", "payment terms, e.g. 30 days"),
    "currency": ("str", "ISO currency code, e.g. GBP"),
    "subtotal": ("number", "net total before VAT"),
    "discount": ("number", "total discount amount"),
    "shipping": ("number", "delivery / carriage charge"),
    "vat_rate": ("number", "VAT rate as a percentage (20, 5 or 0) when one rate applies"),
    "vat_amount": ("number", "total VAT charged"),
    "total_amount": ("number", "gross total including VAT"),
    "amount_paid": ("number", "amount already paid"),
    "balance_due": ("number", "amount still to pay"),
    "bank_name": ("str", "seller's bank"),
    "bank_account_name": ("str", "name on the seller's bank account"),
    "bank_sort_code": ("str", "seller's sort code, e.g. 12-34-56"),
    "bank_account_number": ("str", "seller's bank account number"),
    "iban": ("str", "seller's IBAN"),
    "swift_bic": ("str", "seller's SWIFT/BIC code"),
}

LINE_ITEM_COLUMNS: dict[str, tuple[str, str]] = {
    "description": ("str", "what was sold"),
    "product_code": ("str", "item / product code"),
    "quantity": ("number", "quantity"),
    "unit_price": ("number", "price per unit, net"),
    "vat_rate": ("number", "VAT rate percentage for the line"),
    "vat_amount": ("number", "VAT amount for the line"),
    "line_total": ("number", "line amount"),
}

REQUIRED_FIELDS = ("invoice_number", "invoice_date", "vendor_name", "total_amount")
SECTIONS = ("header", "table", "totals", "footer")
DOCUMENT_TYPES = ("invoice", "receipt", "purchase_order", "credit_note", "bank_statement", "other")


def _coerce_line(v: Any) -> str | None:
    if v is None:
        return None
    m = re.search(r"\d+", str(v))
    return f"L{int(m.group())}" if m else None


def _coerce_scalar(v: Any) -> Any:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


class FieldValue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: str | float | int | None = None
    label: str | None = None
    line: str | None = None
    confidence: str | None = None

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, v):
        return _coerce_line(v)

    @field_validator("value", mode="before")
    @classmethod
    def _scalar(cls, v):
        return _coerce_scalar(v)

    @field_validator("label", "confidence", mode="before")
    @classmethod
    def _text(cls, v):
        v = _coerce_scalar(v)
        return None if v is None or str(v).lower() in ("null", "none") else str(v)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line: str | None = None
    description: str | None = None
    product_code: str | None = None
    quantity: float | int | str | None = None
    unit_price: float | int | str | None = None
    vat_rate: float | int | str | None = None
    vat_amount: float | int | str | None = None
    line_total: float | int | str | None = None

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, v):
        return _coerce_line(v)

    @field_validator("description", "product_code", mode="before")
    @classmethod
    def _text(cls, v):
        v = _coerce_scalar(v)
        return None if v is None else str(v)

    @field_validator("quantity", "unit_price", "vat_rate", "vat_amount", "line_total", mode="before")
    @classmethod
    def _num(cls, v):
        return _coerce_scalar(v)


class TableHeaders(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line: str | None = None
    columns: dict[str, str] = {}

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, v):
        return _coerce_line(v)

    @field_validator("columns", mode="before")
    @classmethod
    def _columns(cls, v):
        if not isinstance(v, dict):
            return {}
        return {str(k): str(c) for k, c in v.items() if c in LINE_ITEM_COLUMNS}


class Extraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fields: dict[str, FieldValue] = {}
    table_headers: TableHeaders | None = None
    line_items: list[LineItem] = []

    @model_validator(mode="before")
    @classmethod
    def _shape(cls, data):
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        data = dict(data)
        raw = data.get("fields")
        if not isinstance(raw, dict):
            # Small models sometimes put the fields at the top level.
            raw = {k: v for k, v in data.items() if k in FIELDS}
        fields = {}
        for name, v in raw.items():
            if name not in FIELDS or v is None:
                continue
            if not isinstance(v, dict):
                v = {"value": v}
            if v.get("value") in (None, "", "null"):
                continue
            fields[name] = v
        data["fields"] = fields
        if not isinstance(data.get("line_items"), list):
            data["line_items"] = []
        data["line_items"] = [i for i in data["line_items"] if isinstance(i, dict)]
        if not isinstance(data.get("table_headers"), dict):
            data["table_headers"] = None
        return data


class ConfirmRequest(BaseModel):
    doc_id: str
    approved: bool = False
    corrected_json: dict | None = None
    overwrite: bool = False


class RejectRequest(BaseModel):
    doc_id: str
