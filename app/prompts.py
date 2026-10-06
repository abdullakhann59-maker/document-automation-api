"""Versioned prompt templates. Bump PROMPT_VERSION in .env when anything here changes."""
from .schemas import DOCUMENT_TYPES, FIELDS, LINE_ITEM_COLUMNS

_EXTRACT_V1 = """You extract data from UK VAT invoices. The invoice text is given below with line numbers (L1, L2, ...).

FIELDS:
<<FIELDS>>

LINE ITEM COLUMNS:
<<COLUMNS>>

RULES:
- Answer with one JSON object only. No explanations.
- Include a field only if it is on the invoice. Leave out fields that are missing. Never invent or guess a value.
- Copy text values exactly as written.
- Dates: convert to YYYY-MM-DD. UK invoices write the day first (03/04/2026 is 3 April 2026).
- Numbers: digits and a decimal point only. No currency symbols, commas, or % signs.
- "line" is the line number where the value appears.
<<LABEL_RULE>>
- vendor_name is the business issuing the invoice, usually the first business name at the top. It has no label.
- buyer_name and buyer_address follow labels like "Bill To", "Invoice To" or "Customer".
- The seller's VAT number and bank details usually sit near the seller's name or in the footer.
- currency: GBP if amounts use the pound sign and nothing else is stated.
- subtotal is before VAT; total_amount includes VAT.
- One line_items entry per product or service row. Totals rows are not line items.

OUTPUT FORMAT:
<<FORMAT>>
<<HINTS>>
INVOICE TEXT:
<<TEXT>>

JSON:"""

_FORMAT_WITH_LABELS = """{"fields": {"<field name>": {"value": <value>, "label": "<label as written, or null>", "line": "L<n>"}},
 "table_headers": {"line": "L<n>", "columns": {"<column header as written>": "<line item column>"}},
 "line_items": [{"line": "L<n>", "description": "...", "quantity": 0, "unit_price": 0, "vat_rate": 0, "line_total": 0}]}"""

_FORMAT_PLAIN = """{"fields": {"<field name>": {"value": <value>, "line": "L<n>"}},
 "line_items": [{"line": "L<n>", "description": "...", "quantity": 0, "unit_price": 0, "vat_rate": 0, "line_total": 0}]}"""

_LABEL_RULE = (
    '- "label" is the label text printed next to or above the value, copied exactly '
    '(for example "Bill No." or "Total Due"). Use null if the value has no label.\n'
    '- "table_headers" maps each column header of the items table, as written, to a line item column.'
)

_HINTS_BLOCK = """
<<HINT_MAP>>
Hints are suggestions from past invoices. If the invoice text clearly contradicts a hint, follow the text.
"""

_CLASSIFY_V1 = """Classify this document. Answer with one JSON object: {"document_type": "<type>"}
Allowed types: <<TYPES>>
- invoice: a request for payment for goods or services
- credit_note: reduces or cancels an invoice
- receipt: confirms a payment already made
- purchase_order: an order sent by a buyer
- bank_statement: a list of account transactions

DOCUMENT (first lines):
<<TEXT>>

JSON:"""

PROMPTS = {"v1": {"extract": _EXTRACT_V1, "classify": _CLASSIFY_V1}}


def _field_list(defs: dict) -> str:
    return "\n".join(f"- {name} ({ftype}): {desc}" for name, (ftype, desc) in defs.items())


def _template(version: str, name: str) -> str:
    if version not in PROMPTS:
        raise ValueError(f"Unknown PROMPT_VERSION '{version}'; known: {', '.join(PROMPTS)}")
    return PROMPTS[version][name]


def build_extract_prompt(version: str, text: str, hint_map: str = "", capture_labels: bool = False) -> str:
    hints = _HINTS_BLOCK.replace("<<HINT_MAP>>", hint_map) if hint_map else ""
    return (
        _template(version, "extract")
        .replace("<<FIELDS>>", _field_list(FIELDS))
        .replace("<<COLUMNS>>", _field_list(LINE_ITEM_COLUMNS))
        .replace("<<LABEL_RULE>>", _LABEL_RULE if capture_labels else "")
        .replace("<<FORMAT>>", _FORMAT_WITH_LABELS if capture_labels else _FORMAT_PLAIN)
        .replace("<<HINTS>>", hints)
        .replace("<<TEXT>>", text)
    )


def build_classify_prompt(version: str, text: str) -> str:
    return (
        _template(version, "classify")
        .replace("<<TYPES>>", ", ".join(DOCUMENT_TYPES))
        .replace("<<TEXT>>", text)
    )
