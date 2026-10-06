"""Value normalization, business-rule checks, and confidence scoring. Nothing here fixes values."""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .hints import HintResult
from .labels import DATE_PATTERNS, find_dates, find_numbers, parse_date, text_matches_pattern, to_number
from .ocr import Line
from .schemas import FIELDS, LINE_ITEM_COLUMNS, REQUIRED_FIELDS, Extraction

TOL = 0.02
UK_VAT_RATES = (0.0, 5.0, 20.0)
UNTRACED_FIELDS = {"currency"}  # often inferred from a £ sign rather than printed


@dataclass
class Issue:
    kind: str       # math | format | date | missing | trace | pattern | label | value
    message: str
    fields: list[str] = field(default_factory=list)


def _squash(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# --- normalization -------------------------------------------------------------

def normalize(ex: Extraction) -> list[Issue]:
    """Put numbers and dates into standard form. Values that can't be read are kept and flagged."""
    warnings = []
    for name, fv in ex.fields.items():
        ftype = FIELDS[name][0]
        if ftype == "number":
            n = to_number(fv.value)
            if n is None:
                warnings.append(Issue("value", f"{name}: '{fv.value}' is not a number", [name]))
            else:
                fv.value = n
        elif ftype == "date":
            parsed = parse_date(str(fv.value))
            if parsed is None:
                warnings.append(Issue("value", f"{name}: '{fv.value}' is not a readable date", [name]))
            else:
                fv.value = parsed[0].isoformat()
        else:
            fv.value = str(fv.value).strip()
    for i, item in enumerate(ex.line_items, start=1):
        for col, (ctype, _) in LINE_ITEM_COLUMNS.items():
            raw = getattr(item, col)
            if ctype != "number" or raw is None:
                continue
            n = to_number(raw)
            if n is None:
                warnings.append(Issue("value", f"line item {i}: {col} '{raw}' is not a number", ["line_items"]))
            else:
                setattr(item, col, n)
    return warnings


# --- format checks -------------------------------------------------------------

def check_uk_vat_number(value: str) -> str | None:
    """Returns a problem description, or None if the number is acceptable."""
    v = re.sub(r"[\s.\-]", "", value.upper())
    if re.fullmatch(r"(GB)?(GD|HA)\d{3}", v):
        return None
    m = re.fullmatch(r"(GB|XI)?(\d{9}|\d{12})", v)
    if not m:
        if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{2,13}", v) and not v.startswith(("GB", "XI")):
            return None  # another country's VAT number; format not checked
        return "is not in the UK format (GB + 9 or 12 digits)"
    digits = m.group(2)[:9]
    total = sum(int(d) * w for d, w in zip(digits[:7], range(8, 1, -1)))
    check = int(digits[7:])
    if (total + check) % 97 == 0 or (total + 55 + check) % 97 == 0:
        return None
    return "fails the VAT checksum (possible misread digit)"


def check_iban(value: str) -> str | None:
    v = re.sub(r"\s", "", value.upper())
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", v):
        return "is not in IBAN format"
    rearranged = v[4:] + v[:4]
    number = int("".join(str(int(ch, 36)) for ch in rearranged))
    return None if number % 97 == 1 else "fails the IBAN checksum (possible misread character)"


def _format_checks(ex: Extraction) -> list[Issue]:
    out = []

    def text(name: str) -> str | None:
        fv = ex.fields.get(name)
        return str(fv.value) if fv and fv.value is not None else None

    for name in ("vendor_vat_number", "buyer_vat_number"):
        if text(name) and (problem := check_uk_vat_number(text(name))):
            out.append(Issue("format", f"{name} '{text(name)}' {problem}", [name]))
    if text("iban") and (problem := check_iban(text("iban"))):
        out.append(Issue("format", f"iban {problem}", ["iban"]))
    if text("bank_sort_code") and len(re.sub(r"\D", "", text("bank_sort_code"))) != 6:
        out.append(Issue("format", "bank_sort_code should have 6 digits", ["bank_sort_code"]))
    if text("bank_account_number") and len(re.sub(r"\D", "", text("bank_account_number"))) != 8:
        out.append(Issue("format", "bank_account_number should have 8 digits", ["bank_account_number"]))
    if text("vendor_company_number") and not re.fullmatch(
        r"\d{8}|[A-Z]{2}\d{6}", re.sub(r"\s", "", text("vendor_company_number").upper())
    ):
        out.append(Issue("format", "vendor_company_number should be 8 characters (e.g. 01234567 or SC123456)",
                           ["vendor_company_number"]))
    return out


# --- math and date checks --------------------------------------------------------

def _math_checks(ex: Extraction) -> list[Issue]:
    out = []

    def num(name: str) -> float | None:
        fv = ex.fields.get(name)
        return fv.value if fv and isinstance(fv.value, (int, float)) else None

    sub, vat, total = num("subtotal"), num("vat_amount"), num("total_amount")
    disc, ship, rate = num("discount") or 0.0, num("shipping") or 0.0, num("vat_rate")
    disc = abs(disc)

    line_totals = []
    for i, item in enumerate(ex.line_items, start=1):
        q, p, lt = item.quantity, item.unit_price, item.line_total
        if isinstance(lt, (int, float)):
            line_totals.append(lt)
        if all(isinstance(x, (int, float)) for x in (q, p, lt)):
            item_rate = item.vat_rate if isinstance(item.vat_rate, (int, float)) else None
            gross = q * p * (1 + item_rate / 100) if item_rate is not None else None
            if not _close(q * p, lt) and not (gross is not None and _close(gross, lt)):
                out.append(Issue("math", f"line item {i}: {q:g} × {p:.2f} does not equal {lt:.2f}", ["line_items"]))

    if sub is not None and line_totals and len(line_totals) == len(ex.line_items):
        s = sum(line_totals)
        tol = TOL + 0.01 * len(line_totals)
        targets = [sub, sub + disc] + ([total] if total is not None else [])
        if not any(_close(s, t, tol) for t in targets):
            out.append(Issue("math", f"line totals add up to {s:.2f}, but subtotal is {sub:.2f}",
                               ["subtotal", "line_items"]))

    if sub is not None and vat is not None and total is not None:
        if not any(_close(sub + vat + ship - d, total) for d in (disc, 0.0)):
            out.append(Issue(
                "math",
                f"subtotal {sub:.2f} + VAT {vat:.2f} − discount {disc:.2f} + shipping {ship:.2f} "
                f"does not equal total {total:.2f}",
                ["subtotal", "vat_amount", "total_amount"],
            ))

    if rate is not None:
        if rate not in UK_VAT_RATES:
            out.append(Issue("format", f"vat_rate {rate:g}% is not a UK VAT rate (20, 5 or 0)", ["vat_rate"]))
        elif sub is not None and vat is not None:
            bases = (sub, sub - disc, sub + ship, sub - disc + ship)
            if not any(_close(b * rate / 100, vat, 0.05) for b in bases):
                out.append(Issue("math", f"VAT amount {vat:.2f} does not equal subtotal {sub:.2f} × {rate:g}%",
                                   ["vat_amount", "vat_rate", "subtotal"]))

    paid, balance = num("amount_paid"), num("balance_due")
    if total is not None and paid is not None and balance is not None and not _close(total - paid, balance):
        out.append(Issue("math", f"total {total:.2f} − paid {paid:.2f} does not equal balance due {balance:.2f}",
                           ["total_amount", "amount_paid", "balance_due"]))
    return out


def _date_checks(ex: Extraction, today: date) -> list[Issue]:
    out = []
    dates: dict[str, date] = {}
    for name in ("invoice_date", "tax_point_date", "due_date"):
        fv = ex.fields.get(name)
        if fv and isinstance(fv.value, str):
            try:
                dates[name] = date.fromisoformat(fv.value)
            except ValueError:
                pass
    for name, d in dates.items():
        if d.year < 2000 or d > today + timedelta(days=730):
            out.append(Issue("date", f"{name} {d.isoformat()} is not a realistic date", [name]))
    inv = dates.get("invoice_date")
    if inv and inv > today + timedelta(days=1):
        out.append(Issue("date", f"invoice_date {inv.isoformat()} is in the future "
                                   "(day and month may be swapped)", ["invoice_date"]))
    if inv and "due_date" in dates and dates["due_date"] < inv:
        out.append(Issue("date", "due_date is before invoice_date", ["due_date", "invoice_date"]))
    return out


# --- tracing values and labels back to the text ------------------------------------

def _window(lines: list[Line], index: dict[str, int], line_id: str | None, before: int, after: int) -> str | None:
    if line_id not in index:
        return None
    i = index[line_id]
    return "\n".join(ln.text for ln in lines[max(0, i - before) : i + after + 1])


def value_in_text(value, ftype: str, text: str) -> bool:
    if ftype == "number" and isinstance(value, (int, float)):
        return any(abs(abs(n) - abs(value)) < 0.005 for n in find_numbers(text))
    if ftype == "date":
        return any(d.isoformat() == value for d, _ in find_dates(text))
    squashed = _squash(text)
    if _squash(value) in squashed:
        return True
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    return bool(tokens) and sum(t in squashed for t in tokens) / len(tokens) >= 0.6


def _trace_checks(ex: Extraction, lines: list[Line], check_labels: bool) -> list[Issue]:
    out = []
    index = {ln.id: i for i, ln in enumerate(lines)}
    for name, fv in ex.fields.items():
        ftype = FIELDS[name][0]
        if name not in UNTRACED_FIELDS:
            # Text values (addresses) may run over the following lines.
            window = _window(lines, index, fv.line, 1, 4 if ftype == "str" else 1)
            if window is None:
                out.append(Issue("trace", f"{name}: no valid source line was given", [name]))
            elif not value_in_text(fv.value, ftype, window):
                out.append(Issue("trace", f"{name}: value '{fv.value}' was not found at {fv.line}", [name]))
        if check_labels and fv.label:
            window = _window(lines, index, fv.line, 1, 1)
            if window is None or _squash(fv.label) not in _squash(window):
                out.append(Issue("label", f"{name}: label '{fv.label}' was not found at {fv.line}", [name]))
    if check_labels and ex.table_headers:
        window = _window(lines, index, ex.table_headers.line, 1, 1) or ""
        for col in ex.table_headers.columns:
            if _squash(col) not in _squash(window):
                out.append(Issue("label", f"table header '{col}' was not found at {ex.table_headers.line}",
                                   ["line_items"]))
    return out


def label_in_text(label: str, line_id: str | None, lines: list[Line]) -> bool:
    index = {ln.id: i for i, ln in enumerate(lines)}
    window = _window(lines, index, line_id, 1, 1)
    if window is None:  # no line given: accept the label if it appears anywhere
        window = "\n".join(ln.text for ln in lines)
    return _squash(label) in _squash(window)


def _pattern_checks(ex: Extraction, lines: list[Line], hints: HintResult | None) -> list[Issue]:
    if hints is None:
        return []
    out = []
    index = {ln.id: i for i, ln in enumerate(lines)}
    for name, fv in ex.fields.items():
        pattern = hints.expected_pattern(name, fv.line)
        if not pattern or pattern in ("TEXT", "NUMBER", "DATE"):
            continue
        # Dates were converted to ISO, so compare the pattern with the source text instead.
        if pattern in DATE_PATTERNS:
            subject = _window(lines, index, fv.line, 0, 1) or ""
        else:
            subject = str(fv.value)
        if not text_matches_pattern(subject, pattern):
            out.append(Issue("pattern", f"{name}: '{fv.value}' does not look like the usual pattern {pattern}",
                               [name]))
    return out


# --- entry points ------------------------------------------------------------------

def run_checks(ex: Extraction, lines: list[Line], hints: HintResult | None = None,
               check_labels: bool = False, today: date | None = None) -> list[Issue]:
    out = _math_checks(ex) + _format_checks(ex) + _date_checks(ex, today or date.today())
    out += _pattern_checks(ex, lines, hints) + _trace_checks(ex, lines, check_labels)
    for name in REQUIRED_FIELDS:
        if name not in ex.fields:
            out.append(Issue("missing", f"required field {name} was not found", [name]))
    return out


def score_confidence(ex: Extraction, lines: list[Line], warnings: list[Issue],
                     hints: HintResult | None = None) -> None:
    """Sets high / medium / low on each field from OCR confidence, hint agreement and failed checks."""
    conf_by_id = {ln.id: ln.conf for ln in lines}
    flagged: dict[str, set[str]] = {}
    for w in warnings:
        for name in w.fields:
            flagged.setdefault(name, set()).add(w.kind)

    for name, fv in ex.fields.items():
        ocr_conf = conf_by_id.get(fv.line, 0.5)
        level = 2 if ocr_conf >= 0.85 else 1 if ocr_conf >= 0.6 else 0
        if hints is not None:
            suggested = hints.fields_at(fv.line)
            if name in suggested:
                level += 1
            elif suggested:
                level -= 1
        kinds = flagged.get(name, set())
        if kinds & {"trace", "value"}:
            level = 0
        elif kinds:
            level = min(level, 1)
        fv.confidence = "high" if level >= 2 else "medium" if level == 1 else "low"


def needs_review(ex: Extraction, warnings: list[Issue]) -> bool:
    if any(w.kind in ("math", "missing") for w in warnings):
        return True
    return any(ex.fields[name].confidence == "low" for name in REQUIRED_FIELDS if name in ex.fields)
