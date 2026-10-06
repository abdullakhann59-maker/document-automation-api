"""Label normalization, value patterns, and candidate label detection."""
import re
from dataclasses import dataclass
from datetime import date, datetime

_DATE_FORMATS = [
    ("%d/%m/%Y", "DD/MM/YYYY"),
    ("%d-%m-%Y", "DD-MM-YYYY"),
    ("%d.%m.%Y", "DD.MM.YYYY"),
    ("%d/%m/%y", "DD/MM/YY"),
    ("%d-%m-%y", "DD-MM-YY"),
    ("%d.%m.%y", "DD.MM.YY"),
    ("%Y-%m-%d", "YYYY-MM-DD"),
    ("%d %B %Y", "DD MONTH YYYY"),
    ("%d %b %Y", "DD MON YYYY"),
    ("%d-%b-%Y", "DD-MON-YYYY"),
    ("%d-%b-%y", "DD-MON-YY"),
    ("%B %d %Y", "MONTH DD YYYY"),
    ("%b %d %Y", "MON DD YYYY"),
]
DATE_PATTERNS = {p for _, p in _DATE_FORMATS} | {"DATE"}

_DATE_TOKEN = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?[ \-][A-Za-z]{3,9}\.?,?[ \-]\d{2,4}\b"
    r"|\b[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b"
)
_NUMBER_TOKEN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_label(text: str) -> str:
    """'Bill No. :' -> 'bill no', 'P.O. Number' -> 'po number', 'Inv #' -> 'inv #'."""
    s = text.lower().replace(".", "")
    s = re.sub(r"[^a-z0-9#/%&]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("/&").strip()


def parse_date(text: str) -> tuple[date, str] | None:
    """Parse a day-first date; returns (date, pattern name) or None."""
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", str(text).strip(), flags=re.I)
    s = re.sub(r"\s+", " ", s.replace(",", " ").replace(".", ". ").strip())
    s = s.replace(". ", ".") if re.fullmatch(r"[\d. ]+", s) else s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    for fmt, name in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date(), name
        except ValueError:
            continue
    return None


def find_dates(text: str) -> list[tuple[date, str]]:
    out = []
    for m in _DATE_TOKEN.finditer(text):
        parsed = parse_date(m.group())
        if parsed:
            out.append(parsed)
    return out


def to_number(value) -> float | None:
    """'£1,234.50' -> 1234.5, '20%' -> 20.0, '(12.00)' -> -12.0. None if not a number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"(?i)gbp|eur|usd|[£$€%\s,()]", "", s)
    if s.endswith("-"):
        negative, s = True, s[:-1]
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    n = float(s)
    return -abs(n) if negative else n


def find_numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER_TOKEN.finditer(text):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return out


def value_pattern(value, ftype: str = "str") -> str | None:
    """Convert a value to a shape that is safe to store: 'INV-204' -> 'INV-###'."""
    if value is None:
        return None
    if ftype == "number":
        return "NUMBER"
    s = str(value).strip()
    if not s:
        return None
    if ftype == "date":
        parsed = parse_date(s)
        return parsed[1] if parsed else "DATE"
    if not re.search(r"\d", s) or len(s) > 25:
        return "TEXT"
    return re.sub(r"\d", "#", s.upper())


def text_matches_pattern(text: str, pattern: str | None) -> bool:
    """Does `text` contain something shaped like `pattern`? Unknown patterns match."""
    if not pattern or pattern == "TEXT":
        return True
    if pattern in DATE_PATTERNS:
        return bool(find_dates(text))
    if pattern == "NUMBER":
        return bool(find_numbers(text))
    rx = re.sub(r"(\\#)+", r"\\d+", re.escape(pattern))
    rx = rx.replace(r"\ ", r"\s*")
    return re.search(rx, text, flags=re.I) is not None


@dataclass
class Candidate:
    line_id: str
    line_idx: int
    text: str        # the label as written
    kind: str        # explicit (before a colon) | prefix (before a number) | bare | joined | table
    value_text: str  # the text the label seems to describe


def split_segments(text: str) -> list[str]:
    return [s for s in re.split(r"\s{3,}", text.strip()) if s]


def _is_short_text(seg: str, max_words: int = 5, max_len: int = 40) -> bool:
    core = re.sub(r"\d+(\.\d+)?\s*%", "", seg)
    return (
        len(seg) <= max_len
        and len(seg.split()) <= max_words
        and re.search(r"[A-Za-z]", core) is not None
        and not re.search(r"\d", core)
        and "@" not in core.replace(" @ ", " ")
    )


_COLON = re.compile(r"^([^:]{2,40}?)\s*:\s*(.*)$")
# Label, optionally ending in a rate ("VAT @ 20%"), then the first number.
_PREFIX = re.compile(
    r"^([A-Za-z][A-Za-z .#/&'()@\-]{1,38}?(?:\s*\d+(?:\.\d+)?\s*%)?)[\s#:]+([£$€(\-]?\d.*)$"
)


def find_candidates(lines: list[tuple[str, str]]) -> list[Candidate]:
    """lines: [(line_id, text)]. Returns every piece of text that could be a field label."""
    out: list[Candidate] = []
    seg_lists = [split_segments(text) for _, text in lines]

    for idx, (line_id, text) in enumerate(lines):
        segs = seg_lists[idx]
        next_text = lines[idx + 1][1] if idx + 1 < len(lines) else ""

        if len(segs) >= 3 and all(_is_short_text(s, max_words=3, max_len=25) for s in segs):
            out.extend(Candidate(line_id, idx, s.rstrip(":").strip(), "table", "") for s in segs)
            continue

        for j, seg in enumerate(segs):
            following = segs[j + 1] if j + 1 < len(segs) else next_text
            m = _COLON.match(seg)
            if m:
                label, value = m.group(1).strip(), m.group(2).strip()
                if (
                    re.search(r"[A-Za-z]", label)
                    and len(label.split()) <= 5
                    and not value.startswith("//")
                ):
                    out.append(Candidate(line_id, idx, label, "explicit", value or following))
                    continue
            m = _PREFIX.match(seg)
            if m and len(m.group(1).split()) <= 4:
                out.append(Candidate(line_id, idx, m.group(1).strip(), "prefix", m.group(2).strip()))
                continue
            if _is_short_text(seg):
                out.append(Candidate(line_id, idx, seg.strip(), "bare", following))

        # A label split over two lines by OCR: try the two short lines joined.
        if (
            len(segs) == 1
            and _is_short_text(segs[0], max_words=3)
            and idx + 1 < len(lines)
            and len(seg_lists[idx + 1]) == 1
            and _is_short_text(seg_lists[idx + 1][0], max_words=3)
        ):
            after = lines[idx + 2][1] if idx + 2 < len(lines) else ""
            out.append(Candidate(line_id, idx, f"{segs[0]} {seg_lists[idx + 1][0]}", "joined", after))
    return out
