"""Looks up the labels found in an invoice and builds the hint map for the prompt."""
import logging
from dataclasses import dataclass, field

from .label_store import LabelRecord, LabelStore
from .labels import Candidate, find_candidates, normalize_label, text_matches_pattern

log = logging.getLogger(__name__)
MAX_OPTIONS = 2  # how many alternative fields to offer for one ambiguous label


@dataclass
class Hint:
    line_id: str
    line_idx: int
    label: str
    kind: str                    # "field" or "table"
    options: list[LabelRecord]   # best first
    via: str                     # "dictionary" or "vector"

    @property
    def weight(self) -> int:
        return max(r.count for r in self.options)


@dataclass
class HintResult:
    hints: list[Hint] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    used: bool = True

    def fields_at(self, line_id: str | None) -> set[str]:
        return {r.field for h in self.hints if h.line_id == line_id and h.kind == "field" for r in h.options}

    def expected_pattern(self, field_name: str, line_id: str | None) -> str | None:
        for h in self.hints:
            if h.line_id == line_id and h.kind == "field":
                for r in h.options:
                    if r.field == field_name and r.pattern:
                        return r.pattern
        return None

    def stats(self) -> dict:
        return {
            "used": self.used,
            "matched": len(self.hints),
            "from_dictionary": sum(h.via == "dictionary" for h in self.hints),
            "from_vector_search": sum(h.via == "vector" for h in self.hints),
            "unmatched_labels": self.unmatched,
        }

    def render(self, max_hints: int | None = None, line_ids: set[str] | None = None) -> str:
        """The hint map text. `max_hints` keeps the highest-count hints; `line_ids` limits to a page."""
        hints = [h for h in self.hints if line_ids is None or h.line_id in line_ids]
        if max_hints is not None and len(hints) > max_hints:
            keep = sorted(hints, key=lambda h: h.weight, reverse=True)[:max_hints]
            hints = [h for h in hints if any(h is k for k in keep)]
        if not hints:
            return ""

        out = ["LABEL HINTS (suggestions from confirmed invoices):"]
        table: dict[str, list[Hint]] = {}
        for h in hints:
            if h.kind == "table":
                table.setdefault(h.line_id, []).append(h)
        done_tables = set()
        for h in sorted(hints, key=lambda h: h.line_idx):
            if h.kind == "table":
                if h.line_id not in done_tables:
                    done_tables.add(h.line_id)
                    cols = ", ".join(f"{t.label}={t.options[0].field}" for t in table[h.line_id])
                    out.append(f"{h.line_id}  table header → {cols}")
                continue
            out.append(f'{h.line_id}  "{h.label}" → ' + " OR ".join(_describe(r) for r in h.options))

        by_field: dict[str, list[str]] = {}
        for h in hints:
            if h.kind == "field" and len(h.options) == 1:
                by_field.setdefault(h.options[0].field, []).append(h.line_id)
        for name, ids in by_field.items():
            if len(ids) > 1:
                note = f"NOTE: {name} is suggested on {', '.join(ids)}; decide from the surrounding text."
                if name == "total_amount":
                    note += " The later line in the totals section is usually the grand total."
                out.append(note)
        return "\n".join(out)


def _describe(r: LabelRecord) -> str:
    parts = []
    if r.pattern and r.pattern != "TEXT":
        parts.append(f"pattern {r.pattern}")
    parts.append(f"seen {r.count}×" if r.count else "seed")
    return f"{r.field} ({', '.join(parts)})"


def _one_per_field(recs: list[LabelRecord]) -> list[LabelRecord]:
    best: dict[str, LabelRecord] = {}
    for r in recs:
        if r.field not in best or r.count > best[r.field].count:
            best[r.field] = r
    return sorted(best.values(), key=lambda r: r.count, reverse=True)


def resolve(cand: Candidate, recs: list[LabelRecord], header_idx: int | None) -> list[LabelRecord]:
    """Pick the field(s) a matched label most likely means, using its section and value shape."""
    is_table = cand.kind == "table"
    same_kind = [r for r in recs if (r.section == "table") == is_table]
    if not same_kind:
        return []
    if is_table:
        return _one_per_field(same_kind)[:1]

    if header_idx is None:
        preferred = same_kind
    elif cand.line_idx < header_idx:
        preferred = [r for r in same_kind if r.section == "header"]
    else:
        preferred = [r for r in same_kind if r.section in ("totals", "footer")]
    pool = _one_per_field(preferred or same_kind)

    if len(pool) > 1 and cand.value_text:
        shaped = [r for r in pool if r.pattern and r.pattern != "TEXT"
                  and text_matches_pattern(cand.value_text, r.pattern)]
        if shaped and len(shaped) < len(pool):
            pool = shaped
    return pool[:MAX_OPTIONS]


def _table_header_from_words(text: str, store: LabelStore, sources) -> list[str]:
    """Split a header row with no wide gaps ('Description Qty Unit Price Amount') into known column labels."""
    words = text.split()
    found, covered, i = [], 0, 0
    while i < len(words):
        for n in (3, 2, 1):
            chunk = " ".join(words[i : i + n])
            if len(words[i : i + n]) == n and any(
                r.section == "table" for r in store._dict.get(normalize_label(chunk), [])
                if sources is None or r.source in sources
            ):
                found.append(chunk)
                covered += n
                i += n
                break
        else:
            i += 1
    return found if len(found) >= 3 and covered >= 0.7 * len(words) else []


def build_hints(lines: list[tuple[str, str]], store: LabelStore, sources: set[str] | None = None) -> HintResult:
    """lines: [(line_id, text)]. `sources` limits the library, e.g. {'seed'} for evaluation."""
    cands = find_candidates(lines)

    # Lookup layer 1: dictionary.
    looked: list[tuple[Candidate, list[LabelRecord], str | None]] = []
    for c in cands:
        recs, via = store.lookup(c.text, sources)
        looked.append((c, recs, via))

    # The table header is the first row where at least two columns are known table labels.
    header_idx = None
    per_line: dict[int, int] = {}
    for c, recs, _ in looked:
        if c.kind == "table" and any(r.section == "table" for r in recs):
            per_line[c.line_idx] = per_line.get(c.line_idx, 0) + 1
    known_rows = [idx for idx, n in per_line.items() if n >= 2]
    if known_rows:
        header_idx = min(known_rows)
    else:
        for idx, (line_id, text) in enumerate(lines):
            cols = _table_header_from_words(text, store, sources)
            if cols:
                header_idx = idx
                looked = [x for x in looked if x[0].line_idx != idx]
                for col in cols:
                    c = Candidate(line_id, idx, col, "table", "")
                    looked.append((c, *store.lookup(col, sources)))
                break

    # Rows that only look like a table header are ordinary text.
    for c, _, _ in looked:
        if c.kind == "table" and c.line_idx != header_idx:
            c.kind = "bare"

    # Lookup layer 2: one batch vector search for unmatched explicit labels and header columns.
    pending = [i for i, (c, recs, _) in enumerate(looked) if not recs and c.kind in ("explicit", "table")]
    if pending:
        try:
            found = store.vector_lookup([looked[i][0].text for i in pending], sources)
            for i, recs in zip(pending, found):
                if recs:
                    looked[i] = (looked[i][0], recs, "vector")
        except Exception as e:  # hints are optional; extraction continues without the vector layer
            log.warning("vector lookup failed, continuing with dictionary hints only: %s", e)

    result = HintResult()
    joined_lines = set()
    for c, recs, via in looked:
        options = resolve(c, recs, header_idx) if recs else []
        if not options:
            if c.kind in ("explicit", "table") and c.text not in result.unmatched:
                result.unmatched.append(c.text)
            continue
        if c.kind == "joined":
            joined_lines.update((c.line_idx, c.line_idx + 1))
        result.hints.append(Hint(
            c.line_id, c.line_idx, c.text, "table" if c.kind == "table" else "field", options,
            "vector" if via == "vector" else "dictionary",
        ))

    # A joined two-line label replaces hints from its separate halves.
    joined_labels = {h.label for h in result.hints if any(
        c.kind == "joined" and c.text == h.label and c.line_idx == h.line_idx for c, _, _ in looked)}
    result.hints = [
        h for h in result.hints
        if h.label in joined_labels or h.kind == "table" or h.line_idx not in joined_lines
    ]
    result.hints.sort(key=lambda h: h.line_idx)
    return result
