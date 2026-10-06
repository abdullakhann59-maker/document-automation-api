"""The shared S1-S5 pipeline and the ingest / confirm / extract flows."""
import logging
import time
import uuid

from pydantic import ValidationError

from . import vendors
from .classifier import classify
from .config import Settings
from .errors import AppError
from .hints import HintResult, build_hints
from .label_store import LabelStore
from .labels import find_dates, value_pattern
from .llm import OllamaClient
from .logging_setup import request_id_var
from .ocr import DocText, Line, OcrEngine, parse_rendered, read_document, render_lines
from .prompts import build_extract_prompt
from .schemas import FIELDS, ConfirmRequest, Extraction
from .storage import Storage, save_output
from .validation import Issue, label_in_text, needs_review, normalize, run_checks, score_confidence

log = logging.getLogger(__name__)

TOTALS_FIELDS = {"subtotal", "discount", "shipping", "vat_rate", "vat_amount", "total_amount",
                 "amount_paid", "balance_due"}
BANK_FIELDS = {"bank_name", "bank_account_name", "bank_sort_code", "bank_account_number", "iban", "swift_bic"}
CHARS_PER_TOKEN = 3       # conservative for invoice text (numbers tokenize poorly)
TRIMMED_HINTS = 15


class Pipeline:
    def __init__(self, settings: Settings, db: Storage, llm: OllamaClient, store: LabelStore, ocr: OcrEngine):
        self.settings = settings
        self.db = db
        self.llm = llm
        self.store = store
        self.ocr = ocr

    # --- shared steps -----------------------------------------------------------
    def read_invoice(self, data: bytes) -> DocText:
        """S1-S5. Raises AppError for bad files, unreadable documents and non-invoices."""
        t = time.perf_counter()
        doc = read_document(data, self.settings, self.ocr)
        log.info("text ready: %d pages, %d lines, ocr=%s (%.1fs)", doc.pages, len(doc.lines), doc.ocr_used,
                 time.perf_counter() - t)
        t = time.perf_counter()
        doc_type = classify(doc, self.llm, self.settings.prompt_version)
        log.info("classified as %s (%.1fs)", doc_type, time.perf_counter() - t)
        if doc_type != "invoice":
            raise AppError(422, f"Not an invoice (detected: {doc_type})", document_type=doc_type)
        return doc

    def _llm_extract(self, lines: list[Line], hint_map: str, capture_labels: bool) -> Extraction:
        prompt = build_extract_prompt(self.settings.prompt_version, render_lines(lines), hint_map, capture_labels)
        return self.llm.generate_json(prompt, Extraction.model_validate)

    def llm_extract(self, doc: DocText, hints: HintResult | None, capture_labels: bool) -> Extraction:
        """One LLM call if the prompt fits the context; otherwise trim hints, then go page by page."""
        s = self.settings
        budget = (s.llm_num_ctx - s.llm_num_predict - 200) * CHARS_PER_TOKEN
        base = len(build_extract_prompt(s.prompt_version, doc.text, "", capture_labels))
        hint_map = hints.render() if hints else ""
        if hints and base + len(hint_map) > budget:
            hint_map = hints.render(max_hints=TRIMMED_HINTS)
        if base + len(hint_map) <= budget or doc.pages == 1:
            if base + len(hint_map) > budget:
                log.warning("prompt exceeds the context budget; raise LLM_NUM_CTX if output is cut off")
            return self._llm_extract(doc.lines, hint_map, capture_labels)

        log.info("prompt too long for one call; extracting %d pages separately", doc.pages)
        merged = Extraction()
        for page in range(1, doc.pages + 1):
            page_lines = [ln for ln in doc.lines if ln.page == page]
            if not page_lines:
                continue
            ids = {ln.id for ln in page_lines}
            page_hints = hints.render(max_hints=TRIMMED_HINTS, line_ids=ids) if hints else ""
            part = self._llm_extract(page_lines, page_hints, capture_labels)
            for name, fv in part.fields.items():
                # Totals are on the last page that shows them; everything else on the first.
                if name not in merged.fields or name in TOTALS_FIELDS:
                    merged.fields[name] = fv
            merged.line_items.extend(part.line_items)
            merged.table_headers = merged.table_headers or part.table_headers
        return merged

    def _duplicate_file_warning(self, doc: DocText) -> list[str]:
        seen = self.db.seen_file(doc.file_hash)
        return [f"This exact file was already processed (doc_id {seen['doc_id']}, {seen['endpoint']})"] if seen else []

    def _meta(self, t0: float) -> dict:
        return {
            "llm_model": self.settings.llm_model,
            "embed_model": self.settings.embed_model,
            "prompt_version": self.settings.prompt_version,
            "processing_time_ms": int((time.perf_counter() - t0) * 1000),
        }

    @staticmethod
    def _data(ex: Extraction) -> dict:
        return ex.model_dump(exclude_none=True)

    # --- Endpoint 1: ingest -------------------------------------------------------
    def ingest(self, data: bytes, filename: str) -> dict:
        t0 = time.perf_counter()
        doc = self.read_invoice(data)
        ex = self.llm_extract(doc, None, capture_labels=True)
        issues = normalize(ex) + run_checks(ex, doc.lines, check_labels=True)
        score_confidence(ex, doc.lines, issues)
        warnings = self._duplicate_file_warning(doc) + [i.message for i in issues]

        doc_id = uuid.uuid4().hex
        s = self.settings
        self.db.save_pending(doc_id, doc.file_hash, filename, doc.text, self._data(ex), warnings,
                             s.llm_model, s.embed_model, s.prompt_version)
        self.db.record_file(doc.file_hash, doc_id, "ingest")
        return {
            "request_id": request_id_var.get(),
            "doc_id": doc_id,
            "filename": filename,
            "status": "pending",
            "document_type": "invoice",
            "data": self._data(ex),
            "validation_warnings": warnings,
            "meta": self._meta(t0),
            "error": None,
        }

    # --- Endpoint 1.2: confirm ----------------------------------------------------
    def confirm(self, req: ConfirmRequest) -> dict:
        rec = self.db.get_pending(req.doc_id)
        if rec is None:
            raise AppError(404, f"No pending record with doc_id {req.doc_id}")
        if rec["status"] != "pending":
            raise AppError(409, f"Record {req.doc_id} is already {rec['status']}")
        if not req.approved and req.corrected_json is None:
            raise AppError(400, 'Send "approved": true or a corrected_json')

        try:
            ex = Extraction.model_validate(req.corrected_json or rec["extracted"])
        except ValidationError as e:
            raise AppError(400, f"corrected_json is invalid: {e.errors()[:3]}") from e
        lines = parse_rendered(rec["raw_text"])
        issues = normalize(ex) + run_checks(ex, lines, check_labels=True)

        key = vendors.vendor_key(ex)
        invoice_number = ex.fields["invoice_number"].value if "invoice_number" in ex.fields else None
        if key is None or not invoice_number:
            raise AppError(400, "Cannot confirm without a vendor (name or VAT number) and an invoice number")
        duplicate = self.db.find_confirmed(key, str(invoice_number))
        if duplicate and not req.overwrite:
            raise AppError(409, f"Invoice {invoice_number} from this vendor is already confirmed "
                                f"(doc_id {duplicate['doc_id']}). Send \"overwrite\": true to replace it.")

        items, skipped = self._label_items(ex, lines)
        added, updated = self.store.add_labels(items, source="confirmed", vendor=key, increment=not duplicate)
        vendors.update_profile(self.db, ex, count_invoice=not duplicate)
        self.db.save_confirmed(key, str(invoice_number), req.doc_id)
        corrected = req.corrected_json is not None
        self.db.set_pending_status(req.doc_id, "confirmed", was_corrected=corrected,
                                   extracted=self._data(ex) if corrected else None)
        log.info("confirmed %s: %d labels added, %d updated, %d skipped", req.doc_id, added, updated, len(skipped))
        return {
            "request_id": request_id_var.get(),
            "doc_id": req.doc_id,
            "status": "confirmed",
            "was_corrected": corrected,
            "labels_added": added,
            "labels_updated": updated,
            "labels_skipped": skipped,
            "validation_warnings": [i.message for i in issues],
        }

    def _label_items(self, ex: Extraction, lines: list[Line]) -> tuple[list[dict], list[str]]:
        """Turn confirmed labels into library records: label + field + section + value pattern (never the value)."""
        index = {ln.id: i for i, ln in enumerate(lines)}
        header_idx = index.get(ex.table_headers.line) if ex.table_headers else None
        items, skipped = [], []

        for name, fv in ex.fields.items():
            if not fv.label:
                continue
            if not label_in_text(fv.label, fv.line, lines):
                skipped.append(f"{name}: '{fv.label}' is not in the invoice text")
                continue
            ftype = FIELDS[name][0]
            if ftype == "date":
                # The value is ISO by now; the pattern must describe the date as written on the page.
                i = index.get(fv.line)
                window = " ".join(ln.text for ln in lines[i : i + 2]) if i is not None else ""
                pattern = next((p for d, p in find_dates(window) if d.isoformat() == fv.value), "DATE")
            else:
                pattern = value_pattern(fv.value, ftype)

            line_idx = index.get(fv.line)
            if header_idx is not None and line_idx is not None and line_idx < header_idx:
                section = "header"
            elif name in TOTALS_FIELDS:
                section = "totals"
            elif name in BANK_FIELDS or (header_idx is not None and line_idx is not None):
                section = "footer"
            else:
                section = "header"
            items.append({"label": fv.label, "field": name, "section": section, "pattern": pattern})

        if ex.table_headers:
            for label, column in ex.table_headers.columns.items():
                if label_in_text(label, ex.table_headers.line, lines):
                    items.append({"label": label, "field": column, "section": "table", "pattern": None})
                else:
                    skipped.append(f"table header '{label}' is not in the invoice text")
        return items, skipped

    # --- Endpoint 2: extract --------------------------------------------------------
    def extract(self, data: bytes, filename: str, hints_mode: str = "all", save: bool = True) -> dict:
        """hints_mode: 'all' (seed + confirmed), 'seed', or 'off'. Never writes to the library."""
        t0 = time.perf_counter()
        doc = self.read_invoice(data)

        hints = None
        if hints_mode != "off" and len(self.store):
            t = time.perf_counter()
            sources = {"seed"} if hints_mode == "seed" else None
            hints = build_hints([(ln.id, ln.text) for ln in doc.lines], self.store, sources)
            log.info("label lookup: %s (%.1fs)\n%s", hints.stats(), time.perf_counter() - t, hints.render())

        t = time.perf_counter()
        ex = self.llm_extract(doc, hints, capture_labels=False)
        log.info("LLM extraction done (%.1fs)", time.perf_counter() - t)

        issues: list[Issue] = normalize(ex) + run_checks(ex, doc.lines, hints)
        score_confidence(ex, doc.lines, issues, hints)
        vendor_check = vendors.check_vendor(self.db, ex)
        warnings = self._duplicate_file_warning(doc) + [i.message for i in issues]
        review = needs_review(ex, issues) or any("BANK DETAILS CHANGED" in w for w in vendor_check["warnings"])

        doc_id = uuid.uuid4().hex
        result = {
            "request_id": request_id_var.get(),
            "doc_id": doc_id,
            "filename": filename,
            "status": "needs_review" if review else "success",
            "document_type": "invoice",
            "data": self._data(ex),
            "validation_warnings": warnings,
            "label_hints": hints.stats() if hints else HintResult(used=False).stats(),
            "vendor_check": vendor_check,
            "meta": self._meta(t0),
            "error": None,
        }
        if save:
            save_output(self.settings.outputs_dir, doc_id, result)
            self.db.record_file(doc.file_hash, doc_id, "extract")
        return result
