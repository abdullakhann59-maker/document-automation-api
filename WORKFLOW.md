# Invoice Extraction with Label-RAG: Complete Workflow (UK VAT edition)

**Decisions this version is built on**

| Decision | Choice | What it changes |
|---|---|---|
| Invoice type | UK VAT invoices (the UK has VAT, not GST) | VAT number, VAT rates, GBP, sort code / account number. No GSTIN, CGST, SGST, IGST. |
| Input | Mixed: digital PDFs and scans/photos, single or multi-page, English | Digital PDFs are read directly; everything else goes through OCR |
| Hardware | CPU only | One LLM call at a time, long timeouts, keyword classifier before the LLM, short JSON output |
| Fields | All detail fields (section 6.1) | Wider schema: parties, dates, totals, bank details, line items |
| Volume | About 100 invoices per day | Sequential processing is enough; no queue system needed |
| Orchestration | Direct calls to Ollama and ChromaDB | No LangChain |

**Reality check:** this design improves a small local LLM's accuracy, but it won't reach 100%. Every result goes through automatic checks, and anything doubtful is flagged `needs_review` instead of being passed through silently.

---

## 1. System overview

**Goal:** convert invoice PDFs and images into accurate, structured JSON using free local tools.

**The problem being solved:** different vendors write the same field differently ("Bill No.", "Inv #", "VAT Invoice No." all mean invoice number). A free 7B model working from a prompt alone often misses or confuses these.

**The solution (label-RAG):** build a library of confirmed label variations. When a new invoice arrives, look up each label in the library and give the LLM a hint map ("line 4 'Bill No.' = invoice number") before it extracts.

| Stage | Endpoint | What it does | Writes to label library? |
|---|---|---|---|
| Learn | `POST /ingest` (Endpoint 1) | Extract values **and the labels as written** from clean invoices | No (saves as pending) |
| Approve | `POST /confirm` (Endpoint 1.2) | Store reviewed labels in the vector DB, update vendor profiles | **Yes** |
| Use | `POST /extract` (Endpoint 2) | Look up labels in the new invoice, then extract with hints | No (saves JSON file) |

**Core principles:**

1. Only human-confirmed labels enter the library.
2. Embed **labels**, not whole invoices, and search **label against label**.
3. Store label **patterns** (like `INV-###`), never real values, so the LLM can't copy old values.
4. The same embedding model for storing and searching.
5. Hints are suggestions. If the invoice context contradicts a hint, the context wins.
6. Never silently change values. Flag problems as warnings.
7. Hints can be switched on and off, so you can measure whether they help.

---

## 2. Technology stack

| Layer | Tool | Purpose |
|---|---|---|
| API | FastAPI, Uvicorn, python-multipart | Endpoints and file uploads |
| PDF handling | PyMuPDF | PDF to image, and reading text directly from digital PDFs |
| Image cleanup | OpenCV, NumPy, Pillow | Grayscale, denoise, deskew, contrast |
| OCR | EasyOCR (+ PyTorch, CPU build) | Text with positions and confidence |
| LLM | Ollama: Qwen2.5 7B (or `qwen2.5:3b` if 7B is too slow on your CPU) | Extraction and classification |
| Embeddings | Ollama: `nomic-embed-text` | Convert labels to vectors |
| Fuzzy matching | RapidFuzz | Fast dictionary lookup for close label spellings |
| HTTP client | httpx | Direct calls to the Ollama REST API |
| Vector DB | ChromaDB (persistent), called directly | Label library vectors |
| Relational DB | SQLite | Pending records, vendor profiles, label records and counts |
| Validation | Pydantic, pydantic-settings | Schemas and config |
| Testing | pytest | Unit tests and evaluation |

LangChain is not used. Verify current package versions in the official docs before installing.

---

## 3. Project structure

```
invoice-label-rag/
├── app/
│   ├── main.py             # FastAPI app, startup, routes
│   ├── config.py           # All settings from .env
│   ├── schemas.py          # Field definitions, Pydantic models
│   ├── ocr.py              # File checks, PDF→text/image, preprocessing, EasyOCR, line building
│   ├── classifier.py       # Is it an invoice? (keywords first, LLM if unclear)
│   ├── labels.py           # Label normalization, value patterns, candidate label detection
│   ├── label_store.py      # Dictionary lookup + vector DB store/search
│   ├── hints.py            # Conflict resolution and the label hint map
│   ├── prompts.py          # Versioned prompt templates
│   ├── llm.py              # Ollama calls, JSON parsing, retries
│   ├── validation.py       # Math, date, VAT number, bank, pattern checks, confidence
│   ├── vendors.py          # Vendor profiles and anomaly checks
│   ├── storage.py          # SQLite (pending, vendors, labels), output files
│   ├── pipeline.py         # Shared S1–S5 pipeline and the extraction flow
│   └── logging_setup.py
├── data/
│   ├── seed_labels.json    # Starter label variations
│   ├── chroma/             # Vector DB
│   ├── app.db              # SQLite
│   ├── outputs/            # Endpoint 2 JSON results
│   └── tmp/                # Uploads (auto-deleted)
├── eval/
│   ├── invoices/           # Held-out invoice files
│   ├── ground_truth/       # One JSON per held-out invoice
│   └── run_eval.py         # Hints OFF vs seed vs seed+confirmed
├── scripts/
│   ├── load_seed.py        # Loads seed labels into the library
│   └── reindex.py          # Rebuilds vectors if the embedding model changes
├── tests/
├── .env
└── requirements.txt
```

---

## 4. Environment setup

1. Install Python 3.10+ and create a virtual environment.
2. Install Ollama and start it (port 11434).
3. `ollama pull qwen2.5:7b` and `ollama pull nomic-embed-text`; test that both respond.
4. Install the CPU build of PyTorch first (it is much smaller than the CUDA build), then `pip install -r requirements.txt`.
5. Edit `.env` (set a real `API_KEY`).
6. `python scripts/load_seed.py`, then `uvicorn app.main:app`.

**CPU expectations:** a 7B model on CPU produces a few tokens per second, so one invoice takes roughly 1 to 4 minutes depending on the machine. 100 invoices a day fits in a few hours of sequential processing. You need about 8 GB of free RAM for the 7B model; with less, use `qwen2.5:3b` and expect lower accuracy.

---

## 5. Configuration (`.env` / `config.py`)

| Setting | Starting value | Notes |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:7b` | |
| `EMBED_MODEL` | `nomic-embed-text` | Must never change without a reindex |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `LLM_TEMPERATURE` | 0 | Deterministic output |
| `LLM_NUM_CTX` | 8192 | Set explicitly; Ollama's default is too small for an invoice plus hints |
| `LLM_NUM_PREDICT` | 2048 | Output token cap |
| `LLM_TIMEOUT_SEC` | 600 | CPU inference is slow; 120 s is not enough |
| `LLM_MAX_RETRIES` | 2 | |
| `OCR_LANGUAGES` | `["en"]` | |
| `PDF_DPI` | 250 | |
| `MAX_FILE_MB` / `MAX_PAGES` | 10 / 10 | |
| `HINTS_ENABLED` | true | Switch for testing |
| `FUZZY_MATCH_MIN` | 90 (out of 100) | Dictionary match threshold |
| `VECTOR_TOP_K` | 3 | |
| `VECTOR_SIM_THRESHOLD` | 0.75 | Cosine similarity; tune in testing. Below this, no hint is given |
| `MIN_LABEL_COUNT` | 1 | Raise later to trust only repeated labels (seed labels are always trusted) |
| `PROMPT_VERSION` | `v1` | |
| `API_KEY` | secret | Sent as the `X-API-Key` header |

---

## 6. Fields and the seed label library

### 6.1 Fields extracted

| Group | Fields |
|---|---|
| Vendor | `vendor_name`, `vendor_address`, `vendor_vat_number`, `vendor_company_number`, `vendor_email`, `vendor_phone` |
| Buyer | `buyer_name`, `buyer_address`, `buyer_vat_number`, `delivery_address` |
| Document | `invoice_number`, `invoice_date`, `tax_point_date`, `due_date`, `po_number`, `payment_terms`, `currency` |
| Totals | `subtotal` (net), `discount`, `shipping`, `vat_rate`, `vat_amount`, `total_amount` (gross), `amount_paid`, `balance_due` |
| Bank | `bank_name`, `bank_account_name`, `bank_sort_code`, `bank_account_number`, `iban`, `swift_bic` |
| Line items | `description`, `product_code`, `quantity`, `unit_price`, `vat_rate`, `vat_amount`, `line_total` |

Required: `invoice_number`, `invoice_date`, `vendor_name`, `total_amount`.

### 6.2 Seed labels

`data/seed_labels.json` makes Endpoint 2 useful from day one. Examples:

| Standard field | Section | Label variations |
|---|---|---|
| `invoice_number` | header | Invoice No, Inv No, Inv #, Bill No, Tax Invoice No, VAT Invoice No, Document No, Invoice Ref |
| `invoice_date` | header | Invoice Date, Inv Date, Bill Date, Date of Invoice, Date, Issue Date |
| `tax_point_date` | header | Tax Point, Tax Point Date, Date of Supply |
| `due_date` | header | Due Date, Payment Due, Pay By |
| `po_number` | header | PO No, P.O. Number, Purchase Order, Order Ref, Your Ref |
| `vendor_vat_number` | header | VAT No, VAT Number, VAT Reg No, VAT Registration No |
| `vendor_company_number` | header | Company No, Company Reg No, Registered No |
| `buyer_name` | header | Bill To, Billed To, Invoice To, Customer, Sold To |
| `buyer_vat_number` | header | Customer VAT No, Buyer VAT No, Your VAT No |
| `subtotal` | totals | Sub Total, Net Total, Net Amount, Total Excl VAT, Total Ex VAT |
| `vat_amount` | totals | VAT, Total VAT, VAT Amount, VAT @ 20% |
| `discount` | totals | Discount, Less Discount, Rebate |
| `shipping` | totals | Shipping, Delivery, Carriage, P&P |
| `total_amount` | totals | Total, Grand Total, Invoice Total, Total Inc VAT, Total Due, Amount Due |
| `bank_sort_code` | footer | Sort Code, S/C |
| `bank_account_number` | footer | Account No, Account Number, Acc No, A/C No |
| `quantity` | table | Qty, Quantity, Units, Hours |
| `unit_price` | table | Rate, Unit Price, Price, Unit Cost |
| `line_total` | table | Amount, Total, Line Total, Net |
| `vat_rate` | table | VAT %, VAT Rate |

Some labels appear under more than one field on purpose ("Total" is a table column and a totals line; "VAT" is a column and a totals line). Section 11.4 resolves these.

Run `scripts/load_seed.py` to embed these and load them with `source: seed`.

---

## 7. Startup (once, when the server starts)

1. Load settings and create the data folders.
2. Open SQLite (pending records, vendor profiles, label records, file hashes).
3. Check that Ollama is running and both models are pulled. Stop with a clear error if not.
4. Warm up both models with one tiny call (on CPU the first call loads the model, which takes a while).
5. Open the ChromaDB `labels` collection. Check that its stored embedding model name matches `EMBED_MODEL`. If not, refuse to start and ask for `reindex.py`.
6. **Build the in-memory dictionary:** every confirmed and seed label (normalized) → standard field + section + pattern + count.
7. The EasyOCR reader is loaded on first use and kept in memory (digital PDFs never need it).

---

## 8. Shared pipeline (both endpoints)

### S1. File validation
- Allowed types: PDF, JPG, PNG, TIFF, checked by the file's first bytes, not the extension.
- Reject empty, corrupt, password-protected, or oversized files, and PDFs over the page limit.
- Save with a UUID name in `data/tmp/`, and always delete it after processing, even on error.
- Compute a SHA-256 file hash to detect re-uploads of the same file.

### S2. Get text-ready pages (decided per page)
- **PDF page with a text layer:** read words and positions directly with PyMuPDF. Faster and more accurate than OCR, and on CPU it skips the slowest non-LLM step.
- **PDF page without a usable text layer:** render at `PDF_DPI` and go to S3.
- **Image file:** go straight to S3.

### S3. Image preprocessing (scans and images only)
- Grayscale, light denoise, deskew
- Contrast boost (CLAHE) for faded scans
- Resize if too small or too large
- Keep it gentle, and test on your real invoices. Over-processing can hurt OCR.

### S4. OCR and line building
- EasyOCR (CPU) returns text fragments with position boxes and confidence.
- **Build lines:** group fragments with similar vertical position into one line, sort left to right. A wide horizontal gap is kept as three spaces, so columns and `label   value` pairs stay separable.
- Number every line (`L1`, `L2`, ...), since hints refer to line numbers.
- Record confidence per line (digital text counts as 1.0).
- Join pages with `--- PAGE n ---`.
- Empty or very short text → error "unreadable document."

### S5. Classification
- **Keywords first:** if the top of the document says "invoice" and nothing says credit note, statement, or purchase order, it is an invoice. No LLM call (saves a minute on CPU).
- **Otherwise:** short LLM prompt with a closed list: invoice, receipt, purchase order, credit note, bank statement, other.
- Not an invoice → stop, return the detected type, store nothing.

---

## 9. Endpoint 1: Ingest (`POST /ingest`)

**Purpose:** extract values **and their labels as written** from clean invoices, for review.

1. Accept one or more files.
2. Run S1 to S5 for each.
3. **LLM extraction with the label-capture prompt.** For every field found, the LLM returns:
   ```json
   "invoice_number": {"value": "INV-204", "label": "Bill No.", "line": "L4"}
   ```
   - For fields with no label (e.g. vendor name at the top), `label` is `null`.
   - Fields not on the invoice are left out (shorter output = faster on CPU).
   - For line items, the **table column headers** as written (`"Qty"`, `"Rate"`, `"Amt"`) mapped to standard columns.
4. Parse and validate the JSON, with retries.
5. **Label sanity check:** confirm each label actually appears in the raw text on or next to the stated line. If not, flag it, since the LLM may have invented it.
6. Run the business checks (section 11.8).
7. Save to the **pending** table: `doc_id`, file hash, raw text with line numbers, extracted JSON, warnings, `status: pending`, models, prompt version, timestamp.
8. Return the results for review.
9. In a batch, one failed file doesn't stop the others.

**Review routes:**
- `GET /pending`: list records waiting for review
- `GET /pending/{doc_id}`: full record with raw text and the extracted fields side by side

---

## 10. Endpoint 1.2: Confirm and store (`POST /confirm`)

**Request:** `doc_id` + `"approved": true` or `corrected_json` (with corrected values and labels), optional `overwrite`.

1. Load the pending record. If it's missing or already confirmed, return an error.
2. Validate any corrections with the same schema and checks.
3. **Duplicate invoice check:** vendor VAT number (or vendor name if there is none) + invoice number already confirmed → reject (409), or replace if `overwrite: true`.
4. **For each field with a label, create or update a label record:**
   - **Normalize** the label: lowercase, remove dots, trailing colons and punctuation, collapse spaces (`"Bill No. :"` → `"bill no"`).
   - Labels that don't appear in the invoice's raw text are skipped and reported.
   - **Already exists** with the same standard field and section → increase its `count`, add this vendor to its vendor list.
   - **New** → embed the label with `EMBED_MODEL` and store it in ChromaDB (metadata: normalized label, field, section, source) and SQLite (pattern, count, vendors, timestamp).
   - **Same label, different field** (e.g. "Date") → separate records per field and section. Endpoint 2 handles the conflict.
5. **Value pattern, not value:** convert the value to a pattern before storing (`INV-204` → `INV-###`, `12/03/2026` → `DD/MM/YYYY`, any amount → `NUMBER`, any free text → `TEXT`). Real values never enter the library.
6. **Update the vendor profile** in SQLite: vendor name, VAT number, usual currency, usual VAT rate, bank details (hash plus last 4 digits only), and the running min/max/average total.
7. Add new labels to the in-memory dictionary immediately.
8. Mark the pending record `confirmed`, with a `was_corrected` flag.
9. Return success with the number of labels added and updated.

**Library management routes:**
- `POST /reject`: mark a pending record rejected (never stored)
- `GET /labels`: list the library (filter by field, source, or count)
- `DELETE /labels/{id}`: remove a wrong mapping
- `GET /vendors`: list vendor profiles

---

## 11. Endpoint 2: Extract with label hints (`POST /extract`)

**Purpose:** extract details from new invoices using the label library. **Read-only**: nothing is written to the library or vendor profiles. `?hints=false` switches hints off for one request.

### 11.1 Read the document
Run S1 to S5.

### 11.2 Find candidate labels
Each line is split at wide gaps into segments. For each segment:
- Text before a colon (`Bill No.: INV-204` → `Bill No.`)
- Text before the first number or value-like token
- Short all-text segments (labels placed above or beside their value)
- **Table header row:** a line with three or more short text-only segments. Each is a column label.
- Two short text lines in a row are also tried joined (a label split by OCR).
- Segments that are clearly values only (numbers, long text) are skipped.

### 11.3 Look up each candidate (two layers)
1. **Dictionary first:** normalize, then exact match, then fuzzy match (RapidFuzz ≥ `FUZZY_MATCH_MIN`).
2. **Vector search for the rest:** embed all remaining explicit candidates in **one batch call**, search ChromaDB for the top `VECTOR_TOP_K`, keep results above `VECTOR_SIM_THRESHOLD`.
3. No match → no hint for that line.

### 11.4 Resolve conflicts
- One label matching several fields ("Total" → table column or grand total; "Date" → invoice date or tax point): use the **section** (header before the table, table header row, totals/footer after it) and the **value pattern** of the value on that line. If still unclear, pass both options with their counts.
- Two lines mapping to the same field (two "Total" lines): pass both with a note that the later one in the totals section is usually the grand total. The LLM decides, the math checks verify.

### 11.5 Build the hint map
```
LABEL HINTS (suggestions from confirmed invoices):
L4   "Bill No."        → invoice_number  (pattern INV-###, seen 23×)
L6   "Dated"           → invoice_date    (pattern DD/MM/YYYY, seen 15×)
L12  "VAT Reg No"      → vendor_vat_number  (pattern GB#########, seen 31×)
L18  table header      → Qty=quantity, Rate=unit_price, Amt=line_total
L31  "Total Due"       → total_amount    (seen 9×)
```

### 11.6 Build the prompt
1. Role and task
2. Field list with definitions
3. Rules: JSON only, leave out missing fields, never invent, dates as YYYY-MM-DD, numbers without symbols or commas
4. **Unlabeled-field rules:** vendor name is usually the first business name at the top; buyer details follow "Bill To"/"Invoice To"
5. **The hint map**, with the rule: "Hints are suggestions from past invoices. If the invoice text clearly contradicts a hint, follow the text."
6. The raw text with line numbers
7. **Token budget check:** if too long, shorten the hints first (keep the highest-count ones). Never cut the invoice text. Still too long: extract per page, then merge.

### 11.7 Extract and validate
- Call Ollama (temperature 0, JSON mode, explicit `num_ctx`, timeout). One LLM call runs at a time.
- Parse, strip stray text, validate with Pydantic.
- On failure, retry with the error message. Still failing → error with the raw output.
- The LLM returns the **line number** for each value, so each value can be traced to the source text.

### 11.8 Business-rule checks
Each failure adds a `validation_warnings` entry, with no auto-fixing:
- Sum of line totals = subtotal (small rounding tolerance)
- Subtotal + VAT − discount + shipping = total
- Quantity × unit price = line total
- VAT amount = subtotal × VAT rate, when a single rate is given
- VAT rate is one of the UK rates (20, 5, 0)
- Total − amount paid = balance due
- Due date not before invoice date; dates realistic and not in the future
- VAT number: `GB` + 9 or 12 digits (or `GBGD`/`GBHA` + 3, or `XI`), with the mod-97 checksum
- Company number: 8 characters; sort code: 6 digits; account number: 8 digits; IBAN checksum
- Value matches the expected pattern from its hint
- Each value actually appears on (or next to) the line the LLM cited
- Required fields present

### 11.9 Vendor profile checks (read-only)
If the vendor VAT number or name matches a stored profile:
- Vendor name slightly different (OCR error) → return `vendor_name_raw` and `vendor_name_suggested`, with a warning
- Different currency or VAT rate than usual → warning
- Total far outside the usual range → warning
- **Bank details different from the stored ones** → strong warning (a common fraud signal)

### 11.10 Confidence and output
- Score each field from the OCR confidence of its line, hint agreement, and passed checks → `high / medium / low`.
- Any low-confidence or missing required field, or any math failure → `status: needs_review`.
- Save the result to `data/outputs/{doc_id}.json` and return the response.

---

## 12. Standard response format

```json
{
  "request_id": "uuid",
  "doc_id": "uuid",
  "status": "success | needs_review | error",
  "document_type": "invoice",
  "data": {
    "fields": {
      "invoice_number": {"value": "INV-204", "line": "L4", "confidence": "high"},
      "total_amount":   {"value": 1180.00, "line": "L31", "confidence": "high"}
    },
    "line_items": [
      {"line": "L19", "description": "Widget", "quantity": 2, "unit_price": 50.0, "line_total": 100.0}
    ]
  },
  "validation_warnings": ["VAT amount 190.00 does not equal subtotal 1000.00 × 20%"],
  "label_hints": {
    "used": true,
    "matched": 9,
    "from_dictionary": 7,
    "from_vector_search": 2,
    "unmatched_labels": ["Ref Code"]
  },
  "vendor_check": {"known_vendor": true, "warnings": []},
  "meta": {
    "llm_model": "qwen2.5:7b",
    "embed_model": "nomic-embed-text",
    "prompt_version": "v1",
    "processing_time_ms": 95000
  },
  "error": null
}
```

`unmatched_labels` shows which label variations to add or review next.

**HTTP status codes:**

| Code | When |
|---|---|
| 200 | Success, including results with warnings |
| 400 | Bad file: wrong type, corrupt, empty, too large |
| 401 | Missing or wrong API key |
| 404 | `doc_id` or label not found |
| 409 | Duplicate invoice in 1.2 |
| 422 | Not an invoice, or unreadable |
| 500 | Unexpected error |
| 503 | Ollama or ChromaDB unavailable |
| 504 | LLM timeout |

---

## 13. Error handling and edge cases

| Situation | Handling |
|---|---|
| Unsupported, corrupt, or password-protected file | 400 with a clear message |
| Blank page, no text found | 422 "unreadable document" |
| Skewed scan | Deskew in S3 |
| Handwritten invoice | Low OCR confidence → `needs_review` |
| Labels above values instead of beside them | Short text segments are candidates; value checks look at the next line too |
| Label split across two lines by OCR | Adjacent short lines are also tried joined |
| OCR typo in a label ("lnvoice No") | Fuzzy match catches it |
| Same label, different meanings | Section + value pattern resolution (11.4) |
| EU or other non-UK vendor VAT number | Format accepted, checksum not checked |
| Several invoices in one PDF | Not split automatically in v1; the math checks will usually flag it for review |
| Text too long for the context | Trim hints first, then extract per page and merge |
| Malformed LLM JSON | Retry with the error, then return the raw output |
| Ollama down or slow | 503 / 504, never hang |
| Empty library | Run without hints |
| Embedding model changed | Startup blocks until `reindex.py` is run |
| Same file uploaded twice | File hash → warning |
| Wrong label mapping got confirmed | `DELETE /labels/{id}`; high-count labels outweigh one-off mistakes |

---

## 14. Professional practices

**Logging and traceability**
- A `request_id` on every request and every log line
- Each stage logged with timing: text extraction/OCR, classification, label lookup, LLM, validation
- The hint map used for each extraction is logged, so wrong outputs can be traced to a bad hint
- VAT numbers and bank details are masked in logs

**Versioning**
- LLM model, embed model, and prompt version stored with every result
- Bump `PROMPT_VERSION` on any prompt change and re-run the evaluation

**Performance (CPU)**
- Endpoints run in worker threads so the API stays responsive
- One LLM call at a time (a semaphore); parallel calls on CPU only slow each other down
- All label embeddings per invoice in one batch call
- Digital PDFs skip OCR; obvious invoices skip the classification LLM call
- For the daily batch, send files to `/extract` one after another from a script; 100/day needs no queue

**Security**
- API key on all routes except `/health`
- File size and page limits; uploads deleted after processing
- Restrict access to `data/`, and back up `chroma/` and `app.db` regularly

**Monitoring the library's health**
- Correction rate in 1.2 (high = the prompt or hints need work)
- Most common `unmatched_labels` (candidates to add)
- Labels mapped to more than one field (review these)

**Health check: `GET /health`**
- Ollama (both models), ChromaDB (label count), SQLite, EasyOCR loaded, free disk space

---

## 15. Testing and evaluation

**Unit tests (pytest):** line building, label normalization, candidate detection, dictionary exact and fuzzy matching, conflict resolution, value-to-pattern conversion, JSON parsing with messy model output, every math/date/VAT/bank rule.

**Accuracy evaluation (the most important test):**
1. Collect invoices from your **real vendor mix**, including vendors with unusual labels.
2. Split them: about 70% for ingest + confirm, 30% held out with correct answers written manually (`eval/ground_truth/`).
3. Run `eval/run_eval.py` on the held-out set three ways: hints OFF, seed labels only, seed + confirmed labels.
4. Measure accuracy **per field**.
5. Label-RAG is earning its place only if the third run beats the first.
6. Tune one setting at a time: fuzzy threshold, vector threshold, top-k, prompt wording, LLM model, DPI.
7. Test failure cases: bad scans, non-invoices, conflicting labels, Ollama stopped.

**Go-live target:** set a minimum per-field accuracy (e.g. 95% on invoice number, date, VAT number, total). Fields below it go to `needs_review`.

---

## 16. Build order

| # | Build | Done when |
|---|---|---|
| 1 | Ollama + models | Both respond to a test call |
| 2 | `ocr.py` | Clean, numbered lines from 5 sample invoices (scanned and digital) |
| 3 | `labels.py` | Correct candidate labels and table headers found on those 5 |
| 4 | `prompts.py` + `llm.py` (no hints) | Valid JSON with values, labels, and line numbers |
| 5 | `schemas.py` + `validation.py` | Bad JSON and math errors caught |
| 6 | `seed_labels.json` + `label_store.py` + `load_seed.py` | Dictionary and vector lookups return the right fields |
| 7 | `storage.py` + Endpoint 1 + pending routes | Upload → review a pending record |
| 8 | Endpoint 1.2 + `vendors.py` | Labels counted and stored; vendor profiles updated |
| 9 | `hints.py` + Endpoint 2 with on/off switch | The same invoice works with and without hints |
| 10 | Conflict resolution, vendor checks, confidence | Ambiguous labels resolved; anomalies flagged |
| 11 | Error handling, logging, health check, API key | Every case in section 13 gives the right response |
| 12 | Evaluation | Three-way accuracy numbers recorded |
| 13 | Tune and freeze `PROMPT_VERSION` | Accuracy target met |

---

## 17. Possible later upgrades

- If **line items or vendor name** are still often wrong, store whole confirmed invoices as layout examples for repeat vendors and include one in the prompt. Keep it off until the evaluation shows label hints alone aren't enough.
- Automatic splitting of PDFs that contain several invoices.
- Reverse-charge and mixed-rate VAT summaries as structured fields.
