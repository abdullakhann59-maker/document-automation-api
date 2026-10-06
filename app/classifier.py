"""S5: is it an invoice? Keywords decide the obvious cases; the LLM decides the rest."""
import re

from .llm import OllamaClient
from .ocr import DocText, render_lines
from .prompts import build_classify_prompt
from .schemas import DOCUMENT_TYPES

_OTHER_TYPES = re.compile(
    r"credit\s+note|credit\s+memo|statement|purchase\s+order|receipt|remittance|quotation|quote\b|"
    r"estimate|pro[\s-]?forma|delivery\s+note|order\s+confirmation",
    re.I,
)


def classify_by_keywords(lines: list[str]) -> str | None:
    """Returns 'invoice' when the top of the document clearly says so, else None (ask the LLM)."""
    top = "\n".join(lines[:25])
    if not re.search(r"\binvoice\b", top, re.I):
        return None
    # "Purchase Order No" is a normal field on an invoice, so only the label-free title counts.
    stripped = re.sub(r"(?i)purchase\s+order\s*(no|number|ref|#|:)", "", top)
    return None if _OTHER_TYPES.search(stripped) else "invoice"


def classify(doc: DocText, llm: OllamaClient, prompt_version: str) -> str:
    quick = classify_by_keywords([ln.text for ln in doc.lines])
    if quick:
        return quick
    prompt = build_classify_prompt(prompt_version, render_lines(doc.lines[:60]))

    def validate(obj: dict) -> str:
        doc_type = str(obj.get("document_type", "")).strip().lower().replace(" ", "_")
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(f"document_type must be one of {', '.join(DOCUMENT_TYPES)}")
        return doc_type

    return llm.generate_json(prompt, validate, num_predict=30)
