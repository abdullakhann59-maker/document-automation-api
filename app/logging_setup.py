import contextvars
import logging
import re

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# VAT numbers, sort codes and account numbers must not appear in logs in full.
_MASKS = [
    (re.compile(r"\b(GB|XI)\s?\d{3}\s?\d{4}\s?\d{2}\b", re.I), "GB*********"),
    (re.compile(r"\b\d{2}-\d{2}-\d{2}\b"), "**-**-**"),
    (re.compile(r"\b\d{8}\b"), "********"),
]


def mask(text: str) -> str:
    for rx, repl in _MASKS:
        text = rx.sub(repl, text)
    return text


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.msg = mask(record.getMessage())
        record.args = None
        return True


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    handler.addFilter(_ContextFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
