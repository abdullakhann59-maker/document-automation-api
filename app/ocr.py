"""S1-S4: file checks, text extraction (digital PDF or OCR), and line building."""
import hashlib
import logging
import statistics
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .errors import AppError

log = logging.getLogger(__name__)

MIN_PAGE_TEXT_CHARS = 20   # below this a PDF page is treated as a scan
MIN_DOC_TEXT_CHARS = 30    # below this the document is unreadable


@dataclass
class Fragment:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0


@dataclass
class Line:
    id: str
    page: int
    text: str
    conf: float


@dataclass
class DocText:
    lines: list[Line]
    pages: int
    file_hash: str
    ocr_used: bool

    @property
    def text(self) -> str:
        return render_lines(self.lines)


def render_lines(lines: list[Line]) -> str:
    out, page = [], None
    for ln in lines:
        if ln.page != page:
            page = ln.page
            out.append(f"--- PAGE {page} ---")
        out.append(f"{ln.id}: {ln.text}")
    return "\n".join(out)


def parse_rendered(text: str) -> list[Line]:
    """Inverse of render_lines, for stored raw text (OCR confidence is not kept)."""
    lines, page = [], 1
    for row in text.splitlines():
        if row.startswith("--- PAGE "):
            page = int(row.strip("- ").split()[1])
            continue
        line_id, sep, body = row.partition(": ")
        if sep:
            lines.append(Line(line_id, page, body, 1.0))
    return lines


def sniff_type(data: bytes) -> str | None:
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return None


def build_lines(fragments: list[Fragment]) -> list[tuple[str, float]]:
    """Group fragments into text lines. Wide horizontal gaps become three spaces."""
    frags = [f for f in fragments if f.text.strip()]
    if not frags:
        return []
    tol = 0.5 * statistics.median(max(f.y1 - f.y0, 1.0) for f in frags)
    rows: list[list[Fragment]] = []
    row_y = 0.0
    for f in sorted(frags, key=lambda f: (f.y0 + f.y1) / 2):
        yc = (f.y0 + f.y1) / 2
        if rows and abs(yc - row_y) <= tol:
            rows[-1].append(f)
            row_y += (yc - row_y) / len(rows[-1])
        else:
            rows.append([f])
            row_y = yc

    out = []
    for row in rows:
        row.sort(key=lambda f: f.x0)
        char_w = sum(f.x1 - f.x0 for f in row) / max(1, sum(len(f.text.strip()) for f in row))
        parts = [row[0].text.strip()]
        for prev, cur in zip(row, row[1:]):
            parts.append("   " if cur.x0 - prev.x1 > 2.0 * char_w else " ")
            parts.append(cur.text.strip())
        out.append(("".join(parts), sum(f.conf for f in row) / len(row)))
    return out


class OcrEngine:
    """Lazy EasyOCR reader, loaded once and kept in memory."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._reader = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._reader is not None

    def read(self, image) -> list[Fragment]:
        with self._lock:  # one OCR job at a time on CPU
            if self._reader is None:
                import easyocr

                log.info("loading EasyOCR reader (CPU)")
                self._reader = easyocr.Reader(self.settings.ocr_languages, gpu=False)
            results = self._reader.readtext(image)
        frags = []
        for box, text, conf in results:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            frags.append(Fragment(text, min(xs), min(ys), max(xs), max(ys), float(conf)))
        return frags


def _deskew(gray):
    """Find the small rotation that makes text rows sharpest, and undo it."""
    import cv2
    import numpy as np

    h, w = gray.shape
    scale = 800 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    sh, sw = binary.shape

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, m, (sw, sh), flags=cv2.INTER_NEAREST)
        return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

    angles = [a / 2 for a in range(-10, 11)]  # -5.0 .. 5.0 degrees
    best = max(angles, key=score)
    if abs(best) < 0.5:
        return gray
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(image):
    """Gentle cleanup for scans: grayscale, resize, denoise, deskew, contrast."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    long_side = max(gray.shape)
    if long_side < 1200:
        f = 1600 / long_side
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    elif long_side > 3500:
        f = 3500 / long_side
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    gray = _deskew(gray)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _pdf_pages(path: Path, settings: Settings, ocr: OcrEngine) -> tuple[list[list[Fragment]], bool]:
    import numpy as np

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    try:
        doc = fitz.open(path)
    except Exception as e:
        raise AppError(400, f"Corrupt or unreadable PDF: {e}") from e
    try:
        if doc.needs_pass:
            raise AppError(400, "Password-protected PDFs are not supported")
        if doc.page_count == 0:
            raise AppError(400, "PDF has no pages")
        if doc.page_count > settings.max_pages:
            raise AppError(400, f"PDF has {doc.page_count} pages; the limit is {settings.max_pages}")
        pages, ocr_used = [], False
        for page in doc:
            words = page.get_text("words")
            if sum(len(w[4]) for w in words) >= MIN_PAGE_TEXT_CHARS:
                pages.append([Fragment(w[4], w[0], w[1], w[2], w[3]) for w in words])
                continue
            pix = page.get_pixmap(dpi=settings.pdf_dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                img = img[:, :, 0]
            elif pix.n > 3:
                img = img[:, :, :3]
            pages.append(ocr.read(preprocess(np.ascontiguousarray(img))))
            ocr_used = True
        return pages, ocr_used
    finally:
        doc.close()


def _image_pages(path: Path, settings: Settings, ocr: OcrEngine) -> list[list[Fragment]]:
    import numpy as np
    from PIL import Image, ImageSequence

    try:
        with Image.open(path) as im:
            frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    except Exception as e:
        raise AppError(400, f"Corrupt or unreadable image: {e}") from e
    if len(frames) > settings.max_pages:
        raise AppError(400, f"Image has {len(frames)} pages; the limit is {settings.max_pages}")
    return [ocr.read(preprocess(f)) for f in frames]


def read_document(data: bytes, settings: Settings, ocr: OcrEngine) -> DocText:
    """Validate the upload and return its text as numbered lines."""
    if not data:
        raise AppError(400, "Empty file")
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise AppError(400, f"File is larger than {settings.max_file_mb} MB")
    kind = sniff_type(data)
    if kind is None:
        raise AppError(400, "Unsupported file type; allowed: PDF, JPG, PNG, TIFF")

    file_hash = hashlib.sha256(data).hexdigest()
    path = settings.tmp_dir / f"{uuid.uuid4().hex}.{kind}"
    path.write_bytes(data)
    try:
        if kind == "pdf":
            pages, ocr_used = _pdf_pages(path, settings, ocr)
        else:
            pages, ocr_used = _image_pages(path, settings, ocr), True
    finally:
        path.unlink(missing_ok=True)

    lines: list[Line] = []
    for page_no, frags in enumerate(pages, start=1):
        for text, conf in build_lines(frags):
            lines.append(Line(f"L{len(lines) + 1}", page_no, text, conf))
    if sum(len(ln.text) for ln in lines) < MIN_DOC_TEXT_CHARS:
        raise AppError(422, "Unreadable document: no text found")
    return DocText(lines, len(pages), file_hash, ocr_used)
