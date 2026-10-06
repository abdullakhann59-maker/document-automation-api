"""Per-field accuracy on held-out invoices: hints OFF vs seed labels vs seed + confirmed labels.

Layout:
  eval/invoices/acme_001.pdf
  eval/ground_truth/acme_001.json   -> {"file": "acme_001.pdf", "fields": {"invoice_number": "INV-204", "total_amount": 120.0}}
Dates in the ground truth are YYYY-MM-DD; numbers are plain numbers.

Run from the project root, with Ollama running:  python eval/run_eval.py
Each invoice is extracted three times, so on CPU allow several minutes per invoice.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.label_store import LabelStore, open_collection  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.ocr import OcrEngine  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.storage import Storage  # noqa: E402

MODES = [("off", "hints off"), ("seed", "seed only"), ("all", "seed+confirmed")]


def matches(expected, actual) -> bool:
    if actual is None:
        return False
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and abs(float(expected) - float(actual)) <= 0.01
    squash = lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())  # noqa: E731
    return squash(expected) == squash(actual)


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    db = Storage(settings.db_path)
    llm = OllamaClient(settings)
    store = LabelStore(db, settings, embed=llm.embed, collection=open_collection(settings))
    pipeline = Pipeline(settings, db, llm, store, OcrEngine(settings))

    truths = sorted((ROOT / "eval" / "ground_truth").glob("*.json"))
    if not truths:
        sys.exit("No ground truth files in eval/ground_truth/")

    # scores[field][mode] = [correct, total]
    scores: dict[str, dict[str, list[int]]] = {}
    for path in truths:
        truth = json.loads(path.read_text(encoding="utf-8"))
        data = (ROOT / "eval" / "invoices" / truth["file"]).read_bytes()
        for mode, _ in MODES:
            print(f"{truth['file']}  [{mode}] ...", flush=True)
            try:
                fields = pipeline.extract(data, truth["file"], hints_mode=mode, save=False)["data"]["fields"]
            except AppError as e:
                print(f"  failed: {e.message}")
                fields = {}
            for name, expected in truth["fields"].items():
                tally = scores.setdefault(name, {m: [0, 0] for m, _ in MODES})[mode]
                tally[1] += 1
                tally[0] += matches(expected, fields.get(name, {}).get("value"))

    header = f"{'field':<24}" + "".join(f"{label:>16}" for _, label in MODES)
    print("\n" + header + "\n" + "-" * len(header))
    totals = {m: [0, 0] for m, _ in MODES}
    for name in sorted(scores):
        row = f"{name:<24}"
        for mode, _ in MODES:
            ok, n = scores[name][mode]
            totals[mode][0] += ok
            totals[mode][1] += n
            row += f"{100 * ok / n:>15.1f}%"
        print(row)
    print("-" * len(header))
    print(f"{'ALL FIELDS':<24}" + "".join(f"{100 * ok / n:>15.1f}%" for ok, n in totals.values()))

    out = {
        "llm_model": settings.llm_model, "embed_model": settings.embed_model,
        "prompt_version": settings.prompt_version, "invoices": len(truths), "scores": scores,
    }
    (ROOT / "eval" / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved eval/results.json")


if __name__ == "__main__":
    main()
