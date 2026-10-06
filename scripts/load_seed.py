"""Load data/seed_labels.json into the label library. Safe to run again: existing labels are left alone.

Run from the project root, with Ollama running:  python scripts/load_seed.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.label_store import LabelStore, open_collection  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.schemas import FIELDS, LINE_ITEM_COLUMNS, SECTIONS  # noqa: E402
from app.storage import Storage  # noqa: E402

_TYPE_PATTERNS = {"date": "DATE", "number": "NUMBER"}


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    store = LabelStore(
        Storage(settings.db_path), settings,
        embed=OllamaClient(settings).embed, collection=open_collection(settings),
    )
    items = []
    for entry in json.loads(settings.seed_path.read_text(encoding="utf-8")):
        defs = LINE_ITEM_COLUMNS if entry["section"] == "table" else FIELDS
        if entry["field"] not in defs or entry["section"] not in SECTIONS:
            sys.exit(f"Bad seed entry: {entry['field']} / {entry['section']}")
        pattern = _TYPE_PATTERNS.get(defs[entry["field"]][0])
        items += [{"label": label, "field": entry["field"], "section": entry["section"], "pattern": pattern}
                  for label in entry["labels"]]
    added, _ = store.add_labels(items, source="seed", increment=False)
    print(f"Seed labels: {added} added, {len(items) - added} already present. Library size: {len(store)}")


if __name__ == "__main__":
    main()
