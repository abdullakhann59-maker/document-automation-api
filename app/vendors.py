"""Vendor profiles: updated on confirm, checked read-only on extract."""
import hashlib
import re

from rapidfuzz import fuzz

from .schemas import Extraction
from .storage import Storage, now_iso

NAME_MATCH_MIN = 90     # fuzzy score to treat a name as a known vendor when there is no VAT number
RANGE_FACTOR = 3        # total more than 3× above max or below min is unusual
MIN_HISTORY = 3         # invoices needed before the range check applies


def _val(ex: Extraction, name: str):
    fv = ex.fields.get(name)
    return fv.value if fv else None


def _norm_vat(value) -> str | None:
    v = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return v or None


def _norm_name(value) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(value or "").lower())).strip()


def vendor_key(ex: Extraction) -> str | None:
    vat = _norm_vat(_val(ex, "vendor_vat_number"))
    if vat:
        return vat
    name = _norm_name(_val(ex, "vendor_name"))
    return f"name:{name}" if name else None


def bank_fingerprint(ex: Extraction) -> tuple[str, str] | None:
    """(sha256 hash, last 4 characters) of the bank identifier. The identifier itself is never stored."""
    iban = re.sub(r"[^A-Z0-9]", "", str(_val(ex, "iban") or "").upper())
    account = re.sub(r"\D", "", str(_val(ex, "bank_sort_code") or "") + str(_val(ex, "bank_account_number") or ""))
    ident = account if len(account) >= 8 else iban
    if not ident:
        return None
    return hashlib.sha256(ident.encode()).hexdigest(), ident[-4:]


def find_profile(db: Storage, ex: Extraction) -> dict | None:
    vat = _norm_vat(_val(ex, "vendor_vat_number"))
    if vat:
        profile = db.one("SELECT * FROM vendors WHERE key = ?", (vat,))
        if profile:
            return profile
    name = _norm_name(_val(ex, "vendor_name"))
    if not name:
        return None
    best, best_score = None, 0.0
    for profile in db.query("SELECT * FROM vendors"):
        score = fuzz.ratio(name, _norm_name(profile["name"]))
        if score >= NAME_MATCH_MIN and score > best_score:
            best, best_score = profile, score
    return best


def check_vendor(db: Storage, ex: Extraction) -> dict:
    """Compare an extraction with the stored profile. Read-only."""
    profile = find_profile(db, ex)
    if profile is None:
        return {"known_vendor": False, "warnings": []}
    result: dict = {"known_vendor": True, "warnings": []}
    warn = result["warnings"].append

    name = _val(ex, "vendor_name")
    if name and profile["name"] and _norm_name(name) != _norm_name(profile["name"]):
        result["vendor_name_raw"] = name
        result["vendor_name_suggested"] = profile["name"]
        warn(f"Vendor name '{name}' differs from the stored name '{profile['name']}'")

    currency = _val(ex, "currency")
    if currency and profile["currency"] and str(currency).upper() != profile["currency"]:
        warn(f"Currency {currency} differs from this vendor's usual {profile['currency']}")

    rate = _val(ex, "vat_rate")
    if isinstance(rate, (int, float)) and profile["vat_rate"] is not None and rate != profile["vat_rate"]:
        warn(f"VAT rate {rate:g}% differs from this vendor's usual {profile['vat_rate']:g}%")

    total = _val(ex, "total_amount")
    if isinstance(total, (int, float)) and profile["count"] >= MIN_HISTORY:
        if total > profile["max_total"] * RANGE_FACTOR or total < profile["min_total"] / RANGE_FACTOR:
            warn(f"Total {total:.2f} is far outside this vendor's usual range "
                 f"({profile['min_total']:.2f} to {profile['max_total']:.2f})")

    bank = bank_fingerprint(ex)
    if bank and profile["bank_hash"] and bank[0] != profile["bank_hash"]:
        warn(f"BANK DETAILS CHANGED: the account on this invoice does not match the one on file "
             f"(ending {profile['bank_last4']}). Verify with the vendor before paying.")
    return result


def update_profile(db: Storage, ex: Extraction, count_invoice: bool = True) -> None:
    key = vendor_key(ex)
    if key is None:
        return
    total = _val(ex, "total_amount")
    total = float(total) if isinstance(total, (int, float)) else None
    rate = _val(ex, "vat_rate")
    rate = float(rate) if isinstance(rate, (int, float)) else None
    currency = str(_val(ex, "currency")).upper() if _val(ex, "currency") else None
    bank = bank_fingerprint(ex) or (None, None)
    old = db.one("SELECT * FROM vendors WHERE key = ?", (key,))

    if old is None:
        db.execute(
            "INSERT INTO vendors (key, name, vat_number, currency, vat_rate, bank_hash, bank_last4,"
            " count, min_total, max_total, avg_total, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, _val(ex, "vendor_name"), _norm_vat(_val(ex, "vendor_vat_number")), currency, rate,
             bank[0], bank[1], 1 if total is not None else 0, total, total, total, now_iso()),
        )
        return

    count, lo, hi, avg = old["count"], old["min_total"], old["max_total"], old["avg_total"]
    if count_invoice and total is not None:
        lo = total if lo is None else min(lo, total)
        hi = total if hi is None else max(hi, total)
        avg = total if not count else (avg * count + total) / (count + 1)
        count += 1
    db.execute(
        "UPDATE vendors SET name = ?, currency = ?, vat_rate = ?, bank_hash = ?, bank_last4 = ?,"
        " count = ?, min_total = ?, max_total = ?, avg_total = ?, updated_at = ? WHERE key = ?",
        (old["name"] or _val(ex, "vendor_name"), currency or old["currency"],
         rate if rate is not None else old["vat_rate"], bank[0] or old["bank_hash"],
         bank[1] or old["bank_last4"], count, lo, hi, avg, now_iso(), key),
    )
