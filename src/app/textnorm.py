"""Turkish plan text: case folding, old-codepage repair, room labels, areas, scale notes."""
import re, difflib

MOJIBAKE = {"þ": "ş", "ý": "ı", "ð": "ğ", "Þ": "Ş", "Ý": "İ", "Ð": "Ğ"}
FOLD = str.maketrans({"Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U", "Â": "A", "Î": "I", "Û": "U"})

# Order matters: first match wins ("EBEVEYN BANYO" must be a bathroom, not a bedroom).
ROOM_WORDS = [
    ("BANYO", "bathroom", {}), ("DUS", "bathroom", {}),
    ("WC", "wc", {}), ("W C", "wc", {}), ("TUVALET", "wc", {}), ("LAVABO", "wc", {}),
    ("AMERIKAN MUTFAK", "kitchen", {"open": True}), ("ACIK MUTFAK", "kitchen", {"open": True}),
    ("MUTFAK", "kitchen", {}),
    ("BALKON", "balcony", {}), ("TERAS", "balcony", {}),
    ("SALON", "living", {}), ("OTURMA", "living", {}), ("MISAFIR", "living", {}),
    ("YEMEK", "dining", {}), ("CALISMA", "study", {}),
    ("EBEVEYN", "bedroom", {"master": True}), ("COCUK", "bedroom", {"child": True}),
    ("YATAK ODASI", "bedroom", {}), ("Y ODASI", "bedroom", {}), ("YATAK", "bedroom", {}),
    ("HOL", "hall", {}), ("ANTRE", "hall", {}), ("GIRIS", "hall", {}), ("KORIDOR", "hall", {}),
    ("GECIS", "hall", {}),
    ("KILER", "storage", {}), ("DEPO", "storage", {}), ("CAMASIR", "storage", {}),
    ("GIYINME", "storage", {}), ("VESTIYER", "storage", {}),
    ("ODA", "bedroom", {}),
]
SCALES = {20, 25, 50, 75, 100, 200, 250, 500}


def repair(s):
    """Fix Windows-1254 text read as 1252 and \\U+XXXX escapes from old DWGs."""
    s = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return "".join(MOJIBAKE.get(c, c) for c in s)


def tr_upper(s):
    return s.replace("i", "İ").replace("ı", "I").upper()


def fold(s):
    """Upper-case the Turkish way, strip accents, keep letters/digits/,.²/ and spaces."""
    s = tr_upper(repair(s)).translate(FOLD)
    s = re.sub(r"[^A-Z0-9,./²:\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _fuzzy_in(pattern, text):
    if re.search(r"(^|\s)" + re.escape(pattern) + r"(\s|$)", text):
        return True
    ptoks, ttoks = pattern.split(), text.split()
    if len(pattern) < 4:
        return False
    for i in range(len(ttoks) - len(ptoks) + 1):
        cand = " ".join(ttoks[i:i + len(ptoks)])
        if difflib.SequenceMatcher(None, pattern, cand).ratio() >= 0.8:
            return True
    return False


def room_type(text):
    t = re.sub(r"[0-9,.²/:]", " ", fold(text))
    t = re.sub(r"\s+", " ", t.replace("0", "O")).strip()
    for pat, typ, extra in ROOM_WORDS:
        if _fuzzy_in(pat, t):
            return typ, dict(extra)
    return None, {}


def parse_area(text):
    t = fold(text).replace(" ", "")
    m = re.search(r"(\d{1,3}(?:[.,]\d{1,2})?)M(?:2|²|\^2)", t)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v if 0.5 <= v <= 500 else None


def parse_scale(text):
    t = fold(text)
    m = re.search(r"(?:OLCEK|OLC|SCALE|M)?\s*[:.]?\s*1\s*[/:]\s*(\d{2,3})\b", t)
    if m and int(m.group(1)) in SCALES:
        return int(m.group(1))
    return None


def parse_number(text):
    t = fold(text).replace(" ", "")
    if re.fullmatch(r"\d{1,4}([.,]\d{1,3})?", t):
        return float(t.replace(",", "."))
    return None


def classify(text):
    """Return {'kind': room_label|area|scale|dimension|other, ...} for one text item."""
    out = {"kind": "other"}
    area = parse_area(text)
    sc = parse_scale(text)
    typ, extra = room_type(text)
    if typ:
        out.update(kind="room_label", type=typ, **extra)
        if area:
            out["area"] = area
    elif area:
        out.update(kind="area", value=area)
    elif sc:
        out.update(kind="scale", value=sc)
    elif parse_number(text) is not None:
        out.update(kind="dimension", value=parse_number(text))
    return out
