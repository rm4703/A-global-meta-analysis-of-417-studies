from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv, json, os, re, collections

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

MANIFEST = json.load(open(str(ROOT / "results/text_manifest_v2.json"), encoding="utf-8"))
ROWS = [r for r in csv.DictReader(open(str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8"))
        if r.get("eligible", "1") != "0"]

NUM = r"[0-9]"

# Evaluation cue required to promote a weak match. Built per metric so a metric
# never vouches for itself (the "precision" cue must not be the word "precision").
BASE_CUE = (r"accurac|classifi|confusion\s+matrix|kappa|segmentat|"
            r"true\s+positive|false\s+(positive|negative)|"
            r"evaluat\w*\s+metric|performance\s+metric|\bmetrics\b")
CUES = {
    "F1-score": re.compile(BASE_CUE + r"|\bprecision\b|\brecall\b|\biou\b"),
    "Precision / User's accuracy": re.compile(BASE_CUE + r"|\brecall\b|f1|f\-?scor"),
    "Recall / Producer's accuracy": re.compile(BASE_CUE + r"|\bprecision\b|f1|f\-?scor"),
    "IoU / Jaccard": re.compile(BASE_CUE + r"|\biou\b|f1|f\-?scor"),
}

METRICS = {
    "Overall accuracy (OA)": dict(
        strong=[r"overall\s+accurac", r"overall\s+classification\s+accurac",
                r"\boa\s*[=:]\s*" + NUM, r"\(\s*oa\s*\)", r"\boa\s+(of|was|reached)\s"],
        weak=[], veto=[]),
    "Kappa": dict(
        strong=[r"\bkappa\b", r"\bcohen'?s?\s+kappa", r"\bkc\s*[=:]\s*0?\." + NUM,
                r"κ\s*[=:]\s*0?\."],
        weak=[], veto=[]),
    "F1-score": dict(
        strong=[r"\bf1[\s\-]?scor", r"\bf[\s\-]?measure", r"\bf[\s\-]?scor",
                r"\bf1\s*[=:]\s*0?\." + NUM, r"\bf1\s*[=:]\s*" + NUM],
        weak=[r"\bf1\b"],
        # F1 is also the first filial generation in the breeding literature
        veto=[r"f1\s+(generation|hybrid|progen|cross|population|seed|plant|line)",
              r"(generation|hybrid|progen|cross)\s+f1\b"]),
    "Precision / User's accuracy": dict(
        strong=[r"user'?s?\s+accurac", r"\bua\s*[=:]\s*" + NUM, r"\(\s*ua\s*\)",
                r"precision\s*[=:]\s*0?\." + NUM,
                r"precision\s+(and|,)\s+recall", r"recall\s+(and|,)\s+precision"],
        weak=[r"\bprecision\b"],
        veto=[r"precision\s+(agricultur|farming|irrigation|planting|seeding|"
              r"fertiliz|nutrient|management|sowing|spray|breed|medicine|"
              r"livestock|viticultur|land\s+level|leveling)",
              r"(high|sub\-?(metre|meter|pixel)|positional|geometric|radiometric|"
              r"navigation|gps|gnss|dem|measurement|instrument)[\s\-]precision",
              r"precision\s+of\s+(the\s+)?(dem|gps|gnss|instrument|sensor|"
              r"measurement|coordinate|geolocation)"]),
    "Recall / Producer's accuracy": dict(
        strong=[r"producer'?s?\s+accurac", r"\bpa\s*[=:]\s*" + NUM, r"\(\s*pa\s*\)",
                r"recall\s*[=:]\s*0?\." + NUM,
                r"precision\s+(and|,)\s+recall", r"recall\s+(and|,)\s+precision"],
        weak=[r"\brecall\b"],
        veto=[r"recall\s+(that|from|the\s+(fact|reader|definition|earlier))",
              r"\brecall(ed|ing)\b", r"\btotal\s+recall\b"]),
    "IoU / Jaccard": dict(
        strong=[r"\biou\b", r"\bmiou\b", r"intersection[\s\-]over[\s\-]union"],
        weak=[r"\bjaccard\b", r"\btanimoto\b"],
        veto=[]),
}

WINDOW = 120  # chars either side of a hit, used for veto / cue checking


def normalise(t):
    return re.sub(r"\s+", " ", t.lower())


def hits(text, name, spec):
    """Return (surface, context, tier) for every surviving metric mention."""
    cue = CUES.get(name)
    found = []
    for tier, pats in (("strong", spec["strong"]), ("weak", spec["weak"])):
        for p in pats:
            for mo in re.finditer(p, text):
                a = max(0, mo.start() - WINDOW)
                b = min(len(text), mo.end() + WINDOW)
                ctx = text[a:b]
                if any(re.search(v, ctx) for v in spec["veto"]):
                    continue
                if tier == "weak" and (cue is None or not cue.search(ctx)):
                    continue
                found.append((mo.group(0), ctx, tier))
    return found


# ------------------------------------------------------- full-text scan
per_study = {}
examples = collections.defaultdict(list)
unread = []

for r in ROWS:
    f = r["file"]
    path = MANIFEST.get(f)
    if not path or not os.path.exists(path):
        unread.append(f)
        continue
    try:
        raw = open(path, errors="ignore", encoding="utf-8").read()
    except Exception:
        unread.append(f)
        continue
    text = normalise(raw)
    rec = {}
    for name, spec in METRICS.items():
        h = hits(text, name, spec)
        # "strong" = at least one unambiguous surface form; the conservative count
        rec[name] = "strong" if any(x[2] == "strong" for x in h) else \
                    ("weak" if h else None)
        if h and len(examples[name]) < 6:
            strong = next((x for x in h if x[2] == "strong"), h[0])
            examples[name].append({"doi": r.get("doi", ""), "match": strong[0],
                                   "tier": strong[2],
                                   "context": strong[1].strip()})
    per_study[f] = rec

n_scanned = len(per_study)


def n_notna(col):
    return sum(1 for r in ROWS if (r.get(col) or "").strip() not in ("", "nan", "NA"))

f1_usable = 0
f1_path = str(ROOT / "data/secondary_f1_verified.csv")
if os.path.exists(f1_path):
    f1_usable = sum(1 for _ in csv.DictReader(open(f1_path, encoding="utf-8")))
else:
    # fall back on the extraction JSONs that fed the S8 subset
    seen = set()
    for j in (str(ROOT / "data/extracted_metrics.json"), str(ROOT / "data/extracted_metrics_new.json")):
        if not os.path.exists(j):
            continue
        for d in json.load(open(j, encoding="utf-8")):
            if d.get("f1_max") is not None:
                seen.add(d.get("doi_in_text") or d.get("file"))
    f1_usable = len(seen)


_NUM = r"(?:(\d{1,3}(?:\.\d+)?)\s*%|(0?\.\d{2,4})\b)"
_LEAD = r"\s*(?:score\s*)?(?:of|was|is|=|:|reached|reaching)\s*"
PARSEABLE_PATTERNS = {
    "Precision / User's accuracy": [r"\bprecision" + _LEAD + _NUM,
                                    r"user'?s?\s+accuracy" + _LEAD + _NUM],
    "Recall / Producer's accuracy": [r"\brecall" + _LEAD + _NUM,
                                     r"producer'?s?\s+accuracy" + _LEAD + _NUM],
    "IoU / Jaccard": [r"\bm?iou" + _LEAD + _NUM,
                      r"intersection[\s-]over[\s-]union" + _LEAD + _NUM],
}
parseable = {k: 0 for k in PARSEABLE_PATTERNS}
for r in ROWS:
    path = MANIFEST.get(r["file"])
    if not path or not os.path.exists(path):
        continue
    txt = normalise(open(path, errors="ignore", encoding="utf-8").read())
    for name, pats in PARSEABLE_PATTERNS.items():
        for pat in pats:
            m = re.search(pat, txt)
            if m:
                val = float(m.group(1)) if m.group(1) else float(m.group(2)) * 100
                if 30 <= val <= 100:
                    parseable[name] += 1
                    break

_n_prec = parseable["Precision / User's accuracy"]
_n_rec = parseable["Recall / Producer's accuracy"]
_n_iou = parseable["IoU / Jaccard"]

USABLE = {
    "Overall accuracy (OA)": (
        n_notna("oa_rep"),
        "**Primary outcome.** The only metric every eligible study reports, each "
        "traced to a verbatim source sentence"),
    "Kappa": (
        n_notna("kappa"),
        "**Co-primary outcome (§3.8).** Chance-corrected; the remaining studies "
        "name Kappa without giving an extractable study-level value"),
    "F1-score": (
        f1_usable,
        "**Secondary cross-check only (§S8).** Reported under mixed macro / micro / "
        "rice-class averaging, so the values do not share a scale"),
    "Precision / User's accuracy": (
        0,
        f"A numeric value is parseable in {_n_prec} "
        "studies, but always for one class among several and against target classes "
        "that differ between studies, so none is reducible to a common scale"),
    "Recall / Producer's accuracy": (
        0,
        f"As above: a value is parseable in {_n_rec} "
        "studies, but as a per-class confusion-matrix quantity rather than a "
        "study-level one"),
    "IoU / Jaccard": (
        0,
        f"Segmentation-specific: a value is parseable in {_n_iou} "
        "studies, confined to recent CNN work and far too few to pool"),
}

# ------------------------------------------------------------------ output
out = []
for name in METRICS:
    strong = sum(1 for rec in per_study.values() if rec[name] == "strong")
    rep = sum(1 for rec in per_study.values() if rec[name])
    use, note = USABLE[name]
    out.append({
        "metric": name,
        "studies_mentioning": strong,          # conservative (unambiguous forms)
        "studies_mentioning_incl_weak": rep,   # + context-promoted ambiguous forms
        "pct_mentioning": round(100 * strong / n_scanned, 1),
        "studies_usable": use,
        "pct_usable": round(100 * use / n_scanned, 1),
        # A study can yield a number without that number being poolable. Reporting
        # only "0" in the table read as a contradiction against the note beside it,
        # so the parseable count is now an explicit field.
        "studies_parseable": parseable.get(name, use),
        "note": note,
    })
out.sort(key=lambda d: -d["studies_usable"])

with open(str(ROOT / "data/metric_census.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

json.dump({"n_eligible": len(ROWS), "n_scanned": n_scanned,
           "unmatched_files": unread, "rows": out, "examples": examples},
          open(str(ROOT / "results/metric_census.json"), "w", encoding="utf-8", newline=""), indent=1)

# markdown for the manuscript (Table 9)
md = ["| Metric | Studies naming it | % naming | Studies with a poolable value | % poolable | Status in this synthesis |",
      "|---|---:|---:|---:|---:|---|"]
for d in out:
    # show the parseable count alongside a zero, so the cell cannot contradict its note
    if d["studies_usable"] == 0 and d["studies_parseable"] > 0:
        pool = f"0 ({d['studies_parseable']} parseable)"
    else:
        pool = str(d["studies_usable"])
    md.append(f"| {d['metric']} | {d['studies_mentioning']} | "
              f"{d['pct_mentioning']:.0f}% | {pool} | "
              f"{d['pct_usable']:.0f}% | {d['note']} |")
md = "\n".join(md) + "\n"
tdir = os.path.join("Manuscript_SR", "tables_out")
if os.path.isdir(tdir):
    open(os.path.join(tdir, "Table9_metric_census.md"), "w",
         encoding="utf-8", newline="").write(md)
    print("wrote Manuscript_SR/tables_out/Table9_metric_census.md")

print(f"scanned {n_scanned}/{len(ROWS)} eligible full texts"
      + (f"  ({len(unread)} unreadable)" if unread else ""))
print(f"{'metric':32s} {'strong':>7s} {'+weak':>6s} {'usable':>7s}")
for d in out:
    print(f"{d['metric']:32s} {d['studies_mentioning']:7d} {d['studies_mentioning_incl_weak']:6d} {d['studies_usable']:7d}")
