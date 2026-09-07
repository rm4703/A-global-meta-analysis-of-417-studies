
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv, glob, os, re, random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / str(ROOT / "data/meta_study_table_v2.csv")

PATHS = {}
for d in glob.glob(str(ROOT / "extracted_text*")):
    if os.path.isdir(d):
        for p in glob.glob(d + "/*.txt"):
            PATHS.setdefault(os.path.basename(p)[:-4], p)

VAL_CUE = re.compile(r"valid|test|train|accuracy|assess|cross[- ]?valid|sampl|split|fold|reference|"
                     r"ground|field|confusion|hold[- ]?out", re.I)
EXPLICIT_SPATIAL = re.compile(r"spatial[- ]?(cross[- ]?validation|block|hold[- ]?out)"
                              r"|block(ed)?\s*cross[- ]?validation|sperrorest"
                              r"|spatial(ly)?\s+(independent|disjoint|blocked)\s+(test|validation|sampl|set|fold|cross)"
                              r"|leave[- ]?one[- ](region|site|tile|area)[- ]?out", re.I)
SPATIAL_TRANSFER = re.compile(
    r"(trained|calibrated|developed|built|constructed|derived)[^.]{0,55}\b(one|a|the|first|single)\s+"
    r"(region|site|area|province|tile|scene|location|watershed|county|state)[^.]{0,45}"
    r"(test|validat|appl|evaluat|transfer)\w*[^.]{0,30}(another|other|different|independent|separate|new|second)\s+"
    r"(region|site|area|province|tile|scene|location|county)"
    r"|(tested|validated|evaluated|assessed)[^.]{0,40}(in|on|over|to|across|at)\s+"
    r"(an?\s+|the\s+|several\s+|multiple\s+|newly\s+\w+\s+)?(independent|separate|different|another|new|unseen|other)\s+"
    r"(region|site|area|province|tile|scene|study area|location)"
    r"|independent\s+(test|validation)\s+(site|region|area)", re.I)
MODAL = re.compile(r"\b(may|might|could|should|would|cannot|can\s?not|will\s+not|challeng|difficult|future work|"
                   r"we suggest|we recommend|plan to|propose to|aim to|intend to|"
                   r"not\s+(be\s+)?(suitable|applicable|appropriate|possible)|has the potential|potential to be|"
                   r"hard to|poor(ly)?\s+(transfer|generaliz)|it is necessary|it is recommended|"
                   r"need(s)?\s+to\s+be\s+adjust|parameters?\s+(need|must|should))\b", re.I)
ACC = re.compile(r"\b(overall accuracy|accuracy of|accurac|kappa|\bOA\b|\bF1\b|producer|user.?s? accuracy|"
                 r"RMSE|error (of|rate|was|less)|\d{2}(\.\d+)?\s?%)", re.I)
RANDOM = re.compile(r"random(ly)?[- ]?(split|divid|select|partition|sampl|assign|chosen|drawn)"
                    r"|\b\d{1,2}\s*[:/%]\s*\d{1,2}\b[^.]{0,40}(split|train|test|ratio|partition)"
                    r"|\b(k|\d+)[- ]?fold\b[^.]{0,20}cross[- ]?valid|cross[- ]?validation|hold[- ]?out", re.I)
FIELD = re.compile(r"field (survey|data|sampl|visit|campaign|reference|measurement|observation|plot|investigation|"
                   r"work|trip|gps|point)|ground[- ]?(truth|reference|control|data|sampl|point)|in[- ]?situ"
                   r"|(reference|validation|test|accuracy)[^.]{0,40}(collect|survey|acquir|obtain|gather)[^.]{0,30}"
                   r"(field|ground|gps)", re.I)


def classify(stem):
    p = PATHS.get(stem)
    if not p:
        return "U", ""
    txt = re.sub(r"\s+", " ", open(p, errors="ignore", encoding="utf-8").read())
    sents = re.split(r"(?<=[.!?])\s+", txt)
    vs = [(i, s) for i, s in enumerate(sents) if VAL_CUE.search(s) and 25 < len(s) < 400]
    # 1) spatial: explicit method, OR a performed (non-modal) transfer with reported accuracy in context
    for i, s in vs:
        if EXPLICIT_SPATIAL.search(s):
            return "S", s.strip()[:300]
    for i, s in vs:
        if SPATIAL_TRANSFER.search(s) and not MODAL.search(s):
            ctx = " ".join(sents[max(0, i - 1):i + 2])
            if ACC.search(ctx):
                return "S", s.strip()[:300]
    # 2) random partition (leakage-defining), then 3) field/ground reference, then 4) unspecified
    rnd = next((s for _, s in vs if RANDOM.search(s)), None)
    if rnd:
        return "R", rnd.strip()[:300]
    fld = next((s for _, s in vs if FIELD.search(s)), None)
    if fld:
        return "G", fld.strip()[:300]
    return "U", ""


MAP = {"S": "spatial/independent", "R": "random CV/split", "G": "ground/field reference", "U": ""}


def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def validate_against_gold():
    """Re-coded labels vs the 40-study human gold (reproduces validate_coding.py sample)."""
    files = sorted(glob.glob(str(ROOT / "extracted_text/*.txt"))) + sorted(glob.glob(str(ROOT / "extracted_text_new/*.txt")))
    random.seed(42); sample = random.sample(files, 40)
    gold = ['S','G','G','U','R','G','G','U','G','G','G','R','U','U','G','S','U','U','R','G','G','U','R','U',
            'G','G','R','U','G','G','G','U','R','G','U','U','R','R','U','U']
    rec = [classify(os.path.basename(f)[:-4])[0] for f in sample]
    sb = lambda L: [1 if x == 'S' else 0 for x in L]
    return {"recoded_4way_kappa": round(kappa(rec, gold), 2),
            "recoded_4way_agreement_pct": round(100 * sum(x == y for x, y in zip(rec, gold)) / 40, 1),
            "recoded_spatial_vs_not_kappa": round(kappa(sb(rec), sb(gold)), 2)}


def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        cols = list(csv.DictReader(f).fieldnames); f.seek(0); rows = list(csv.DictReader(f))
    for c in ["validation_auto", "validation_evidence"]:
        if c not in cols:
            cols.insert(cols.index("validation") + 1, c)
    elig = {id(r) for r in rows if r["oa_rep"] not in ("", "None") and r.get("eligible", "1") != "0"}
    changed = 0; codes = Counter(); spatial = []
    for r in rows:
        r.setdefault("validation_auto", r.get("validation", "")); r.setdefault("validation_evidence", "")
        if id(r) not in elig:
            continue
        code, ev = classify(r["file"]); old = r.get("validation", "")
        r["validation_auto"] = old; r["validation"] = MAP[code]; r["validation_evidence"] = ev
        codes[code] += 1
        if r["validation"] != old:
            changed += 1
        if code == "S":
            spatial.append((r["doi"] or r["file"], ev))
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    g = validate_against_gold()
    print(f"re-coded validation for {sum(codes.values())} eligible studies; {changed} changed from the auto label.")
    print(f"  distribution S/R/G/U: {dict(codes)}")
    print(f"  spatially-independent (re-coded, evidence-confirmed): {codes['S']}")
    for doi, ev in spatial:
        print(f"    {doi}: {ev[:130]}")
    print(f"  vs 40-study human gold: 4-way kappa={g['recoded_4way_kappa']} "
          f"(agreement {g['recoded_4way_agreement_pct']}%), spatial-vs-not kappa={g['recoded_spatial_vs_not_kappa']}")
    print("  (compare: original auto-coding 4-way kappa=0.27)")
    import json
    json.dump({"recode_vs_gold": g, "distribution": dict(codes), "n_changed_from_auto": changed,
               "spatial_studies": [d for d, _ in spatial]},
              open(ROOT / str(ROOT / "results/validation_recode_report.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
