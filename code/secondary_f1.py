from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv, json, math, re
from pathlib import Path
from collections import defaultdict
import statistics as st

ROOT = Path(__file__).resolve().parent.parent
def fnum(x):
    try:
        v = float(x); return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None

rows = [r for r in csv.DictReader(open(ROOT / str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8"))
        if r.get("eligible", "1") != "0" and fnum(r["oa_rep"]) and 50 <= fnum(r["oa_rep"]) <= 100]
manifest = json.load(open(ROOT / str(ROOT / "results/text_manifest.json"), encoding="utf-8"))

STRICT = re.compile(
    r"(?:overall|mean|macro|average|weighted|micro)?\s*F[\-\s]?(?:1[\-\s]?)?(?:score|measure)\s*"
    r"(?:value\s*)?(?:of|was|=|:|reached|is|reaches|achiev\w*)?\s*"
    r"(0?\.\d{2,3}|\d{2}(?:\.\d+)?)\s*(%?)(\s*[–\-]\s*\d)?", re.I)
BAD = re.compile(r"(equation|harmonic mean|∑|F1macro|F1i|F1k|F1class|F1 =|=\s*TP|TP\s*\+|"
                 r"\bFoM\b|IMF|i[\-\s]?th|denomin|formula|is defined|defined as|are defined|"
                 r"is the harmonic|Scenario|1st decad|2nd decad|class[\-\s]?wise F1)", re.I)

def strict_f1(text):
    cands = []
    for m in STRICT.finditer(text):
        if m.group(3):                      # number is part of an "NN-NN" range -> skip
            continue
        s = re.sub(r"\s+", " ", text[max(0, m.start()-45): m.end()+25]).strip()
        if BAD.search(s):
            continue
        raw = fnum(m.group(1))
        if raw is None:
            continue
        v = raw*100 if raw <= 1.0 else raw
        if not (50 <= v <= 99.5):
            continue
        cands.append((v, s))
    if not cands:
        return None, None
    cands.sort()
    return cands[len(cands)//2]

kept = []
for r in rows:
    p = manifest.get(r.get("file"))
    if not p or not (ROOT / p).exists():
        continue
    v, s = strict_f1((ROOT / p).read_text(errors="ignore"))
    if v is not None:
        kept.append({"file": r["file"], "doi": r["doi"], "method": r["method_auto"],
                     "scope": r["scope"], "oa": fnum(r["oa_rep"]), "f1": round(v, 2), "snippet": s})

DEEP = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer", "Transfer/Pre-trained"}
vals = [k["f1"] for k in kept]
bym = defaultdict(list)
for k in kept:
    bym[k["method"]].append(k["f1"])

from scipy import stats
groups = [l for l in bym.values() if len(l) >= 3]
H, p = stats.kruskal(*groups)
deep = [k["f1"] for k in kept if k["method"] in DEEP]
clas = [k["f1"] for k in kept if k["method"] not in DEEP]
U, pmw = stats.mannwhitneyu(deep, clas, alternative="two-sided")

summary = {
    "n_studies": len(kept),
    "f1_mean": round(st.mean(vals), 1), "f1_median": round(st.median(vals), 1),
    "f1_min": round(min(vals), 1), "f1_max": round(max(vals), 1),
    "by_method": {m: {"n": len(l), "mean": round(st.mean(l), 1)} for m, l in
                  sorted(bym.items(), key=lambda kv: -len(kv[1]))},
    "kruskal_method_H": round(float(H), 3), "kruskal_method_p": round(float(p), 4),
    "deep_n": len(deep), "deep_mean": round(st.mean(deep), 1),
    "classical_n": len(clas), "classical_mean": round(st.mean(clas), 1),
    "deep_vs_classical_mannwhitney_p": round(float(pmw), 4),
    "note": "Secondary triangulation metric. F1 extracted from full text with verbatim snippets; "
            "values mix averaging conventions (macro/micro/mean/rice-class) and are a coarse "
            "cross-check on the OA result, not a primary synthesis."
}
print(json.dumps(summary, indent=1))

with open(ROOT / str(ROOT / "data/secondary_f1_verified.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "doi", "method", "scope", "oa", "f1", "snippet"])
    w.writeheader(); w.writerows(kept)
json.dump(summary, open(ROOT / str(ROOT / "results/secondary_f1_summary.json"), "w", encoding="utf-8"), indent=1)
print("\nwrote analysis/secondary_f1_verified.csv and secondary_f1_summary.json")
