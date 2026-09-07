from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv, glob, os, re, json, statistics, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHS = {}
for d in glob.glob(str(ROOT / "extracted_text*")):
    if os.path.isdir(d):
        for p in glob.glob(d + "/*.txt"):
            PATHS.setdefault(os.path.basename(p)[:-4], p)

OA_LABEL = re.compile(r"overall\s+accurac(?:y|ies)|\bOA\b", re.I)
OTHER_ONLY = re.compile(r"^(?=.*(kappa|\bF1\b|producer|user))(?!.*overall)", re.I)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


rows = [r for r in csv.DictReader(open(ROOT / str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8"))
        if r["oa_rep"] not in ("", "None") and r.get("eligible", "1") != "0"]

out = []
for r in rows:
    oa = fnum(r["oa_rep"]); oamax = fnum(r["oa_max"]); snip = (r.get("oa_snippet") or "").strip()
    rec = {"doi": r["doi"], "file": r["file"], "oa_rep": oa, "oa_max": oamax,
           "snippet": snip,
           "metric_anchored": bool(OA_LABEL.search(snip)) and not OTHER_ONLY.search(snip),
           "value_in_snippet": False, "in_range": (oamax is None) or (oa is not None and oa <= oamax + 1e-6)}
    # is the oa_rep value (or a rounding of it) present in its own snippet?
    nums = [float(x) for x in re.findall(r"(\d{2,3}(?:\.\d+)?)\s*%", snip)]
    rec["value_in_snippet"] = any(abs(v - oa) <= 1.0 for v in nums) if (oa is not None and nums) else False
    out.append(rec)

n = len(out)
anchored = sum(r["metric_anchored"] for r in out)
inrange = sum(r["in_range"] for r in out)
val_in_snip = sum(r["value_in_snippet"] for r in out)
# high-confidence subset: metric-anchored AND value present in snippet AND single-value (oa_rep==oa_max)
hi = [r for r in out if r["metric_anchored"] and r["value_in_snippet"]]
allmean = statistics.mean([r["oa_rep"] for r in out])
himean = statistics.mean([r["oa_rep"] for r in hi]) if hi else None

res = {"n_eligible": n,
       "pct_metric_anchored_OA_sentence": round(100 * anchored / n, 1),
       "pct_oa_rep_within_study_OA_range": round(100 * inrange / n, 1),
       "pct_value_matches_its_snippet": round(100 * val_in_snip / n, 1),
       "n_high_confidence": len(hi),
       "pooled_OA_all": round(allmean, 2),
       "pooled_OA_high_confidence": round(himean, 2) if himean else None}
json.dump(res, open(ROOT / str(ROOT / "results/oa_verification.json"), "w", encoding="utf-8"), indent=1)
with open(ROOT / "analysis/oa_verification.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

# random sample with evidence for manual adjudication
random.seed(7); sample = random.sample(out, 50)
with open(ROOT / "analysis/oa_verification_sample.txt", "w", encoding="utf-8") as f:
    for r in sample:
        f.write(f"{r['doi']}\n  oa_rep={r['oa_rep']}  oa_max={r['oa_max']}  anchored={r['metric_anchored']}\n"
                f"  snippet: {r['snippet'][:160]}\n\n")

print(f"eligible studies: {n}")
print(f"  OA value anchored to an explicit Overall-Accuracy sentence: {anchored} ({res['pct_metric_anchored_OA_sentence']}%)")
print(f"  oa_rep value matches a number in its own verbatim snippet:  {val_in_snip} ({res['pct_value_matches_its_snippet']}%)")
print(f"  oa_rep within the study's [min,max] OA range:               {inrange} ({res['pct_oa_rep_within_study_OA_range']}%)")
print(f"  pooled OA: all = {res['pooled_OA_all']}%  vs  high-confidence subset (n={len(hi)}) = {res['pooled_OA_high_confidence']}%")
print("  not metric-anchored (need manual check):")
for r in out:
    if not r["metric_anchored"]:
        print(f"    {r['doi']}: {r['snippet'][:90]}")
