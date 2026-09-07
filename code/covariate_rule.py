from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / str(ROOT / "data/meta_study_table_v2.csv")

DEEP = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer"}
SENSORS = {"Sentinel-2", "Landsat", "Sentinel-1", "MODIS"}
UNASSIGNED_REGION = ("", "?")

EXPECTED = {"eligible": 420, "subset": 315, "clusters": 25,
            "full_corpus_clusters": 26,
            "dropped_year": 15, "dropped_method": 7, "dropped_res": 83}


def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def region_of(row):
    r = (row.get("region") or "").strip()
    return "unassigned" if r in UNASSIGNED_REGION else r


def eligible_rows(path=CSV):
    """Note the filter is eligible != "0", so a blank counts as eligible."""
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)
                if str(r.get("eligible", "1")).strip() != "0"]


def build_subset(rows):
    """Apply the four dropping conditions. Returns (records, drop_counts)."""
    recs, drops = [], {"oa": 0, "year": 0, "method": 0, "res": 0}
    for r in rows:
        oa, yr, res = num(r.get("oa_rep")), num(r.get("year")), num(r.get("res_m"))
        method = (r.get("method_auto") or "").strip()
        if oa is None:
            drops["oa"] += 1; continue
        if yr is None:
            drops["year"] += 1; continue
        if not method or method == "Unclassified":
            drops["method"] += 1; continue
        if not res or res <= 0:
            drops["res"] += 1; continue
        sensor = (r.get("sensor") or "other").strip() or "other"
        val = (r.get("validation") or "").strip().lower()
        recs.append(dict(
            oa=oa,
            year=yr - 2013,                                   # centred
            deep=int(method in DEEP),
            sensor=sensor if sensor in SENSORS else "other",
            logres=math.log10(res),
            scope="rice" if (r.get("scope") or "").strip() == "rice" else "crop",
            val=("spatial" if "spatial" in val else
                 "field" if ("ground" in val or "field" in val) else "other"),
            region=region_of(r),
        ))
    return recs, drops


def main():
    rows = eligible_rows()
    recs, drops = build_subset(rows)
    clusters = {r["region"] for r in recs}
    full_clusters = {region_of(r) for r in rows}

    checks = [
        ("eligible studies", len(rows), EXPECTED["eligible"]),
        ("dropped: missing year", drops["year"], EXPECTED["dropped_year"]),
        ("dropped: no method", drops["method"], EXPECTED["dropped_method"]),
        ("dropped: no resolution", drops["res"], EXPECTED["dropped_res"]),
        ("meta-regression subset", len(recs), EXPECTED["subset"]),
        ("meta-regression clusters", len(clusters), EXPECTED["clusters"]),
        ("full-corpus clusters", len(full_clusters), EXPECTED["full_corpus_clusters"]),
    ]
    ok = True
    for label, got, want in checks:
        flag = "OK" if got == want else "MISMATCH"
        ok &= got == want
        print(f"  {label:<26} {got:>4}   expected {want:>4}   {flag}")
    print("\nAll checks passed." if ok else "\nFAILED: the rule no longer reproduces the paper.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
