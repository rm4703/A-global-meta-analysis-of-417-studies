
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEED = 20240117
N_BOOT = 4000

DEEP = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer"}
CLASSICAL = {"Phenology/Threshold", "Random Forest/SVM", "Object-Based"}


def family(row):
    cat = (row.get("method_auto") or "").strip()
    if cat in DEEP:
        return "deep"
    return "classical" if cat in CLASSICAL else None


def main():
    rng = np.random.default_rng(SEED)
    rows = [r for r in csv.DictReader(
                open(ROOT / str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8"))
            if str(r.get("eligible", "1")).strip() != "0"
            and (r.get("oa_rep") or "").strip() not in ("", "None")]
    sub = [(family(r), float(r["oa_rep"]), (r.get("region") or "NA"))
           for r in rows if family(r)]

    deep = [oa for fam, oa, _ in sub if fam == "deep"]
    clas = [oa for fam, oa, _ in sub if fam == "classical"]
    diff = float(np.mean(deep) - np.mean(clas))
    se_naive = float(np.sqrt(np.var(deep, ddof=1) / len(deep)
                             + np.var(clas, ddof=1) / len(clas)))

    # resample whole regions, so studies sharing one travel together
    regions = sorted({g for *_, g in sub})
    by_region = {g: [(f, oa) for f, oa, gg in sub if gg == g] for g in regions}
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(regions), len(regions), replace=True)
        d = [oa for i in pick for f, oa in by_region[regions[i]] if f == "deep"]
        c = [oa for i in pick for f, oa in by_region[regions[i]] if f == "classical"]
        if d and c:
            boot.append(np.mean(d) - np.mean(c))
    se_cluster = float(np.std(boot, ddof=1))

    out = {
        "n_deep": len(deep), "n_classical": len(clas),
        "diff_pp": round(diff, 3),
        "se_naive_pp": round(se_naive, 3),
        "se_cluster_pp": round(se_cluster, 3),
        "se_inflation": round(se_cluster / se_naive, 2),
        "n_region_clusters": len(regions),
        "n_boot": N_BOOT, "seed": SEED,
        "tost_cluster_robust_p": {
            str(m): round(float(stats.norm.sf((m - abs(diff)) / se_cluster)), 4)
            for m in (2, 3, 5)
        },
        "primary_margin_pp": 5,
        "note": ("Pre-specified primary bound is +/-5 pp (main text 2.7). Tighter "
                 "margins are sensitivity analyses and do not hold once region "
                 "clustering is propagated."),
    }
    path = ROOT / "analysis" / str(ROOT / "results/cluster_robust_tost.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.name}")
    print(f"  diff {out['diff_pp']:+} pp | SE {out['se_naive_pp']} -> "
          f"{out['se_cluster_pp']} (x{out['se_inflation']}), "
          f"{out['n_region_clusters']} clusters")
    for m, p in out["tost_cluster_robust_p"].items():
        print(f"  +/-{m} pp: p = {p}{'  <- primary' if m == '5' else ''}")


if __name__ == "__main__":
    main()
