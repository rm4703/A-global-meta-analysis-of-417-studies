from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv
import json
import statistics as st
from pathlib import Path

from scipy import stats as sps

NL = chr(10)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / str(ROOT / "results/kappa_consistency.json")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def kw(pairs):
    g = {}
    for k, _, r in pairs:
        g.setdefault((r.get("method_auto") or "?").strip(), []).append(k)
    g = {a: b for a, b in g.items() if len(b) >= 3}
    H, p = sps.kruskal(*g.values())
    N = sum(len(v) for v in g.values())
    return {"H": round(float(H), 2), "p": round(float(p), 4), "k": len(g), "N": N,
            "eps2": round(float(H) / (N - 1), 3),
            "eps2_null": round((len(g) - 1) / (N - 1), 3)}


def main():
    rows = [r for r in csv.DictReader(open(ROOT / str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8"))
            if str(r.get("eligible", "1")).strip() != "0"]
    pairs = [(num(r["kappa"]), num(r["oa_rep"]), r) for r in rows
             if num(r.get("kappa")) is not None and num(r.get("oa_rep")) is not None]
    good = [(k, o, r) for k, o, r in pairs if k <= o / 100]
    bad = [(k, o, r) for k, o, r in pairs if k > o / 100]

    res = {
        "n_reporting_both": len(pairs),
        "n_internally_consistent": len(good),
        "n_impossible": len(bad),
        "pct_impossible": round(100 * len(bad) / len(pairs), 1),
        "pooled_kappa_all": round(st.mean([k for k, _, _ in pairs]), 3),
        "pooled_kappa_consistent": round(st.mean([k for k, _, _ in good]), 3),
        "method_kw_all": kw(pairs),
        "method_kw_consistent": kw(good),
        "oa_kappa_r_all": round(float(sps.pearsonr([o for _, o, _ in pairs],
                                                   [k * 100 for k, _, _ in pairs]).statistic), 3),
        "oa_kappa_r_consistent": round(float(sps.pearsonr([o for _, o, _ in good],
                                                          [k * 100 for k, _, _ in good]).statistic), 3),
    }
    res["oa_kappa_r2_consistent"] = round(res["oa_kappa_r_consistent"] ** 2, 3)

    print(f"reporting both OA and Kappa : {res['n_reporting_both']}")
    print(f"  impossible (kappa > OA)   : {res['n_impossible']} ({res['pct_impossible']}%)")
    print(f"  pooled Kappa  all / clean : {res['pooled_kappa_all']} / "
          f"{res['pooled_kappa_consistent']}")
    print(f"  method KW     all / clean : p={res['method_kw_all']['p']} / "
          f"p={res['method_kw_consistent']['p']}")
    print(f"  OA-Kappa r    all / clean : {res['oa_kappa_r_all']} / "
          f"{res['oa_kappa_r_consistent']}")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
