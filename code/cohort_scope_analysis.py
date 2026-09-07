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
CSV = ROOT / str(ROOT / "data/meta_study_table_v2.csv")
OUT = ROOT / "analysis" / str(ROOT / "results/cohort_scope.json")

DEEP = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer"}
CLASSICAL = {"Phenology/Threshold", "Random Forest/SVM", "Object-Based"}
ERAS = [("2000-2010", 2000, 2010), ("2011-2015", 2011, 2015),
        ("2016-2020", 2016, 2020), ("2021-2026", 2021, 2026)]


def load():
    rows = []
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if str(r.get("eligible", "1")).strip() == "0":
            continue
        try:
            oa = float(r["oa_rep"])
        except (ValueError, TypeError, KeyError):
            continue
        try:
            yr = int(str(r.get("year") or "").strip()[:4])
        except ValueError:
            yr = None
        rows.append({"oa": oa, "year": yr, "method": (r.get("method_auto") or "").strip(),
                     "scope": (r.get("scope") or "").strip()})
    return rows


def rank_biserial(a, b):
    """Mann-Whitney U expressed as a rank-biserial correlation."""
    if not a or not b:
        return None
    u = sps.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def contrast(rows):
    """Deep vs classical inside a subset."""
    d = [r["oa"] for r in rows if r["method"] in DEEP]
    c = [r["oa"] for r in rows if r["method"] in CLASSICAL]
    out = {"n_deep": len(d), "n_classical": len(c)}
    if len(d) >= 3 and len(c) >= 3:
        out["mean_deep"] = round(st.mean(d), 2)
        out["mean_classical"] = round(st.mean(c), 2)
        out["diff_pp"] = round(st.mean(d) - st.mean(c), 2)
        out["mannwhitney_p"] = round(float(sps.mannwhitneyu(d, c).pvalue), 4)
        out["r_rb"] = round(rank_biserial(d, c), 3)
    else:
        out["note"] = "too few studies in one arm for a test (min 3 per arm)"
    return out


def kw_methods(rows, min_n=3):
    """Kruskal-Wallis across method categories, dropping categories below min_n."""
    g = {}
    for r in rows:
        if r["method"]:
            g.setdefault(r["method"], []).append(r["oa"])
    g = {k: v for k, v in g.items() if len(v) >= min_n}
    if len(g) < 2:
        return {"note": "fewer than two usable categories"}
    H, p = sps.kruskal(*g.values())
    N = sum(len(v) for v in g.values())
    return {"H": round(float(H), 3), "p": round(float(p), 4), "k": len(g), "N": N,
            "eps2": round(float(H) / (N - 1), 3),
            "eps2_null": round((len(g) - 1) / (N - 1), 3)}


def describe(rows):
    oa = [r["oa"] for r in rows]
    return {"n": len(rows), "mean_oa": round(st.mean(oa), 2) if oa else None,
            "sd": round(st.pstdev(oa), 2) if len(oa) > 1 else None}


def main():
    rows = load()
    res = {"n_total": len(rows), "deep_definition": sorted(DEEP),
           "classical_definition": sorted(CLASSICAL)}

    # ---------------- Major 5: temporal cohorts -------------------------
    dated = [r for r in rows if r["year"]]
    res["n_dated"] = len(dated)
    eras = {}
    for lbl, lo, hi in ERAS:
        sub = [r for r in dated if lo <= r["year"] <= hi]
        e = describe(sub)
        e["deep_share_pct"] = (round(100 * sum(1 for r in sub if r["method"] in DEEP)
                                     / len(sub), 1) if sub else None)
        e["method_kruskal"] = kw_methods(sub)
        e["deep_vs_classical"] = contrast(sub)
        eras[lbl] = e
    res["eras"] = eras

    # does the deep-classical gap itself move across eras?
    gaps = [(lbl, e["deep_vs_classical"].get("diff_pp"))
            for lbl, e in eras.items() if e["deep_vs_classical"].get("diff_pp") is not None]
    res["era_gaps_pp"] = dict(gaps)
    testable = [lbl for lbl, _ in gaps]
    res["eras_testable"] = testable
    res["eras_untestable"] = [lbl for lbl, _, _ in ERAS if lbl not in testable]

    # year as a continuous covariate within the deep and classical arms separately
    for arm, names in (("deep", DEEP), ("classical", CLASSICAL)):
        sub = [r for r in dated if r["method"] in names]
        if len(sub) >= 10:
            sl, ic, rv, pv, se = sps.linregress([r["year"] for r in sub],
                                                [r["oa"] for r in sub])
            res[f"{arm}_year_trend"] = {"n": len(sub), "slope_pp_per_yr": round(sl, 4),
                                        "p": round(float(pv), 4)}

    # ---------------- Major 4: scope strata -----------------------------
    strata = {}
    for lbl, key in (("rice-only", "rice"), ("crop-only", "crop-classification")):
        sub = [r for r in rows if r["scope"] == key]
        s = describe(sub)
        s["method_kruskal"] = kw_methods(sub)
        s["deep_vs_classical"] = contrast(sub)
        # cohorts inside each stratum: the two confounders together
        s["by_era"] = {}
        for elbl, lo, hi in ERAS:
            es = [r for r in sub if r["year"] and lo <= r["year"] <= hi]
            if es:
                s["by_era"][elbl] = {**describe(es), "deep_vs_classical": contrast(es)}
        strata[lbl] = s
    res["scope_strata"] = strata

    rice = [r["oa"] for r in rows if r["scope"] == "rice"]
    crop = [r["oa"] for r in rows if r["scope"] == "crop-classification"]
    res["scope_contrast"] = {
        "mannwhitney_p": round(float(sps.mannwhitneyu(rice, crop).pvalue), 5),
        "r_rb": round(rank_biserial(rice, crop), 3),
        "mean_diff_pp": round(st.mean(rice) - st.mean(crop), 2)}

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---------------- readable summary ----------------------------------
    print(f"corpus {res['n_total']}  dated {res['n_dated']}{NL}")
    print("MAJOR 5 - within-era architecture contrast")
    print(f"  {'era':<11}{'n':>5}{'mean OA':>9}{'deep%':>7}{'n deep':>8}"
          f"{'n clas':>8}{'diff pp':>9}{'MW p':>8}")
    for lbl, e in eras.items():
        d = e["deep_vs_classical"]
        print(f"  {lbl:<11}{e['n']:>5}{e['mean_oa'] or 0:>9.1f}{e['deep_share_pct'] or 0:>7.1f}"
              f"{d['n_deep']:>8}{d['n_classical']:>8}"
              f"{d.get('diff_pp', float('nan')):>9}{d.get('mannwhitney_p', float('nan')):>8}")
    print(f"  testable eras: {', '.join(testable) or 'none'}")
    print(f"  untestable   : {', '.join(res['eras_untestable']) or 'none'}")
    for arm in ("deep", "classical"):
        k = f"{arm}_year_trend"
        if k in res:
            print(f"  {arm:<10} year trend {res[k]['slope_pp_per_yr']:+.3f} pp/yr "
                  f"(p = {res[k]['p']}, n = {res[k]['n']})")

    print(f"{NL}MAJOR 4 - scope strata")
    for lbl, s in strata.items():
        d, kwv = s["deep_vs_classical"], s["method_kruskal"]
        print(f"  {lbl:<11} n={s['n']:<5} mean {s['mean_oa']:.1f}  "
              f"KW p={kwv.get('p')}  eps2={kwv.get('eps2')} (null {kwv.get('eps2_null')})")
        print(f"              deep {d['n_deep']} vs classical {d['n_classical']}  "
              f"diff {d.get('diff_pp')} pp  MW p={d.get('mannwhitney_p')}")
    sc = res["scope_contrast"]
    print(f"  rice vs crop: {sc['mean_diff_pp']:+} pp, MW p={sc['mannwhitney_p']}, "
          f"r_rb={sc['r_rb']}")
    print(f"{NL}wrote {OUT}")


if __name__ == "__main__":
    main()
