from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv, json, math
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
def fn(x):
    try:
        v = float(x); return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None
rows = [r for r in csv.DictReader(open(ROOT / str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8"))
        if str(r.get("eligible", "1")).strip() != "0"]
DEEP = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer"}
m_binary, m_coarse = 0.08, 0.275

def synth(values_by_row, colname, eq_margin, label):
    """Full descriptive synthesis for a metric column."""
    vals = np.array([v for v in values_by_row if v is not None])
    n = len(vals); mean = vals.mean(); sd = vals.std(ddof=1)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - 1.96*sd/math.sqrt(n), mean + 1.96*sd/math.sqrt(n))
    pi = (mean - tcrit*sd*math.sqrt(1+1/n), mean + tcrit*sd*math.sqrt(1+1/n))
    # by method
    groups = {}
    for r in rows:
        v = fn(r.get(colname))
        if v is None: continue
        m = r["method_auto"]
        if m and m not in ("", "Unclassified"): groups.setdefault(m, []).append(v)
    big = [g for g in groups.values() if len(g) >= 3]
    H, kw_p = stats.kruskal(*big) if len(big) >= 2 else (float("nan"), float("nan"))
    # deep vs classical
    deep = [fn(r.get(colname)) for r in rows if r["method_auto"] in DEEP and fn(r.get(colname)) is not None]
    clas = [fn(r.get(colname)) for r in rows if r["method_auto"] and r["method_auto"] not in DEEP
            and r["method_auto"] not in ("", "Unclassified") and fn(r.get(colname)) is not None]
    diff = np.mean(deep) - np.mean(clas)
    mw = float(stats.mannwhitneyu(deep, clas, alternative="two-sided").pvalue)
    # TOST equivalence within eq_margin
    try:
        from statsmodels.stats.weightstats import ttost_ind
        tost_p = float(ttost_ind(deep, clas, -eq_margin, eq_margin, usevar="unequal")[0])
    except Exception:
        tost_p = None
    return {
        "n": n, "pooled_mean": round(float(mean), 3), "sd": round(float(sd), 3),
        "ci95": [round(ci[0], 3), round(ci[1], 3)], "pi95": [round(pi[0], 3), round(pi[1], 3)],
        "kruskal_method_H": round(float(H), 2), "kruskal_method_p": round(float(kw_p), 3),
        "deep_mean": round(float(np.mean(deep)), 3), "n_deep": len(deep),
        "classical_mean": round(float(np.mean(clas)), 3), "n_classical": len(clas),
        "deep_minus_classical": round(float(diff), 3),
        "deattenuated": [round(diff/(1-2*m_binary), 3), round(diff/(1-2*m_coarse), 3)],
        "mannwhitney_p": round(mw, 3), "tost_p_within_margin": (round(tost_p, 3) if tost_p is not None else None),
        "eq_margin": eq_margin,
    }

out = {}
out["OA"] = synth([fn(r.get("oa_rep")) for r in rows], "oa_rep", 3.0, "OA")
out["Kappa"] = synth([fn(r.get("kappa")) for r in rows], "kappa", 0.03, "Kappa")
# F1 from released summary (already provenance-traced); add pooled/PI from the verified CSV
try:
    f1rows = list(csv.DictReader(open(ROOT / str(ROOT / "data/secondary_f1_verified.csv"), encoding="utf-8")))
    f1 = np.array([fn(r["f1"]) for r in f1rows if fn(r["f1"]) is not None])
    n = len(f1); tcrit = stats.t.ppf(0.975, n-1); mean=f1.mean(); sd=f1.std(ddof=1)
    summ = json.load(open(ROOT / str(ROOT / "results/secondary_f1_summary.json"), encoding="utf-8"))
    out["F1"] = {"n": n, "pooled_mean": round(float(mean),2), "median": summ["f1_median"], "sd": round(float(sd),2),
                 "pi95": [round(mean-tcrit*sd*math.sqrt(1+1/n),2), round(min(mean+tcrit*sd*math.sqrt(1+1/n),100),2)],
                 "kruskal_method_p": summ["kruskal_method_p"],
                 "deep_mean": summ["deep_mean"], "classical_mean": summ["classical_mean"],
                 "n_deep": summ["deep_n"], "n_classical": summ["classical_n"],
                 "deep_minus_classical": round(summ["deep_mean"]-summ["classical_mean"],2),
                 "mannwhitney_p": summ["deep_vs_classical_mannwhitney_p"]}
except Exception as e:
    out["F1"] = {"error": str(e)[:120]}

json.dump(out, open(ROOT / str(ROOT / "results/reviewer_metrics.json"), "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
