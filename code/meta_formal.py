from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
METHOD_ORDER = ["Phenology/Threshold", "Random Forest/SVM", "Object-Based",
                "CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer",
                "Transfer/Pre-trained"]


def fnum(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


# ---- load -------------------------------------------------------------------
rows = []
with open(ROOT / str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        r["oa"] = fnum(r.get("oa_rep"))
        r["kappa_f"] = fnum(r.get("kappa"))
        r["res"] = fnum(r.get("res_m"))
        r["nval"] = fnum(r.get("n_val"))
        r["year_i"] = int(r["year"]) if str(r.get("year", "")).isdigit() else None
        if r.get("eligible", "1") == "0":          # eligibility re-audit (M2)
            continue
        if r["oa"] is not None and 50 <= r["oa"] <= 100:
            rows.append(r)

oa_rows = rows
methods = [r for r in oa_rows if r["method_auto"] in METHOD_ORDER]


def ci_t(vals):
    a = np.array(vals, float)
    m = a.mean(); sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a)) if len(a) else 0.0
    h = stats.t.ppf(0.975, len(a) - 1) * se if len(a) > 1 else 0.0
    return dict(n=len(a), mean=round(m, 3), sd=round(sd, 3),
                ci_low=round(m - h, 3), ci_high=round(m + h, 3),
                min=round(a.min(), 2), max=round(a.max(), 2))


# ============================================================ 1. DESCRIPTIVE
desc = {"all": ci_t([r["oa"] for r in oa_rows]), "by_method": {}}
groups = defaultdict(list)
for r in methods:
    groups[r["method_auto"]].append(r["oa"])
for m in METHOD_ORDER:
    if groups[m]:
        desc["by_method"][m] = ci_t(groups[m])
# Kruskal-Wallis across methods  (Eq. 12)
present = [groups[m] for m in METHOD_ORDER if len(groups[m]) >= 2]
H, p = stats.kruskal(*present)
desc["kruskal"] = {"H": round(H, 3), "p": round(p, 4), "k_groups": len(present)}
# 95% prediction interval on the % scale (assumption-free heterogeneity)  (Eq. 9)
a = np.array([r["oa"] for r in oa_rows], float)
k = len(a); sd = a.std(ddof=1)
pi = stats.t.ppf(0.975, k - 1) * sd * math.sqrt(1 + 1 / k)
desc["prediction_interval"] = {"low": round(a.mean() - pi, 2), "high": round(min(100, a.mean() + pi), 2),
                               "between_study_sd": round(sd, 3)}
# temporal trend (OLS on % vs year)
yr = [(r["year_i"], r["oa"]) for r in oa_rows if r["year_i"]]
X = sm.add_constant(np.array([y for y, _ in yr], float))
mt = sm.OLS(np.array([o for _, o in yr], float), X).fit()
desc["temporal"] = {"n": len(yr), "beta_per_yr": round(mt.params[1], 4), "p": round(mt.pvalues[1], 4)}


# ============================================================ 2. INVERSE-VARIANCE RE
def ft_transform(p, n):
    """Freeman-Tukey double-arcsine of a proportion.  (Eq. 1, 2)"""
    x = p * n
    t = math.asin(math.sqrt(x / (n + 1))) + math.asin(math.sqrt((x + 1) / (n + 1)))
    v = 1.0 / (n + 0.5)
    return t, v


def ft_backtransform(t, nbar):
    """Miller back-transform of a pooled FT mean to a proportion.  (Eq. 11)"""
    s = math.sin(t / 2) ** 2  # close approximation for the pooled mean
    # Barendregt et al. (2013) harmonic-mean back-transform:
    val = 0.5 * (1 - np.sign(math.cos(t)) *
                 math.sqrt(1 - (math.sin(t) + (math.sin(t) - 1.0 / math.sin(t)) / nbar) ** 2))
    if not (0 <= val <= 1) or math.isnan(val):
        val = s
    return val


def re_meta(subset):
    """DerSimonian-Laird random-effects on FT scale. Returns dict + per-study."""
    per = []
    for r in subset:
        p = r["oa"] / 100.0
        n = int(r["nval"])
        t, v = ft_transform(min(p, 0.99999), n)
        per.append({"file": r["file"], "doi": r["doi"], "method": r["method_auto"],
                    "year": r["year_i"], "oa": r["oa"], "n": n,
                    "t": t, "v": v, "se": math.sqrt(v), "w": 1.0 / v})
    t_arr = np.array([d["t"] for d in per]); v_arr = np.array([d["v"] for d in per])
    w = 1.0 / v_arr
    t_fe = np.sum(w * t_arr) / np.sum(w)                          # fixed-effect mean
    Q = float(np.sum(w * (t_arr - t_fe) ** 2))                    # Eq. 4
    kk = len(per); df = kk - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0              # Eq. 6
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0          # Eq. 5
    H2 = Q / df if df > 0 else float("nan")
    wst = 1.0 / (v_arr + tau2)                                   # Eq. 7
    t_re = float(np.sum(wst * t_arr) / np.sum(wst))
    var_re = 1.0 / np.sum(wst)
    se_re = math.sqrt(var_re)
    ci = (t_re - 1.96 * se_re, t_re + 1.96 * se_re)              # Eq. 8
    tcrit = stats.t.ppf(0.975, df - 1) if df > 1 else 1.96
    pi = (t_re - tcrit * math.sqrt(tau2 + var_re), t_re + tcrit * math.sqrt(tau2 + var_re))  # Eq. 9
    nbar = stats.hmean([d["n"] for d in per])
    def bt(x):  # back-transform to %
        return round(100 * ft_backtransform(x, nbar), 2)
    # Egger's regression test (SND ~ precision)   (Eq. 10)
    snd = t_arr / np.sqrt(v_arr)
    prec = 1.0 / np.sqrt(v_arr)
    eg = sm.OLS(snd, sm.add_constant(prec)).fit()
    egger = {"intercept": round(eg.params[0], 3), "se": round(eg.bse[0], 3),
             "t": round(eg.tvalues[0], 3), "p": round(eg.pvalues[0], 4)}
    out = {"k": kk, "pooled_ft": round(t_re, 4), "pooled_pct": bt(t_re),
           "ci_pct": [bt(ci[0]), bt(ci[1])], "Q": round(Q, 2), "df": df,
           "I2": round(I2, 1), "tau2": round(tau2, 5), "H2": round(H2, 2),
           "pred_int_pct": [bt(pi[0]), bt(pi[1])], "egger": egger,
           "harmonic_n": round(float(nbar), 1)}
    return out, per, (t_re, tau2, v_arr, t_arr, nbar)


realN = [r for r in oa_rows if r["nval"] and 30 <= r["nval"] <= 5_000_000]
iv_primary, per_study, core = re_meta(realN)

# sensitivity: sample-based assessment only (exclude pixel-census N > 10,000)
realN_small = [r for r in realN if r["nval"] <= 10000]
iv_sens, _, _ = re_meta(realN_small)

# leave-one-out (recompute pooled %, on the primary subset)
loo = []
for i in range(len(realN)):
    sub = realN[:i] + realN[i + 1:]
    res, _, _ = re_meta(sub)
    loo.append({"dropped": realN[i]["file"], "pooled_pct": res["pooled_pct"]})
loo_vals = [d["pooled_pct"] for d in loo]
loo_summary = {"min": min(loo_vals), "max": max(loo_vals),
               "most_influential": max(loo, key=lambda d: abs(d["pooled_pct"] - iv_primary["pooled_pct"]))["dropped"]}

# Baujat: x = contribution to Q, y = influence on pooled estimate
t_re0, tau2_0, v_arr0, t_arr0, nbar0 = core
w0 = 1.0 / v_arr0
t_fe0 = np.sum(w0 * t_arr0) / np.sum(w0)
baujat = []
for d, ti, vi in zip(per_study, t_arr0, v_arr0):
    qx = (1.0 / vi) * (ti - t_fe0) ** 2
    # influence: squared standardized change in pooled estimate when omitted
    wst = 1.0 / (v_arr0 + tau2_0)
    mask = t_arr0 != ti
    t_re_i = np.sum((wst * t_arr0)[mask]) / np.sum(wst[mask])
    yinf = (t_re0 - t_re_i) ** 2 / (1.0 / np.sum(wst))
    baujat.append({"file": d["file"], "method": d["method"], "x": round(float(qx), 3),
                   "y": round(float(yinf), 4)})

# subgroup synthesis by method (real-N), Q_between
sub_by = {}
for m in METHOD_ORDER:
    g = [r for r in realN if r["method_auto"] == m]
    if len(g) >= 2:
        res, _, _ = re_meta(g)
        sub_by[m] = {"k": res["k"], "pooled_pct": res["pooled_pct"], "ci_pct": res["ci_pct"]}
# Q_between on FT scale using RE group means weighted by inverse group variance
gmeans = []
for m, s in sub_by.items():
    g = [r for r in realN if r["method_auto"] == m]
    _, _, c = re_meta(g)
    t_re_g, tau2_g, v_g, t_g, _ = c
    wst_g = 1.0 / (v_g + tau2_g)
    gmeans.append((t_re_g, np.sum(wst_g)))
if len(gmeans) >= 2:
    gm = np.array([x for x, _ in gmeans]); gw = np.array([w for _, w in gmeans])
    grand = np.sum(gw * gm) / np.sum(gw)
    Qb = float(np.sum(gw * (gm - grand) ** 2))
    dfb = len(gmeans) - 1
    subgroup = {"by_method": sub_by, "Q_between": round(Qb, 2), "df": dfb,
                "p": round(1 - stats.chi2.cdf(Qb, dfb), 4)}
else:
    subgroup = {"by_method": sub_by}

# ============================================================ 3. META-REGRESSION
# full-corpus OLS: OA ~ log10(resolution) + year
mr = [(math.log10(r["res"]), r["year_i"], r["oa"]) for r in oa_rows
      if r["res"] and r["res"] > 0 and r["year_i"]]
Xr = sm.add_constant(np.array([[lr, yr] for lr, yr, _ in mr], float))
yv = np.array([o for *_, o in mr], float)
mreg = sm.OLS(yv, Xr).fit()
ols_full = {"n": len(mr), "r2": round(mreg.rsquared, 4),
            "logres_beta": round(mreg.params[1], 3), "logres_p": round(mreg.pvalues[1], 4),
            "year_beta": round(mreg.params[2], 4), "year_p": round(mreg.pvalues[2], 4)}

# weighted (RE) meta-regression on FT scale over the real-N subset
mrn = [r for r in realN if r["res"] and r["res"] > 0 and r["year_i"]]
if len(mrn) >= 8:
    ts, vs, lrs, yrs = [], [], [], []
    for r in mrn:
        t, v = ft_transform(min(r["oa"] / 100, 0.99999), int(r["nval"]))
        ts.append(t); vs.append(v); lrs.append(math.log10(r["res"])); yrs.append(r["year_i"])
    vs = np.array(vs)
    Xw = sm.add_constant(np.column_stack([lrs, yrs]))
    wls = sm.WLS(np.array(ts), Xw, weights=1.0 / (vs + iv_primary["tau2"])).fit()
    wls_realN = {"n": len(mrn), "logres_beta_ft": round(wls.params[1], 4), "logres_p": round(wls.pvalues[1], 4),
                 "year_beta_ft": round(wls.params[2], 5), "year_p": round(wls.pvalues[2], 4)}
else:
    wls_realN = {"n": len(mrn), "note": "too few real-N studies for weighted meta-regression"}

# ============================================================ KAPPA (descriptive + by-method)
kap = [r["kappa_f"] for r in oa_rows if r["kappa_f"] is not None]
kappa = ci_t(kap) if kap else None
# Kappa is chance-corrected, so testing it by method is a second-metric robustness
# check on the no-method-effect result (main text §4.9).
DEEP_SET = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer", "Transfer/Pre-trained"}
kap_by = defaultdict(list)
for r in oa_rows:
    if r["kappa_f"] is not None and 0 < r["kappa_f"] <= 1 and r["method_auto"] in METHOD_ORDER:
        kap_by[r["method_auto"]].append(r["kappa_f"])
if kappa:
    kappa["by_method"] = {m: dict(n=len(kap_by[m]), mean=round(float(np.mean(kap_by[m])), 3))
                          for m in METHOD_ORDER if kap_by[m]}
    kap_present = [kap_by[m] for m in METHOD_ORDER if len(kap_by[m]) >= 3]
    Hk, pk = stats.kruskal(*kap_present)
    kappa["kruskal_method"] = {"H": round(float(Hk), 3), "p": round(float(pk), 4), "k_groups": len(kap_present)}
    dk = [v for m in kap_by for v in kap_by[m] if m in DEEP_SET]
    ck = [v for m in kap_by for v in kap_by[m] if m not in DEEP_SET]
    _, pk_dc = stats.mannwhitneyu(dk, ck, alternative="two-sided")
    kappa["deep_vs_classical"] = {"deep_n": len(dk), "deep_mean": round(float(np.mean(dk)), 3),
                                  "classical_n": len(ck), "classical_mean": round(float(np.mean(ck)), 3),
                                  "mannwhitney_p": round(float(pk_dc), 4)}

# ============================================================ WRITE
out = {
    "n_oa": len(oa_rows), "n_realN": len(realN), "n_kappa": len(kap),
    "descriptive": desc,
    "kappa": kappa,
    "inverse_variance": {
        "primary": iv_primary,
        "sensitivity_N_le_10000": iv_sens,
        "leave_one_out": loo_summary,
        "subgroup": subgroup,
        "per_study": per_study,
        "baujat": baujat,
    },
    "meta_regression": {"ols_full": ols_full, "wls_realN": wls_realN},
}
(ROOT / "analysis").mkdir(exist_ok=True)
with open(ROOT / str(ROOT / "results/meta_formal.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=float)

# ---- console summary ----
print(f"OA studies (descriptive synthesis): {len(oa_rows)}")
print(f"  pooled mean OA = {desc['all']['mean']}%  (95% CI {desc['all']['ci_low']}-{desc['all']['ci_high']})")
print(f"  95% prediction interval: {desc['prediction_interval']['low']}-{desc['prediction_interval']['high']}%")
print(f"  method effect (Kruskal-Wallis): H={desc['kruskal']['H']}, p={desc['kruskal']['p']}")
print(f"  temporal: beta={desc['temporal']['beta_per_yr']} %/yr, p={desc['temporal']['p']}")
print(f"\nInverse-variance RE (real-N subset, k={iv_primary['k']}):")
print(f"  pooled OA = {iv_primary['pooled_pct']}%  (95% CI {iv_primary['ci_pct']})")
print(f"  Q={iv_primary['Q']} (df={iv_primary['df']}), I2={iv_primary['I2']}%, tau2={iv_primary['tau2']}, H2={iv_primary['H2']}")
print(f"  95% prediction interval: {iv_primary['pred_int_pct']}")
print(f"  Egger intercept={iv_primary['egger']['intercept']} (p={iv_primary['egger']['p']})")
print(f"  sensitivity (N<=10000, k={iv_sens['k']}): pooled={iv_sens['pooled_pct']}%, I2={iv_sens['I2']}%")
print(f"  leave-one-out pooled range: {loo_summary['min']}-{loo_summary['max']}%")
print(f"  subgroup Q_between p = {subgroup.get('p')}")
print(f"\nMeta-regression (full OLS): logres beta={ols_full['logres_beta']} (p={ols_full['logres_p']}), "
      f"year beta={ols_full['year_beta']} (p={ols_full['year_p']})")
print("Wrote analysis/meta_formal.json")
