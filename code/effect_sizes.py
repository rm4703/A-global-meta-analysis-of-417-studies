from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv, json, math
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

DEEP = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer"}
BOOT = 10000
RNG = np.random.default_rng(20260728)


def fn(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


rows = [r for r in csv.DictReader(open(ROOT / str(ROOT / "data/meta_study_table_v2.csv"), encoding="utf-8"))
        if str(r.get("eligible", "1")).strip() != "0"]


# ------------------------------------------------------------ effect sizes
def _eps2(big):
    n = int(sum(len(g) for g in big))
    H = stats.kruskal(*big).statistic
    return H / (n - 1)


def kw_effect(groups, min_size=3):
    """Kruskal-Wallis with epsilon-squared and eta-squared, plus a bootstrap CI.

    min_size mirrors the grouping convention of the analysis being reproduced:
    reviewer_metrics.py drops method categories with <3 studies, but the
    N-reporting subset test in §3.6 was run over all categories (min_size=1).

    eps2 is upward-biased when k is large relative to n (its expectation under the
    null is roughly (k-1)/(n-1), not 0), so both the null expectation and a
    stratified-bootstrap CI are reported: without them a subset like F1 (n=32,
    k=8) shows a 'large' point estimate that is mostly bias.
    """
    big = [np.asarray(g, float) for g in groups if len(g) >= min_size]
    n = int(sum(len(g) for g in big))
    k = len(big)
    H, p = stats.kruskal(*big)
    eps2 = H / (n - 1)
    eta2 = (H - k + 1) / (n - k)
    boot = []
    for _ in range(BOOT):
        res = [RNG.choice(g, size=len(g), replace=True) for g in big]
        res = [g for g in res if len(np.unique(g)) > 1]
        if len(res) >= 2:
            try:
                boot.append(_eps2(res))
            except ValueError:
                pass
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2
    return dict(test="Kruskal-Wallis", n=n, k=k, H=round(float(H), 3),
                p=float(p), eps2=round(float(eps2), 4),
                eps2_ci95=[round(float(lo), 4), round(float(hi), 4)],
                eps2_null_expectation=round((k - 1) / (n - 1), 4),
                eta2_H=round(float(max(eta2, 0.0)), 4))


def mw_effect(a, b):
    """Mann-Whitney U with rank-biserial correlation (= Cliff's delta)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    U1 = stats.mannwhitneyu(a, b, alternative="two-sided")
    r_rb = 2 * U1.statistic / (len(a) * len(b)) - 1
    return dict(test="Mann-Whitney U", n1=len(a), n2=len(b),
                U=float(U1.statistic), p=float(U1.pvalue),
                rank_biserial=round(float(r_rb), 3),
                cliffs_delta=round(float(r_rb), 3))


def wilcoxon_effect(d):
    """Wilcoxon signed-rank with matched-pairs rank-biserial correlation."""
    d = np.asarray([x for x in d if x != 0], float)
    res = stats.wilcoxon(d, alternative="two-sided")
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    r_rb = (w_pos - w_neg) / (w_pos + w_neg)
    # bootstrap CI on the matched-pairs rank-biserial
    boot = []
    for _ in range(BOOT):
        s = RNG.choice(d, size=len(d), replace=True)
        rk = stats.rankdata(np.abs(s))
        pos, neg = rk[s > 0].sum(), rk[s < 0].sum()
        if pos + neg:
            boot.append((pos - neg) / (pos + neg))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(test="Wilcoxon signed-rank", n_pairs=len(d),
                W=float(res.statistic), p=float(res.pvalue),
                rank_biserial=round(float(r_rb), 3),
                rank_biserial_ci95=[round(float(lo), 3), round(float(hi), 3)])


def magnitude(v, kind):
    """Conventional verbal anchors, stated so the table is self-interpreting."""
    a = abs(v)
    if kind == "variance":                      # eps2 / eta2
        return "negligible" if a < 0.01 else "small" if a < 0.06 else \
               "moderate" if a < 0.14 else "large"
    return "negligible" if a < 0.147 else "small" if a < 0.33 else \
           "medium" if a < 0.474 else "large"   # Cliff's delta / rank-biserial


# -------------------------------------------------------------- the tests
def method_groups(col, subset=None):
    src = subset if subset is not None else rows
    g = {}
    for r in src:
        v = fn(r.get(col))
        if v is None:
            continue
        m = r["method_auto"]
        if m and m not in ("", "Unclassified"):
            g.setdefault(m, []).append(v)
    return list(g.values())


def deep_classical(col, subset=None):
    src = subset if subset is not None else rows
    deep, clas = [], []
    for r in src:
        v = fn(r.get(col))
        if v is None:
            continue
        m = r["method_auto"]
        if not m or m in ("", "Unclassified"):
            continue
        (deep if m in DEEP else clas).append(v)
    return deep, clas


results = {}

# 1-2. method effect on OA and on Kappa (main text §3.2, §3.8)
results["OA ~ method category (all studies, §3.2)"] = \
    kw_effect(method_groups("oa_rep")) | {"quoted_p": 0.51}
results["Kappa ~ method category (§3.8)"] = \
    kw_effect(method_groups("kappa")) | {"quoted_p": 0.26}

# 3-4. scope subsets (§3.7)
rice = [r for r in rows if (r.get("scope") or "").strip() == "rice"]
crop = [r for r in rows if (r.get("scope") or "").strip() == "crop-classification"]
results["OA ~ method, rice-only subset (§3.7)"] = \
    kw_effect(method_groups("oa_rep", rice)) | {"quoted_p": 0.28}
results["OA ~ method, crop-only subset (§3.7)"] = \
    kw_effect(method_groups("oa_rep", crop)) | {"quoted_p": 0.60}

# 5. N-reporting (weighted-synthesis) subset (§3.6)
nrep = [r for r in rows if fn(r.get("n_val")) is not None]
results["OA ~ method, N-reporting subset (§3.6)"] = \
    kw_effect(method_groups("oa_rep", nrep), min_size=1) | {"quoted_p": 0.88}

# 6. rice vs crop scope contrast (§3.7)
results["OA: rice-only vs crop-only scope (§3.7)"] = mw_effect(
    [fn(r["oa_rep"]) for r in rice if fn(r["oa_rep"]) is not None],
    [fn(r["oa_rep"]) for r in crop if fn(r["oa_rep"]) is not None]) | {"quoted_p": 0.0002}

# 6b. selection check: does OA differ between N-reporting and non-reporting studies? (§3.5)
results["OA: N-reporting vs non-reporting studies (§3.5)"] = mw_effect(
    [fn(r["oa_rep"]) for r in nrep if fn(r["oa_rep"]) is not None],
    [fn(r["oa_rep"]) for r in rows
     if fn(r.get("n_val")) is None and fn(r["oa_rep"]) is not None]) | {"quoted_p": 0.15}

# 7-8. deep vs classical, between studies (§3.2, §3.8)
d, c = deep_classical("oa_rep")
results["OA: deep vs classical, between studies (§3.2)"] = mw_effect(d, c) | {"quoted_p": 0.118}
d, c = deep_classical("kappa")
results["Kappa: deep vs classical, between studies (§3.8)"] = mw_effect(d, c) | {"quoted_p": 0.21}

# 9-11. F1 secondary subset (§3.8 / §S8)
f1p = ROOT / str(ROOT / "data/secondary_f1_verified.csv")
if f1p.exists():
    f1rows = list(csv.DictReader(open(f1p, encoding="utf-8")))
    fg = {}
    for r in f1rows:
        v = fn(r.get("f1"))
        if v is None:
            continue
        fg.setdefault(r.get("method") or r.get("method_auto", ""), []).append(v)
    results["F1 ~ method category (§S8)"] = \
        kw_effect(list(fg.values())) | {"quoted_p": 0.18}
    F1DEEP = DEEP | {"Transfer/Pre-trained"}   # the released F1 grouping (§3.8)
    dd = [fn(r["f1"]) for r in f1rows
          if (r.get("method") in F1DEEP) and fn(r.get("f1")) is not None]
    cc = [fn(r["f1"]) for r in f1rows
          if r.get("method") and r["method"] not in F1DEEP and fn(r.get("f1")) is not None]
    results["F1: deep vs classical (§S8)"] = mw_effect(dd, cc) | {"quoted_p": 0.878}

# 12-13. within-study paired head-to-head (§3.3, §S7)
paired = json.load(open(ROOT / str(ROOT / "results/paired_verified.json"), encoding="utf-8"))
deltas = [p["delta"] for p in paired]
results["Within-study deep - classical, all pairs (§3.3)"] = \
    wilcoxon_effect(deltas) | {"quoted_p": 0.009}
trimmed = sorted(deltas)[:-3]          # drop the three large deltas the text discounts
results["Within-study deep - classical, 3 outliers removed (§3.3)"] = \
    wilcoxon_effect(trimmed) | {"quoted_p": 0.067}

# ------------------------------------------------------------------ output
for name, r in results.items():
    if "eps2" in r:
        r["effect_size"] = f"eps2 = {r['eps2']:.3f} [{r['eps2_ci95'][0]:.3f},{r['eps2_ci95'][1]:.3f}]"
        # An eps2 at or below its own null expectation is not a "small effect" -
        # it is what a random grouping produces, and the anchor must say so.
        r["magnitude"] = ("at/below chance" if r["eps2"] <= r["eps2_null_expectation"]
                          else magnitude(r["eps2"], "variance"))
    else:
        r["effect_size"] = f"r_rb = {r['rank_biserial']:+.3f}"
        r["magnitude"] = magnitude(r["rank_biserial"], "rb")
    r["p_recomputed"] = round(r["p"], 4)
    r["p_matches_manuscript"] = abs(r["p"] - r["quoted_p"]) < max(0.011, 0.1 * r["quoted_p"])

json.dump(results, open(ROOT / str(ROOT / "results/effect_sizes.json"), "w", encoding="utf-8"), indent=1, default=str)

cols = ["test", "n", "n1", "n2", "n_pairs", "k", "H", "U", "W", "p_recomputed",
        "quoted_p", "p_matches_manuscript", "eps2", "eps2_ci95", "eps2_null_expectation", "eta2_H", "rank_biserial",
        "rank_biserial_ci95", "cliffs_delta", "magnitude"]
with open(ROOT / str(ROOT / "data/effect_sizes.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["analysis"] + cols)
    for name, r in results.items():
        w.writerow([name] + [r.get(c, "") for c in cols])

print(f"{'analysis':52s} {'p(new)':>8s} {'p(ms)':>7s} {'ok':>3s}  effect size")
for name, r in results.items():
    ok = "y" if r["p_matches_manuscript"] else "NO"
    print(f"{name:52s} {r['p_recomputed']:8.4f} {r['quoted_p']:7.4f} {ok:>3s}  "
          f"{r['effect_size']:34s} ({r['magnitude']})")
