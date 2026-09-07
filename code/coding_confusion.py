from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import glob, random, json
from collections import Counter, defaultdict

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
files = sorted(glob.glob(ROOT + "extracted_text/*.txt")) + sorted(glob.glob(ROOT + "extracted_text_new/*.txt"))
random.seed(42); sample = random.sample(files, 40)

auto = {}
for mf in [str(ROOT / "data/extracted_metrics.json"), str(ROOT / "data/extracted_metrics_new.json")]:
    for r in json.load(open(ROOT + mf, encoding="utf-8")):
        auto[r["file"]] = {"method": r["method_category"] or "Unclassified"}

# gold manual method labels (from validate_coding.py G dict), keyed by sample index 1..40
GM = {1:"Random Forest/SVM",2:"Phenology/Threshold",3:"Phenology/Threshold",4:"Random Forest/SVM",
5:"Object-Based",6:"Attention/Transformer",7:"Phenology/Threshold",8:"Phenology/Threshold",
9:"Phenology/Threshold",10:"Phenology/Threshold",11:"Random Forest/SVM",12:"CNN/Deep Learning",
13:"Unclassified",14:"Phenology/Threshold",15:"Phenology/Threshold",16:"Random Forest/SVM",
17:"Phenology/Threshold",18:"Phenology/Threshold",19:"Foundation Model",20:"Phenology/Threshold",
21:"Phenology/Threshold",22:"Phenology/Threshold",23:"Random Forest/SVM",24:"Phenology/Threshold",
25:"Random Forest/SVM",26:"Random Forest/SVM",27:"Random Forest/SVM",28:"Phenology/Threshold",
29:"Random Forest/SVM",30:"Random Forest/SVM",31:"Phenology/Threshold",32:"Phenology/Threshold",
33:"CNN/Deep Learning",34:"RNN/LSTM",35:"Phenology/Threshold",36:"Phenology/Threshold",
37:"Attention/Transformer",38:"RNN/LSTM",39:"Phenology/Threshold",40:"Unclassified"}

PARADIGM = {"Phenology/Threshold":"Knowledge-driven","Random Forest/SVM":"Shallow ML",
            "Object-Based":"Shallow ML","CNN/Deep Learning":"Deep learning","RNN/LSTM":"Deep learning",
            "Attention/Transformer":"Deep learning","Transfer/Pre-trained":"Deep learning",
            "Foundation Model":"Deep learning","Unclassified":"Unclassified"}

def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a)/n) * (sum(y == c for y in b)/n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

g7, a7 = [], []
for i, f in enumerate(sample, 1):
    a = auto.get(f.split("/")[-1][:-4])
    if not a:
        continue
    g7.append(GM[i]); a7.append(a["method"])

# confusion (gold -> auto) for disagreements only
confus = Counter((g, a) for g, a in zip(g7, a7) if g != a)
# adjacency: same paradigm vs cross-paradigm
same_par = sum(1 for g, a in zip(g7, a7) if g != a and PARADIGM[g] == PARADIGM[a])
cross_par = sum(1 for g, a in zip(g7, a7) if g != a and PARADIGM[g] != PARADIGM[a])

gp = [PARADIGM[x] for x in g7]; ap = [PARADIGM[x] for x in a7]

# deep vs classical binary (the distinction the headline comparison actually uses),
# on the studies the manual coder placed in a named category
DEEP = {"CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer", "Transfer/Pre-trained", "Foundation Model"}
gb = [("deep" if x in DEEP else "classical") for x, y in zip(g7, a7) if x != "Unclassified" and y != "Unclassified"]
ab = [("deep" if y in DEEP else "classical") for x, y in zip(g7, a7) if x != "Unclassified" and y != "Unclassified"]

out = {
    "n": len(g7),
    "method_7way": {"agreement_pct": round(100*sum(x==y for x,y in zip(g7,a7))/len(g7),1),
                    "kappa": round(kappa(g7, a7), 2)},
    "method_3paradigm": {"agreement_pct": round(100*sum(x==y for x,y in zip(gp,ap))/len(gp),1),
                         "kappa": round(kappa(gp, ap), 2)},
    "method_deep_vs_classical": {"n": len(gb),
                                 "agreement_pct": round(100*sum(x==y for x,y in zip(gb,ab))/len(gb),1),
                                 "kappa": round(kappa(gb, ab), 2)},
    "n_disagreements": sum(1 for g,a in zip(g7,a7) if g!=a),
    "disagreements_same_paradigm": same_par,
    "disagreements_cross_paradigm": cross_par,
    "top_confusions_gold_to_auto": {f"{g} -> {a}": c for (g, a), c in confus.most_common()},
}
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT + str(ROOT / "results/coding_confusion.json"), "w", encoding="utf-8"), indent=1)
print("\nwrote analysis/coding_confusion.json")
