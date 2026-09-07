from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import json, statistics

V = [
 dict(doi="10.3390/agriculture11100977", deep="BiLSTM-Attention", deep_oa=93.51, cls="RF", cls_oa=88.09,
      src='"accuracy of BiLSTM-Attention was 0.9351"; "RF (0.8809)"', kind="own"),
 dict(doi="10.3390/agriculture12122083", deep="3D-CNN", deep_oa=97.0, cls="XGBoost", cls_oa=84.0,
      src='"more than 97% overall accuracy"; "XGBoost ... (OA = 84%)"', kind="own (deep is lower bound)"),
 dict(doi="10.3390/agronomy13112800", deep="LSTM", deep_oa=90.61, cls="RF", cls_oa=83.09,
      src='"LSTM ... OA of 90.61%"; "7.52% ... improvement in OA compared to RF"', kind="derived"),
 dict(doi="10.5281/zenodo.18168302", deep="T2VRCM (ViT/Mamba family)", deep_oa=92.33, cls="RF", cls_oa=83.01,
      src='"overall accuracy of 92.33%"; "RF ... (OA=83.01%)"', kind="own (benchmark dataset)"),
 dict(doi="10.3390/computers14080336", deep="CNN-1D", deep_oa=92.8, cls="RF", cls_oa=93.48,
      src='"0.928 ... CNN-1D"; "Random Forest ... 93.48%"', kind="own (pure-deep vs RF)"),
 dict(doi="10.1016/j.eswa.2024.124771", deep="3D-CNN", deep_oa=96.67, cls="SVM", cls_oa=95.23,
      src='"3DCNN ... 0.9667"; "SVM was 0.9523"', kind="own"),
 dict(doi="10.3390/agriculture16090920", deep="RF+LSTM (temporal DL)", deep_oa=93.61, cls="RF", cls_oa=85.40,
      src='"incorporating the LSTM, OA increased to ... 93.61%"; "Without ... LSTM, OA was 84.67% for RF ... 85.40%"', kind="own (ablation)"),
 dict(doi="10.3390/rs13071360", deep="deep-learning model", deep_oa=91.2, cls="RF/SVM", cls_oa=88.5,
      src='"pheno-deep ... 88.8%, only 2.4% lower than the deep learning method"; "88.5% for ... RF ... and SVM"', kind="derived"),
 dict(doi="10.1007/s12518-025-00659-x", deep="U-Net", deep_oa=94.0, cls="RF", cls_oa=94.8,
      src='"U-Net (94%)"; "random forest ... 94.8%"', kind="own"),
 dict(doi="10.3390/rs14153721", deep="MLSTM-FCN", deep_oa=97.21, cls="RF", cls_oa=97.04,
      src='"MLSTM-FCN ... 0.9721"; "0.17-1.23% improvement compared to ... random forest" (conservative 0.17)', kind="derived (conservative)"),
 dict(doi="10.3390/rs17183207", deep="2D CNN-GRU", deep_oa=99.12, cls="XGBoost", cls_oa=95.49,
      src='"2D CNN-GRU model, with an accuracy of 99.12%"; "Xgboost performing best at 95.49%"', kind="own"),
 dict(doi="10.1080/14498596.2023.2174196", deep="U-Net", deep_oa=93.0, cls="RF", cls_oa=95.9,
      src='"accuracy of the U-Net network was the highest" (OA 0.93); "RF model with OA = 95.9%"', kind="own (classical wins)"),
 dict(doi="10.1109/jstars.2025.3560992", deep="LSTM-OB", deep_oa=88.2, cls="RF", cls_oa=85.48,
      src='"LSTM-OB method reached an OA of 88.20%"; "RF achieved an OA of 85.48%"', kind="own"),
 dict(doi="10.3390/rs14030498", deep="dual-attention CNN", deep_oa=98.54, cls="RF", cls_oa=87.0,
      src='"highest accuracy ... OA = 98.5%" / "98.54%"; "outperformed other classifiers with an overall accuracy of 87%"', kind="own"),
]
for r in V:
    r["delta"] = round(r["deep_oa"] - r["cls_oa"], 2)

deltas = [r["delta"] for r in V]
n = len(deltas)
print("Hand-verified head-to-head papers:", n)
print("deltas (deep - classical):", deltas)
print("mean delta: %.2f pp | median: %.2f pp | SD: %.2f" %
      (statistics.mean(deltas), statistics.median(deltas), statistics.stdev(deltas)))
pos = sum(d > 0 for d in deltas); neg = sum(d < 0 for d in deltas)
print("deep wins: %d / %d   classical wins: %d" % (pos, n, neg))

# Wilcoxon signed-rank
try:
    from scipy.stats import wilcoxon, binomtest
    w, p = wilcoxon(deltas)
    print("Wilcoxon signed-rank: W=%.0f, p=%.4f" % (w, p))
    bt = binomtest(pos, n, 0.5)
    print("sign test (deep>classical): p=%.4f" % bt.pvalue)
except Exception as e:
    print("scipy n/a:", e)

# bootstrap CI of the mean delta
import random
random.seed(11)
boot = sorted(statistics.mean([random.choice(deltas) for _ in deltas]) for _ in range(20000))
print("mean delta 95%% bootstrap CI: %.2f to %.2f pp" % (boot[500], boot[19500]))

json.dump(V, open(str(ROOT / "results/paired_verified.json"), "w", encoding="utf-8"), indent=1)
print("wrote analysis/paired_verified.json")
