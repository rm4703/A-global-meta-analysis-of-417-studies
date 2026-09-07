from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / str(ROOT / "data/meta_study_table_v2.csv")

# --- studies whose reported OA is NOT an accuracy of paddy/crop extent or type
# (hand-confirmed against title + extracted snippet; M2). Keyed by DOI.
INELIGIBLE = {
    "10.3390/plants9050559":        "Ground-based stereo computer-vision weed classification, not satellite paddy mapping (dasun M2)",
    "10.3390/ijerph19052567":       "Heavy-metal stress detection in rice (biophysical retrieval), not paddy-extent mapping (dasun M2)",
    "10.3390/s18072172":            "Vegetation index for heavy-metal stress-level discrimination, not extent/type mapping",
    "10.3390/agriengineering7070205":"Classification of paddy diseases, not paddy-extent mapping",
    "10.3390/app14062575":          "Near-ground image processing of rice-seedling planting condition (YOLOv8n), not satellite extent mapping",
    "10.1080/01431161.2019.1706112":"Weed mapping from UAV imagery, not satellite crop-extent classification",
    "10.1080/01431161.2018.1441569":"Early-season weed mapping from UAV imagery, not paddy-extent mapping",
    "10.1371/journal.pone.0309982": "Rice yield estimation, not paddy-extent mapping",
    "10.3390/land12091680":         "Digital mapping of soil-salinity levels, not crop-type classification",
    "10.1007/s11707-019-0803-7":    "Typhoon-induced rice flooding/lodging damage detection, not extent mapping",
    "10.4209/aaqr.2012.06.0150":    "Rice-straw burned-area mapping for an emissions inventory (burn status, not paddy extent)",
    "10.3390/rs70505077":           "Object-based flood mapping / affected-rice estimation (flood extent, not paddy-type accuracy)",
    "10.3390/agriculture14030496":  "Methane-emission spatial analysis from LST/agronomic flooding, not paddy-extent classification",
}

FAMILIES = {"Phenology/Threshold", "Random Forest/SVM", "Object-Based",
            "CNN/Deep Learning", "RNN/LSTM", "Attention/Transformer",
            "Foundation Model", "Transfer/Pre-trained"}


def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        rows = list(reader)

    for c in ["eligible", "eligibility_reason", "method_audit_note", "multi_method"]:
        if c not in cols:
            cols.append(c)

    n_excl = n_renamed = n_multi = 0
    for r in rows:
        doi = (r.get("doi") or "").strip()
        # eligibility
        if doi in INELIGIBLE:
            r["eligible"] = "0"
            r["eligibility_reason"] = INELIGIBLE[doi]
            n_excl += 1
        else:
            r["eligible"] = "1"
            r["eligibility_reason"] = ""
        # method-leaf correction (M3)
        note = ""
        if r.get("method_auto") == "Foundation Model":
            r["method_auto"] = "Transfer/Pre-trained"
            note = "Auto-label 'Foundation Model' corrected to 'Transfer/Pre-trained': not a self-supervised EO foundation model (M3)"
            n_renamed += 1
        r["method_audit_note"] = note
        # multi-method flag (m3)
        fams = [m for m in (r.get("methods_mentioned") or "").split("|") if m in FAMILIES]
        r["multi_method"] = "1" if len(set(fams)) > 1 else "0"
        if r["multi_method"] == "1":
            n_multi += 1

    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    elig = sum(1 for r in rows if r["eligible"] == "1")
    print(f"rows: {n}  eligible: {elig}  excluded: {n_excl}")
    print(f"method leaf renamed (Foundation Model -> Transfer/Pre-trained): {n_renamed}")
    print(f"multi-method studies flagged: {n_multi}")
    print("excluded DOIs:")
    for d in INELIGIBLE:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
