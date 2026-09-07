from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
# ------------------------------------------------------------------------------
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as sps

NL = chr(10)
ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / str(ROOT / "data/meta_study_table_v2.csv")
OUT = ROOT / "analysis" / str(ROOT / "results/resolution_reliability.json")

# finest ground sample distance each platform can actually deliver, in metres
FINEST = {"Sentinel-2": 10, "Sentinel-1": 5, "Landsat": 15, "MODIS": 250,
          "RADARSAT": 3, "ENVISAT ASAR": 30, "ALOS/PALSAR": 10, "SPOT": 1.5}


def native(sensor):
    for k, v in FINEST.items():
        if sensor.lower().startswith(k.lower()):
            return v
    return None


def main():
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
            if str(r.get("eligible", "1")).strip() != "0"]

    usable, impossible, unclassified = [], [], 0
    for r in rows:
        try:
            oa = float(r["oa_rep"])
            res = float(r["res_m"])
        except (ValueError, TypeError, KeyError):
            continue
        n = native((r.get("sensor") or "").strip())
        if n is None:
            unclassified += 1
            continue
        (impossible if res < n else usable).append((oa, res))

    tot = len(usable) + len(impossible)
    res = {"n_with_sensor_and_resolution": tot,
           "n_physically_possible": len(usable),
           "n_finer_than_native": len(impossible),
           "pct_finer_than_native": round(100 * len(impossible) / tot, 1),
           "n_sensor_unclassified": unclassified}

    def fit(pairs, label):
        if len(pairs) < 10:
            return None
        oa = np.array([p[0] for p in pairs])
        lr = np.log10(np.array([p[1] for p in pairs]))
        sl, ic, rv, pv, se = sps.linregress(lr, oa)
        out = {"n": len(pairs), "beta_pp_per_log10m": round(float(sl), 3),
               "p": round(float(pv), 4), "r2": round(float(rv ** 2), 4)}
        print(f"  {label:34} n={out['n']:>4}  β={out['beta_pp_per_log10m']:+.3f} pp/log10 m  "
              f"p={out['p']:.3f}")
        return out

    print(f"sensor+resolution pairs: {tot}  "
          f"({res['pct_finer_than_native']}% finer than native)")
    res["fit_all"] = fit(usable + impossible, "all coded pairs")
    res["fit_possible_only"] = fit(usable, "physically possible pairs only")
    res["fit_impossible_only"] = fit(impossible, "impossible pairs only")

    a, b = res["fit_all"], res["fit_possible_only"]
    res["null_survives_restriction"] = bool(b and b["p"] >= 0.05)
    print(f"{NL}resolution null survives the restriction: "
          f"{'yes' if res['null_survives_restriction'] else 'NO'}")
    if a and b:
        print(f"  β moves from {a['beta_pp_per_log10m']:+.3f} to {b['beta_pp_per_log10m']:+.3f}")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
