"""Rebuild the training data, train the "Kdaj odpreti okno?" model and test it honestly.

Run from the hekaton folder:   venv/bin/python train_model.py
Output: okno_model_live.joblib (used by app.py) and the test results printed below.

Data (ADA Hack 2026 data pack, source ARSO):
  ARSOhack/01_meritve/arso_lj_pm_urno_2016_2026.csv.gz        hourly PM2.5 (measurement)
  ARSOhack/03_vreme_inverzija/inverzija_urno_2016_2026.csv     hourly temperatures and wind (derived from measurements)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ARSOhack" / "primeri"))
from paket import nalozi_arso, nalozi_inverzijo  # noqa: E402

# no 'station' input: the three stations follow the same pattern, so one model serves all of Ljubljana
FEATURES = ["pm_now", "pm_l1", "pm_l3", "pm_m6", "pm_m24", "pm_l24", "wind_now", "hour0",
            "h", "hour_t", "dow_t", "month", "dt_topol", "dt_krvavec"]
STATION_CODE = {"E403": 0, "E404": 1, "E405": 2}          # row label only; Celovška (2) is used only for testing
TRAIN_END, TEST_START, TEST_END = "2025-10-01", "2025-10-01", "2026-04-01"
WINTER = [10, 11, 12, 1, 2, 3]


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                         min_samples_leaf=100, random_state=0)


# ---------------------------------------------------------------- 1. data
arso = nalozi_arso(["E403", "E404", "E405"])
arso["time_utc"] = pd.to_datetime(arso.time_utc, utc=True)
arso = arso.drop_duplicates(["station_code", "time_utc"])
idx = pd.date_range("2021-01-01", "2026-11-12 23:00", freq="h", tz="UTC")
PM25 = arso.pivot(index="time_utc", columns="station_code", values="pm25").reindex(idx)
IV = nalozi_inverzijo()
IV["time_utc"] = pd.to_datetime(IV.time_utc, utc=True)
IV = IV.set_index("time_utc").reindex(idx)


def build(s):
    """One row per (start hour t0, lead h = 1..12). Inputs use only data up to t0; y = measured PM2.5 at t0 + h."""
    pm = PM25[s]
    o = pd.DataFrame(index=idx)
    o["pm_now"] = pm
    o["pm_l1"] = pm.shift(1)
    o["pm_l3"] = pm.shift(3)
    o["pm_m6"] = pm.rolling(6, min_periods=4).mean()
    o["pm_m24"] = pm.rolling(24, min_periods=18).mean()
    o["pm_l24"] = pm.shift(24)
    o["wind_now"] = IV.wind_speed_lj_bezigrad
    o["dt_topol"] = IV.dt_bezigrad_topol            # T(Bežigrad) − T(Topol); negative = inversion
    o["dt_krvavec"] = IV.dt_bezigrad_krvavec
    o["inv_pasja"] = IV.inv_pasja_ravan              # only for the notebook-05 baseline
    o["hour0"] = idx.hour
    out = []
    for h in range(1, 13):
        f = o.copy()
        th = idx + pd.Timedelta(hours=h)
        f["h"] = h
        f["time_t"] = th
        f["hour_t"] = th.hour
        f["dow_t"] = th.day_of_week
        f["month"] = th.month
        f["y"] = pm.reindex(th).values
        f["station"] = STATION_CODE[s]
        out.append(f.reset_index(names="t0"))
    return pd.concat(out, ignore_index=True)


D = pd.concat([build(s) for s in STATION_CODE], ignore_index=True)
D = D[D.t0.dt.month.isin(WINTER)].dropna(subset=["pm_now", "y"])
# target of the main ("shape") model: how much cleaner/dirtier each hour is than the 12-hour average (log scale)
D["rel"] = np.log1p(D.y) - D.groupby(["station", "t0"]).y.transform(lambda v: np.log1p(v).mean())

TR = D[(D.t0 < TRAIN_END) & (D.station < 2)]
print(f"training rows: {len(TR)}  ({TR.t0.min():%Y-%m-%d} – {TR.t0.max():%Y-%m-%d}, Bežigrad + Vič, Oct–Mar)")

# ---------------------------------------------------------------- 2. train
shape = gbm().fit(TR[FEATURES], TR.rel)
level = gbm().fit(TR[FEATURES], np.log1p(TR.y))
joblib.dump({"model": shape, "level_model": level, "features": FEATURES,
             "note": "Shape model: log PM2.5 relative to the 12 h window mean (lower = cleaner hour). "
                     "Level model: log1p PM2.5. Trained on winters Oct–Mar 2021–2025, Bežigrad + Vič."},
            ROOT / "okno_model_live.joblib")
print("saved okno_model_live.joblib")

# ---------------------------------------------------------------- 3. honest test on winter 2025/26 (never seen)
TE = D[(D.t0 >= TEST_START) & (D.t0 < TEST_END)].copy()
TE["model"] = shape.predict(TE[FEATURES])

# notebook-05 baseline: typical winter profile by station / inversion / weekend / hour, anchored to the current value
base = D[(D.h == 1) & (D.t0 >= "2023-11-01") & (D.t0 < TRAIN_END) & D.t0.dt.month.isin([11, 12, 1, 2, 3])].dropna(subset=["inv_pasja"])
TIP = base.groupby(["station", "inv_pasja", base.t0.dt.day_of_week >= 5, base.t0.dt.hour]).pm_now.mean()
TE = TE.dropna(subset=["inv_pasja"])
p_t = TIP.reindex(list(zip(TE.station, TE.inv_pasja, TE.time_t.dt.day_of_week >= 5, TE.hour_t))).values
p_0 = TIP.reindex(list(zip(TE.station, TE.inv_pasja, TE.t0.dt.day_of_week >= 5, TE.hour0))).values
TE["notebook05"] = p_t * np.clip(TE.pm_now.values / p_0, 0.3, 3.0)


def score(df, cols, waking):
    """Share of start hours where the chosen hour is among the 3 actually cleanest of the next 12 (waking: 07–23 local)."""
    d = df
    if waking:
        loc = (d.hour_t + 1) % 24
        d = d[(loc >= 7) & (loc <= 22)]
    hits = {c: 0 for c in cols}
    rnd, less = [], {c: [] for c in cols}
    n = 0
    for _, g in d.groupby(["station", "t0"], sort=False):
        if len(g) < (4 if waking else 10):
            continue
        n += 1
        y = g.y.values
        top3 = set(np.argsort(y)[:3])
        rnd.append(min(3, len(y)) / len(y))
        for c in cols:
            k = int(np.argmin(g[c].values))
            hits[c] += k in top3
            less[c].append((y[k], y.mean()))
    out = {c: (hits[c] / n, 1 - sum(a for a, _ in less[c]) / sum(b for _, b in less[c])) for c in cols}
    return n, float(np.mean(rnd)), out


print("\nTest winter 2025/26 — chosen hour among the 3 actually cleanest of the next 12 hours")
for label, sel in [("Bežigrad + Vič", TE.station < 2), ("Celovška (never in training)", TE.station == 2)]:
    for waking in (True, False):
        n, rnd, res = score(TE[sel], ["notebook05", "model"], waking)
        print(f"  {label:28s} {'waking hours' if waking else 'all hours   '} ({n} starts): random {rnd:.0%} | "
              + " | ".join(f"{c} {h:.1%} ({l:.0%} less PM2.5 than a random hour)" for c, (h, l) in res.items()))
