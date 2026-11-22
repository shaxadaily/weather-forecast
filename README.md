# 🏠 Kdaj odpreti okno? ("When should I open the window?")

**ADA Hack 2026 · challenge *Zrak, ki ga dihamo* · Ljubljana basin · data: ARSO**

A web app that tells Ljubljana residents, in one glance, whether they can open the window now and **when the air will be cleanest in the next hours**, so they can air out their flat with as little particulate pollution (PM2.5) as possible. It's built on a machine-learning model trained on four winters of ARSO measurements and tested honestly on winters it never saw.

---

## Contents

1. [The problem](#1-the-problem)
2. [What we built](#2-what-we-built)
3. [Data](#3-data)
4. [The model](#4-the-model)
5. [How we trained and tested it](#5-how-we-trained-and-tested-it)
6. [Results](#6-results)
7. [Experiments: what helped and what didn't](#7-experiments-what-helped-and-what-didnt)
8. [Interesting findings](#8-interesting-findings)
9. [Limitations](#9-limitations)
10. [Possible improvements](#10-possible-improvements)
11. [How to run](#11-how-to-run)

---

## 1. The problem

In winter, the Ljubljana basin often traps polluted air. On cold, calm days a **temperature inversion** forms: warmer air sits above colder air in the valley like a lid, so smoke from household heating and traffic exhaust stays near the ground.

Almost everyone airs out their flat every day, but nobody knows **when the air outside is cleanest**. Official data exists (ARSO stations, weather measurements, models), but it's scattered and hard to read for ordinary people.

**Target users:** citizens, especially vulnerable groups (parents of small children, older people, people with asthma or heart disease, runners).
**Decision we support:** *"Should I open the window now, and if not, when?"*

---

## 2. What we built

### The app (`app.py`, Streamlit)

**Main screen, deliberately minimal:**

- **📍 Kje si?** Choose one of 17 Ljubljana districts. The app uses the nearest of the three official ARSO stations (Bežigrad, Vič, Celovška).
- **One big traffic-light answer:**

| Air now | Answer |
|---|---|
| good or fair | 🟢 **OKNO LAHKO ODPREŠ**: "Po napovedi bo najčistejši ob 16:00." |
| moderate | 🟡 **ODPRI LE NA KRATKO**: "Po napovedi bo čistejši ob …" |
| poor or very poor | 🔴 **OKNO ZAPRI**: "Po napovedi bo čistejši jutri ob 7:00. Takrat prezrači za 5–10 minut." |

- **One honest source line:** which station the data comes from and how far it is.

**"Več podrobnosti" (folded away):**

- the measured PM2.5 and air-quality class,
- an hour-by-hour strip with the best hour ✅ and night hours 🌙,
- recommendation tiles: airing out, exercise outside, a walk with a baby, older people / asthma / heart,
- a plain explanation of inversion, wind and temperature,
- a graph: last 12 hours measured, the forecast with a **±5 µg/m³ band**, the best hour as a green dot, a fixed 0–60 scale, and a legend.

**Demo mode "Pretekli dan"** (side panel): pick any hour from October 2025 to 12 November 2026. The app shows its advice **and what ARSO really measured afterwards** (an orange line), plus whether the suggested hour turned out to be among the 3 cleanest. It works offline.

**Measurement vs. estimate** is labelled everywhere: measured values come from ARSO, the forecast is marked as an estimate.

### Files

| File | Purpose |
|---|---|
| `app.py` | the web app |
| `train_model.py` | rebuilds the training data, trains the model, runs the honest test |
| `okno_model_live.joblib` | the trained model, used by the app |
| `.streamlit/config.toml` | light theme |
| `ARSOhack/` | the organisers' data pack (not ours) |

---

## 3. Data

All data is from the organisers' data pack (source: **ARSO**, Slovenian Environment Agency).

**Training and testing:**

| Data | File | Type |
|---|---|---|
| Hourly PM2.5 at LJ Bežigrad, LJ Vič, LJ Celovška | `01_meritve/arso_lj_pm_urno_2016_2026.csv.gz` + `…_2026_11_dopolnitev.csv` | measurement |
| Hourly temperatures (Bežigrad, Topol 660 m, Krvavec 1742 m) and wind at Bežigrad | `03_vreme_inverzija/inverzija_urno_2016_2026.csv` | derived from measurements |

**Live, in the app:**

| Data | Source |
|---|---|
| PM2.5 / PM10, last days | organisers' cloud capture of the ARSO XML feed |
| Temperature and wind | ARSO weather stations Ljubljana Bežigrad, Topol, Krvavec (`meteo.arso.gov.si`) |

**Why Topol and Krvavec, not Pasja ravan?** The organisers recommend Pasja ravan (1019 m) as the best inversion indicator, but **ARSO doesn't publish it live**. Topol and Krvavec are published live, and together they give the same accuracy.

---

## 4. The model

### Algorithm

**Gradient boosting** with `HistGradientBoostingRegressor` from scikit-learn. It builds 400 small decision trees one after another, and each tree corrects the errors of the previous ones. Settings: learning rate 0.06, at most 63 leaves per tree, at least 100 examples per leaf.

The model file contains two models:

1. **Main ("shape") model:** ranks the next 12 hours from cleanest to dirtiest, which chooses the best hour.
2. **Level model:** estimates approximate PM2.5 values for the graph.

### Inputs (14, all available live from ARSO)

| Group | Inputs |
|---|---|
| PM2.5 at the station | now, 1 h ago, 3 h ago, 6 h average, 24 h average, same hour yesterday |
| Inversion and wind | T(Bežigrad) − T(Topol), T(Bežigrad) − T(Krvavec), wind speed at Bežigrad |
| Time | hour now, hours ahead (1–12), hour being forecast, day of week, month |

There's no station input: one model serves all of Ljubljana (see section 7).

### Output

For each of the next 12 hours, a **cleanliness score**:

```
score(hour) = log(1 + PM2.5 in that hour) − average of log(1 + PM2.5) over the 12 hours
```

A **negative** score means cleaner than average. The app suggests the hour with the **lowest score** between 07:00 and 23:00 that hasn't passed yet.

The key idea is that the model learns the **shape** of the coming hours, *when* it will be cleaner, rather than exact levels. Choosing the best hour only needs the hours in the right order, and this was what made the model work.

The traffic light (🟢🟡🔴) is **not** from the model. It comes from the current ARSO measurement and the class thresholds (`razred_iqz`, an approximation of the ARSO scale).

---

## 5. How we trained and tested it

### Building training examples ("pretend it's now")

For every past hour:
1. compute the inputs from data **up to that hour only** (no future information),
2. look 1–12 hours ahead and take the **actually measured** PM2.5,
3. compute the target score.

**Training set:** winters (October–March) from **January 2021 to March 2025**, stations Bežigrad and Vič, **439,172 examples**.

### Splitting the data

We split **by time, not randomly**. Neighbouring hours are almost identical, so a random split would leak answers into training and make the model look far better than it is.

| | Period | Stations |
|---|---|---|
| Training | winters Jan 2021 – Mar 2025 | Bežigrad, Vič |
| Test | winter Oct 2025 – Mar 2026 | Bežigrad, Vič, **Celovška** (never used in training) |
| Extra check | leave-one-winter-out over the last 4 winters | all three |

### The score: "top-3 hit"

For every hour of the test winter as "now": is the suggested hour one of the **3 actually cleanest** of the next 12 hours (daytime only, 07:00–23:00)? We took this measure from the organisers' notebook 05 so the comparison is fair. With about 8 daytime hours to choose from, **random guessing scores ~44%**.

We also measured:
- whether the suggested hour is within 5 µg/m³ of the truly cleanest hour,
- the average error of the predicted PM2.5 levels.

### Baselines

- **Random guess**
- **Average pattern:** the average PM2.5 by month and hour from past winters
- **Organisers' method (notebook 05):** a typical profile by inversion / weekend / hour, scaled to the current value, optionally adjusted with CAMS. We rebuilt it fairly, using only past winters.

---

## 6. Results

### Choosing the best hour (top-3 hit, daytime hours)

**Test winter 2025/26:**

| Method | Bežigrad + Vič | Celovška (never seen) | All three |
|---|---|---|---|
| Random guess | 44% | 44% | 44% |
| Organisers' method (notebook 05) | 61.9% | 58.7% | ~61% |
| Average pattern | | | 67.3% |
| **Our model** | **68.0%** | **69.0%** | **~68%** |

**Robustness check, 4 winters** (train on 3, test on the 4th, all stations):

| Test winter | Random | Average pattern | **Our model** |
|---|---|---|---|
| 2022/23 | 44% | 65.1% | 65.8% |
| 2023/24 | 44% | 64.3% | 65.3% |
| 2024/25 | 44% | 63.8% | 67.6% |
| 2025/26 | 44% | 67.3% | 68.3% |
| **Average** | **44%** | **65.1%** | **66.8%** |

➡ **The honest headline number is about 67%.** It's stable between 65% and 68% across winters and stations.

**How close is a "miss"?** In **87%** of cases the suggested hour is within 5 µg/m³ of the truly cleanest hour, and the typical gap is only **0.5 µg/m³**.

**With stricter or looser definitions (winter 2025/26):**

| Hit if the suggested hour is… | Random | Average pattern | Our model |
|---|---|---|---|
| the single cleanest hour | 15% | 35% | 37% |
| among the 2 cleanest | 30% | 54% | 55% |
| among the 3 cleanest | 44% | 67% | 68% |
| among the 4 cleanest | 59% | 77% | 78% |

**Benefit to the user:** at the suggested hour, PM2.5 is on average **about 16% lower** than at a random daytime hour.

### Predicting levels (the grey line in the graph)

| Hours ahead | 1 | 3 | 6 | 9 | 12 |
|---|---|---|---|---|---|
| Our model, average error (µg/m³) | 2.6 | 3.8 | 5.0 | 5.7 | 6.2 |
| "Air stays as it is now" | 2.2 | 4.3 | 6.2 | 7.0 | 7.6 |

**Daily error, test winter 2025/26 (183 days):**

| Method | Average daily error | Days within 5 µg/m³ | On polluted days (above 25) |
|---|---|---|---|
| **Our model** | **4.8 µg/m³** | **64%** | **8.3** |
| Organisers: typical profile | 5.8 µg/m³ | 52% | 9.6 |
| Organisers: + CAMS (their default) | 8.5 µg/m³ | 28% | 16.9 |

**Error by air quality that day:** good days 2.8 · fair 4.4 · moderate 6.1 · poor 7.8 · very poor 15.4 µg/m³.
**By month:** October 3.1 · November 4.5 · **December 6.0 · January 7.0** · February 4.8 · March 3.8.

The real value falls within the **±5 µg/m³ band** about **61–66%** of the time.

---

## 7. Experiments: what helped and what didn't

Every variant was compared on the same test hours. For small differences we used a **day-block bootstrap** (resampling whole days) to check whether a difference is real or noise. Our rule: **only change the model if the change is clearly better**.

| Experiment | Result | Decision |
|---|---|---|
| Extra inputs: other stations, PM10, NO₂, deeper inversion, same hour on previous days | +0.1 points | rejected (no gain) |
| Train directly on "is this hour in the top 3?" (classifier) | +0.9 points, within noise | rejected |
| Add Celovška to training | +0.8 points, within noise | rejected; Celovška kept as an independent test |
| **Random forest** instead of gradient boosting | −0.9 points | rejected |
| **Remove the station input** | −0.2 points, within noise | **adopted**: simpler, one model for all stations |
| **Three separate models**, one per station | +0.6 points, within noise | rejected (less data per model, loses the "never seen" test) |
| Remove all PM2.5 measurements (only time + inversion + wind) | −0.8 points | rejected, but very revealing (section 8) |
| **Remove inversion** | **−1.9 points, real loss** | inversion kept |
| Remove wind | −0.7 points, borderline | wind kept (consistent small help) |
| **Train on all seasons** | winter −0.8 (noise); **summer +3.3 (real)** | kept winter-only (heating-season app) |
| **Weather as input** (ERA5, the *actual* past weather = best case for a forecast) | best hour −0.3; levels better: 4.9 → 4.5 µg/m³, 12 h ahead 6.2 → 5.4 | not adopted before the deadline; a good next step |
| CAMS air-quality model as input | levels worse | rejected |
| Pasja ravan → Topol + Krvavec (for live use) | 68.2% → 68.2% | adopted (needed for live data) |

---

## 8. Interesting findings

1. **Ljubljana's winter air follows a strong daily rhythm.** Simply knowing the average pattern by month and hour scores 67%. Our model adds about 1–2 points on top, consistently in every winter. Most of our advantage over the organisers' method comes from **a better use of this pattern**, not from complex machine learning. Their method splits the pattern into many small groups (inversion × weekend × hour) and scales it to the current value, which makes it noisier.

2. **What the model relies on** (importance test): **time ~70%** (hour being forecast alone ~45%), **current PM2.5 ~15%**, **inversion and wind ~14%**. A model without any PM2.5 measurement still reaches 67.4%. The measurements matter more for the traffic light ("is it bad *now*?") than for choosing the hour.

3. **The inversion matters measurably.** Removing it is the only removal that clearly hurts (−1.9 points), on every station.

4. **The ceiling is low because hours are often tied.** In **57%** of cases, the three cleanest hours differ by **less than 2 µg/m³**. Which one is "number one" is then basically chance, so no model can reach 100%.

5. **The model generalises to a new station.** Celovška is a traffic station the model never saw, yet it scores the same (69%).

6. **The regional models overestimate Ljubljana.**
   - CAMx (ARSO, 4.4 km) is now **10–24 µg/m³ too high** in winter, and the gap grows each year. It also predicts a morning peak (6–8 UTC) that measurements don't show.
   - CAMS (Europe, ~10 km) shows the same trend, which is why adding CAMS made the organisers' forecast worse (8.5 vs 5.8 µg/m³ daily error).

7. **Levels are hardest exactly when they matter most.** The forecast error grows with pollution, and it's largest in December and January during episodes. The worst day (23 January 2026, avg. 65 µg/m³) was off by 21 µg/m³, and New Year's Eve (fireworks) by 16. The model can't know sudden weather changes, such as a front or wind clearing the basin at night.

8. **Even perfect weather knowledge helps only a little.** With the actual past weather as input, the level error drops just 0.3 µg/m³ overall (0.8 µg/m³ 12 h ahead). The hard part is sudden changes inside the basin.

9. **A random 80/20 split would have fooled us.** Neighbouring hours are nearly identical, so only a time-based split gives honest numbers.

---

## 9. Limitations

- **Only three official stations** in Ljubljana, so several districts share one station. The forecast is for the station, not the user's street.
- **Levels are approximate:** a typical error of ~5 µg/m³, larger during pollution episodes. The model is built to *rank hours*, not to predict exact values.
- **No future weather:** sudden weather changes during the night are missed.
- **Heating season only:** the model is trained on October–March and is weaker in summer.
- **Test caveat:** several variants were compared on the same test winter, which can make the final number slightly optimistic. We almost always kept the original model, and the 4-winter check (66.8%) confirms the result.
- **Prototype:** a browser app, not a mobile app. **Not medical advice.**

---

## 10. Possible improvements

**Model**
- **Live weather forecast (Open-Meteo) as input for the levels.** Our best-case test shows ~12% smaller error 12 hours ahead. It would need validation with *real* forecasts, not reanalysis.
- **CAMx as an input**, if ARSO published the numbers live (now only images).
- **All-season model**, so the app is also useful in summer (+3.3 points in summer, no loss in winter).
- **Uncertainty that grows with lead time** instead of a fixed ±5 band.

**Data and coverage**
- **Calibrated citizen sensors** (Sensor.Community, PurpleAir) as extra local context, clearly labelled as estimates.
- **District-level estimates** using the GRAL dispersion model (ARSO, 15 m) and building/heating data.

**Product**
- **Push notifications:** *"Tomorrow, air out at 8:00"* in the evening, and an alert when the air turns bad.
- **Kindergartens and schools as first users:** when to air classrooms and when children can go outside.
- **Mobile app or permanent hosting** (e.g. Streamlit Community Cloud).
- **User testing** with real residents to check that the messages are understood.

---

## 11. How to run

From the `hekaton` folder:

```bash
# start the app (opens http://localhost:8501)
venv/bin/streamlit run app.py

# retrain the model and reproduce the test results
venv/bin/python train_model.py
```

- Live mode needs internet. "Pretekli dan" mode works offline.
- To share on the same Wi-Fi, use the "Network URL" that Streamlit prints.
- To stop the app: `Ctrl + C` in Terminal, or `pkill -f "streamlit run app.py"`.

**Requirements:** Python 3.10+ with `streamlit`, `pandas`, `numpy`, `scikit-learn` (1.9.1, the version the model was trained with), `joblib`, `requests`, `altair`, `geopandas`.

---

*Data source: ARSO (Agencija Republike Slovenije za okolje), via the ADA Hack 2026 data pack. Prototype built at ADA Hack 2026.*
