"""Kdaj odpreti okno? — napoved najboljše ure za prezračevanje v Ljubljani (ADA Hack 2026).

Zagon (iz mape hekaton):  venv/bin/streamlit run app.py
Model: okno_model_live.joblib (zgradi ga train_model.py). Vir podatkov: ARSO. Napoved je ocena modela, ne meritev.
"""
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
PK = ROOT / "ARSOhack"
sys.path.insert(0, str(PK / "primeri"))
sys.path.insert(0, str(PK / "v_zivo"))
from paket import RAZRED_IME, haversine_km, razred_iqz  # noqa: E402
import zrak_v_zivo as zv  # noqa: E402

TZ = ZoneInfo("Europe/Ljubljana")
H = pd.Timedelta(hours=1)
STATIONS = {"E403": ("LJ Bežigrad", 46.0655449, 14.5127203), "E404": ("LJ Vič", 46.037487, 14.4892717),
            "E405": ("LJ Celovška", 46.0720916, 14.4871183)}
METEO = "https://meteo.arso.gov.si/uploads/probase/www/observ/surface/text/sl/observationAms_{}_latest.xml"
MEJE_PM25 = [(0, 10, 1), (10, 20, 2), (20, 25, 3), (25, 50, 4), (50, 120, 5)]
SVETLE = {1: "#e8f6ec", 2: "#f1f8e4", 3: "#fdf6d8", 4: "#fdeedd", 5: "#fbe3e6"}
TEMNE = {1: "#1e7a3c", 2: "#5a8a1f", 3: "#8a6d00", 4: "#a24f00", 5: "#9b0d23"}
OBRAZ = {1: "😀", 2: "🙂", 3: "😐", 4: "😷", 5: "🤢"}
POMEN = {
    1: "Zrak je čist. Uživaj na prostem.",
    2: "Zrak je v redu za večino ljudi.",
    3: "Zrak je zmerno onesnažen. Občutljivi naj bodo zunaj manj časa.",
    4: "Zrak je onesnažen. Občutljivi naj ostanejo notri, ostali naj se zunaj manj naprezajo.",
    5: "Zrak je zelo onesnažen. Vsi naj omejijo čas zunaj.",
}
DNI = ["ponedeljek", "torek", "sreda", "četrtek", "petek", "sobota", "nedelja"]

st.set_page_config(page_title="Kdaj odpreti okno?", page_icon="🏠", layout="centered")


# ---------------------------------------------------------------- model in podatki
@st.cache_resource
def model():
    return joblib.load(ROOT / "okno_model_live.joblib")


@st.cache_data
def cetrti():
    import geopandas as gpd
    cs = gpd.read_file(PK / "04_prostor" / "mol_cetrtne_skupnosti.geojson")
    c = cs.to_crs(3794).geometry.centroid.to_crs(4326)
    df = pd.DataFrame({"ime": cs.cs_ime.str.replace("ČETRTNA SKUPNOST ", "").str.title(), "lat": c.y, "lon": c.x})
    return df.sort_values("ime").reset_index(drop=True)


def najblizja(lat, lon):
    return min(STATIONS, key=lambda k: haversine_km(lat, lon, STATIONS[k][1], STATIONS[k][2]))


@st.cache_data(ttl=600, show_spinner=False)
def live_pm():
    zg = zv.arso_zgodovina(4)
    return zg[zg.station_code.isin(STATIONS)][["time_utc", "station_code", "pm25", "pm10"]]


@st.cache_data(ttl=600, show_spinner=False)
def live_meteo():
    out = {}
    for sid in ["LJUBL-ANA_BEZIGRAD", "TOPOL", "KRVAVEC"]:
        try:
            x = requests.get(METEO.format(sid), timeout=20).text
            g = lambda tag: (re.search(rf"<{tag}>([^<]*)</{tag}>", x) or [None, None])[1]  # noqa: E731
            out[sid] = dict(t=float(g("t")) if g("t") else np.nan, ff=float(g("ff_val")) if g("ff_val") else np.nan)
        except Exception:  # noqa: BLE001
            out[sid] = dict(t=np.nan, ff=np.nan)
    return out


@st.cache_data(show_spinner="Nalagam arhiv ARSO ...")
def arhiv():
    use = ["time_utc", "station_code", "pm25", "pm10"]
    pm = pd.concat([pd.read_csv(PK / "01_meritve" / "arso_lj_pm_urno_2016_2026.csv.gz", usecols=use),            # do 31. 10. 2026
                    pd.read_csv(PK / "01_meritve" / "arso_lj_pm_urno_2026_11_dopolnitev.csv", usecols=use)],      # 1.–21. 11. 2026
                   ignore_index=True)
    pm["time_utc"] = pd.to_datetime(pm.time_utc, utc=True)
    pm = pm[pm.station_code.isin(STATIONS) & (pm.time_utc >= "2025-09-25")].drop_duplicates(["station_code", "time_utc"])
    iv = pd.read_csv(PK / "03_vreme_inverzija" / "inverzija_urno_2016_2026.csv",
                     usecols=["time_utc", "wind_speed_lj_bezigrad", "temp_lj_bezigrad", "dt_bezigrad_topol", "dt_bezigrad_krvavec"])
    iv["time_utc"] = pd.to_datetime(iv.time_utc, utc=True)
    return pm, iv[iv.time_utc >= "2025-09-25"].set_index("time_utc")


# ---------------------------------------------------------------- napoved
def napovej(pm25, t0, wind, dt_topol, dt_krv):
    """pm25: urna vrsta (UTC, začetek ure) do vključno t0. Vrne naslednjih 12 ur z oceno."""
    m = model()
    s = pm25.sort_index()
    s = s[~s.index.duplicated(keep="last")].reindex(pd.date_range(t0 - 30 * H, t0, freq="h"))
    win6, win24 = s.loc[t0 - 5 * H:t0], s.loc[t0 - 23 * H:t0]
    base = dict(pm_now=s.get(t0), pm_l1=s.get(t0 - H), pm_l3=s.get(t0 - 3 * H),
                pm_m6=win6.mean() if win6.count() >= 4 else np.nan, pm_m24=win24.mean() if win24.count() >= 18 else np.nan,
                pm_l24=s.get(t0 - 24 * H), wind_now=wind, hour0=t0.hour,
                dt_topol=dt_topol, dt_krvavec=dt_krv)
    X = pd.DataFrame([{**base, "h": h, "hour_t": (t0 + h * H).hour, "dow_t": (t0 + h * H).day_of_week,
                       "month": (t0 + h * H).month, "time_utc": t0 + h * H} for h in range(1, 13)])
    rel = m["model"].predict(X[m["features"]])
    lvl = np.expm1(m["level_model"].predict(X[m["features"]])).clip(min=0.5)
    X["rel"] = rel
    X["ocena_pm25"] = np.exp(np.log(lvl).mean() + rel - rel.mean())  # raven iz modela ravni, potek iz glavnega modela
    X["cas"] = X.time_utc.dt.tz_convert(TZ)
    X["ura"] = X.cas.dt.hour
    return X


def priporocilo(X, now_utc):
    """Najboljša ura v budnem času (07–23), samo ure, ki še niso minile."""
    c = X[(X.ura >= 7) & (X.ura <= 22) & (X.time_utc + H > now_utc)]
    if c.empty:
        c = X[X.time_utc + H > now_utc]
    best = c.loc[c.rel.idxmin()]
    return best, c


def ure(ts, ref=None, ob=False):
    txt = ("ob " if ob else "") + f"{ts:%H}:00–{(ts + H):%H}:00"
    if ref is not None and ts.date() != ref.date():
        txt = ("jutri " if (ts.date() - ref.date()).days == 1 else DNI[ts.weekday()] + " ") + txt
    return txt


def kratko(ts, ref):
    """'zdaj', 'ob 7:00', 'jutri ob 7:00'."""
    if ts.date() == ref.date() and ts.hour == ref.hour:
        return "zdaj"
    t = f"ob {ts.hour}:00"
    if ts.date() != ref.date():
        t = ("jutri " if (ts.date() - ref.date()).days == 1 else DNI[ts.weekday()] + " ") + t
    return t


def nasveti(razred, best_txt):
    """Ploščice: (ikona, naslov, odgovor, stopnja)."""
    okna = ("Odpri kadarkoli", "ok") if razred <= 2 else (f"Na kratko, predvidoma {best_txt}", "pazi")
    vadba = {1: ("Primerno", "ok"), 2: ("Primerno", "ok"), 3: ("Lažja vadba", "pazi"), 4: ("Raje notri", "ne"), 5: ("Raje notri", "ne")}[razred]
    otroci = {1: ("Brez omejitev", "ok"), 2: ("Brez omejitev", "ok"), 3: ("Krajši sprehod", "pazi"), 4: ("Raje notri", "ne"), 5: ("Ostanite notri", "ne")}[razred]
    obc = {1: ("Brez omejitev", "ok"), 2: ("Brez omejitev", "ok"), 3: ("Manj napora", "pazi"), 4: ("Ostanite notri", "ne"), 5: ("Ostanite notri", "ne")}[razred]
    return [("💨", "Prezračevanje", *okna), ("🏃", "Vadba zunaj", *vadba), ("👶", "Sprehod z otrokom", *otroci),
            ("🧓", "Starejši, astma, srce", *obc)]


# ---------------------------------------------------------------- izgled
st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 2rem;}
.hero {border-radius: 18px; padding: 22px 24px; display:flex; gap:22px; align-items:center; margin: 6px 0 14px 0;}
.hero .big {font-size: 64px; line-height: 1;}
.hero .val {font-size: 44px; font-weight: 800; line-height:1;}
.hero .unit {font-size: 15px; opacity:.8; font-weight:600;}
.hero .lab {font-size: 26px; font-weight: 800; margin-top:4px;}
.hero .txt {font-size: 15px; margin-top:6px;}
.tag {display:inline-block; font-size:11px; font-weight:700; letter-spacing:.04em; text-transform:uppercase;
      padding:2px 8px; border-radius:20px; background:rgba(0,0,0,.08); margin-bottom:6px;}
.strip {display:flex; gap:5px; padding: 4px 2px 10px 2px;}
.chip {flex:1 1 0; min-width:0; text-align:center; border-radius:12px; padding:8px 4px; font-size:13px; border:2px solid transparent;}
.chip .h {font-weight:700; color:#33404f; white-space:nowrap;}
.chip .v {font-weight:800; font-size:16px; margin-top:4px;}
.chip .i {font-size:16px; height:20px;}
.chip.best {border-color:#1e7a3c; box-shadow:0 0 0 2px #bfe5cb;}
.chip.night {opacity:.45;}
.tiles {display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin: 6px 0 14px 0;}
.tile {border-radius:14px; padding:12px; background:#fff; border:1px solid #e6e9ee; text-align:center;}
.tile .ic {font-size:28px;}
.tile .t {font-size:13px; color:#56606e; margin-top:4px;}
.tile .r {font-size:14px; font-weight:800; margin-top:4px;}
.tile.ok .r {color:#1e7a3c;} .tile.pazi .r {color:#a26b00;} .tile.ne .r {color:#b4152b;}
.facts {display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-bottom: 10px;}
.fact {border-radius:14px; padding:10px 12px; background:#fff; border:1px solid #e6e9ee;}
.fact .k {font-size:12px; color:#56606e;} .fact .v {font-size:18px; font-weight:800; color:#14213d;}
.fact .d {font-size:12px; color:#56606e;}
.small {font-size:12.5px; color:#6b7480;}
@media (max-width: 640px) {.tiles {grid-template-columns: repeat(2, 1fr);} .hero .big{font-size:48px} .hero .val{font-size:36px}}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- izbira
st.markdown("## 🏠 Kdaj odpreti okno?")
cs = cetrti()
izbor = st.selectbox("📍 Kje si?", cs.ime, index=int(np.flatnonzero(cs.ime == "Vič")[0]) if (cs.ime == "Vič").any() else 0)
lat, lon = cs.set_index("ime").loc[izbor, ["lat", "lon"]]
station = najblizja(lat, lon)
with st.sidebar:
    st.header("Demo")
    nacin = st.radio("Podatki", ["V živo", "Pretekli dan (preverjanje)"],
                     help="'Pretekli dan' uporabi arhiv ARSO iz zime 2025/26, ki je model ni videl, in pokaže, kaj se je res zgodilo.")
    if nacin != "V živo":
        dan = st.date_input("Dan", pd.Timestamp("2026-01-20"), min_value=pd.Timestamp("2025-10-02"), max_value=pd.Timestamp("2026-11-12"),
                            help="Do 12. 11. 2026 (zadnji dan s podatki o inverziji v paketu). Model je za kurilno sezono (okt.–mar.).")
        ura0 = st.slider("Ura (lokalno)", 0, 23, 17)
ime_postaje, slat, slon = STATIONS[station]
dist = haversine_km(lat, lon, slat, slon)

# ---------------------------------------------------------------- podatki
dejansko = None
try:
    if nacin == "V živo":
        with st.spinner("Pridobivam zadnje meritve ARSO ..."):
            pmdf = live_pm()
            met = live_meteo()
        s25 = pmdf[pmdf.station_code == station].set_index("time_utc").pm25.dropna()
        s10 = pmdf[pmdf.station_code == station].set_index("time_utc").pm10
        t0 = s25.index.max()
        tb, tt, tk = met["LJUBL-ANA_BEZIGRAD"]["t"], met["TOPOL"]["t"], met["KRVAVEC"]["t"]
        wind, dt_topol, dt_krv = met["LJUBL-ANA_BEZIGRAD"]["ff"], tb - tt, tb - tk
        now_utc = pd.Timestamp.now(tz="UTC")
    else:
        pm, iv = arhiv()
        t0 = (pd.Timestamp(dan).tz_localize(TZ) + ura0 * H).tz_convert("UTC")
        p = pm[pm.station_code == station].set_index("time_utc")
        s25_all, s10 = p.pm25.dropna(), p.pm10
        s25 = s25_all[s25_all.index <= t0]
        r = iv.reindex([t0]).iloc[0]
        tb, wind, dt_topol, dt_krv = r.temp_lj_bezigrad, r.wind_speed_lj_bezigrad, r.dt_bezigrad_topol, r.dt_bezigrad_krvavec
        now_utc = t0 + H
        dejansko = s25_all.reindex(pd.date_range(t0 + H, t0 + 12 * H, freq="h"))
        if t0 not in s25.index:
            st.warning("Za to uro ni meritve ARSO. Izberi drugo uro.")
            st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Podatkov ni bilo mogoče pridobiti ({type(e).__name__}). Preveri internet ali v stranski vrstici izberi 'Pretekli dan'.")
    st.stop()

X = napovej(s25, t0, wind, dt_topol, dt_krv)
best, cand = priporocilo(X, now_utc)
pm_now = float(s25.loc[t0])
pm10_now = s10.get(t0, np.nan)
razred = int(razred_iqz(pm25=pm_now, pm10=None if pd.isna(pm10_now) else float(pm10_now)))
inverzija = (not np.isnan(dt_topol)) and dt_topol < 0
ref_loc = now_utc.tz_convert(TZ) if nacin == "V živo" else t0.tz_convert(TZ)
best_txt = ure(best.cas, ref_loc, ob=True)
t_meritve = t0.tz_convert(TZ)

# ---------------------------------------------------------------- GLAVNI ODGOVOR
kdaj = kratko(best.cas, ref_loc)
if razred <= 2:
    luc, barva, ozadje, naslov = "🟢", "#1e7a3c", "#e8f6ec", "OKNO LAHKO ODPREŠ"
    stavek1 = "Zrak je zdaj čist."
    stavek2 = "Po napovedi je zdaj tudi najčistejši." if kdaj == "zdaj" else f"Po napovedi bo najčistejši {kdaj}."
elif razred == 3:
    luc, barva, ozadje, naslov = "🟡", "#8a6d00", "#fdf6d8", "ODPRI LE NA KRATKO"
    stavek1 = "Zrak je zdaj zmerno onesnažen."
    stavek2 = "Po napovedi bo v naslednjih urah podoben ali slabši." if kdaj == "zdaj" else f"Po napovedi bo čistejši {kdaj}."
else:
    luc, barva, ozadje, naslov = "🔴", "#9b0d23", "#fbe3e6", "OKNO ZAPRI"
    stavek1 = "Zrak je zdaj onesnažen." if razred == 4 else "Zrak je zdaj zelo onesnažen."
    stavek2 = ("Po napovedi bo v naslednjih urah še slabši; če moraš, prezrači zdaj na kratko." if kdaj == "zdaj"
               else f"Po napovedi bo čistejši {kdaj}. Takrat prezrači za 5–10 minut.")

st.markdown(f"""
<div style="background:{ozadje}; border-radius:24px; padding:42px 24px; text-align:center; margin:10px 0 18px 0;">
  <div style="font-size:72px; line-height:1;">{luc}</div>
  <div style="font-size:40px; font-weight:900; color:{barva}; margin-top:14px; letter-spacing:.01em;">{naslov}</div>
  <div style="font-size:22px; color:#14213d; margin-top:18px;">{stavek1}</div>
  <div style="font-size:22px; color:#14213d; font-weight:700; margin-top:4px;">{stavek2}</div>
</div>""", unsafe_allow_html=True)
st.markdown(f"<div class='small' style='text-align:center'>Podatki: uradna postaja ARSO <b>{ime_postaje}</b>, {dist:.1f} km od središča četrti "
            f"{izbor}. V Ljubljani so tri uradne postaje, zato si več četrti deli isto postajo.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- preverjanje (samo pretekli dan)
if dejansko is not None and dejansko.notna().sum() >= 4:
    dd = pd.DataFrame({"cas": X.cas, "y": dejansko.values}).set_index("cas").loc[cand.cas].dropna()
    if len(dd) >= 3:
        ok = best.cas in dd.y.nsmallest(3).index
        (st.success if ok else st.warning)(
            f"**Kaj se je res zgodilo:** predlagana ura {'je bila' if ok else 'ni bila'} med tremi najčistejšimi. "
            f"Ob njej je bilo izmerjeno {dd.y.get(best.cas, np.nan):.0f} µg/m³, v povprečju budnih ur {dd.y.mean():.0f} µg/m³; "
            f"najčistejša je bila ura {ure(dd.y.idxmin(), ref_loc)} ({dd.y.min():.0f} µg/m³).")

# ---------------------------------------------------------------- podrobnosti
with st.expander("Več podrobnosti"):
    st.markdown(f"""
    <div class="hero" style="background:{SVETLE[razred]}; color:{TEMNE[razred]}">
      <div class="big">{OBRAZ[razred]}</div>
      <div>
        <span class="tag">izmerjeno · {ime_postaje} · {t_meritve:%H}:00–{(t_meritve + H):%H}:00</span>
        <div><span class="val">{pm_now:.0f}</span> <span class="unit">µg/m³ PM2.5</span></div>
        <div class="lab">{RAZRED_IME[razred].capitalize()}</div>
        <div class="txt">{POMEN[razred]}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    chips = []
    for _, r_ in X.iterrows():
        if r_.time_utc + H <= now_utc:
            continue
        k = int(razred_iqz(pm25=float(r_.ocena_pm25)))
        noc = not (7 <= int(r_.ura) <= 22)
        je_best = r_.time_utc == best.time_utc
        ikona = "✅" if je_best else ("🌙" if noc else "")
        chips.append(f"<div class='chip{' best' if je_best else ''}{' night' if noc else ''}' style='background:{SVETLE[k]}'>"
                     f"<div class='h'>{int(r_.ura)}h</div><div class='i'>{ikona}</div>"
                     f"<div class='v' style='color:{TEMNE[k]}'>{r_.ocena_pm25:.0f}</div></div>")
    st.markdown("**Naslednje ure** <span class='small'>· napoved modela (ocena), PM2.5 v µg/m³</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='strip'>{''.join(chips)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>✅ najboljša ura za prezračevanje · 🌙 nočne ure (niso predlagane) · "
                "napoved pove predvsem, <i>kdaj</i> bo čisteje; vrednosti so približne.</div>", unsafe_allow_html=True)

    st.markdown("#### Priporočila zdaj")
    tiles = "".join(f"<div class='tile {lvl}'><div class='ic'>{ic}</div><div class='t'>{t}</div><div class='r'>{odg}</div></div>"
                    for ic, t, odg, lvl in nasveti(razred, best_txt))
    st.markdown(f"<div class='tiles'>{tiles}</div>", unsafe_allow_html=True)

    inv_txt = ("Hladen zrak je ujet v kotlini kot pod pokrovko, zato dim in izpuhi ostajajo pri tleh."
               if inverzija else "Zrak se lahko meša navzgor, onesnaženje se laže razredči.")
    vet = "brez vetra" if (np.isnan(wind) or wind < 1) else ("šibak veter" if wind < 3 else "veter")
    st.markdown(f"""
    <div class="facts">
      <div class="fact"><div class="k">Kotlina</div><div class="v">{'Inverzija' if inverzija else 'Brez inverzije'}</div><div class="d">{inv_txt}</div></div>
      <div class="fact"><div class="k">Veter (Bežigrad)</div><div class="v">{'–' if np.isnan(wind) else f'{wind:.1f} m/s'}</div><div class="d">{vet.capitalize()}{'; onesnaženje se ne odnaša' if vet == 'brez vetra' else ''}.</div></div>
      <div class="fact"><div class="k">Temperatura (Bežigrad)</div><div class="v">{'–' if np.isnan(tb) else f'{tb:.0f} °C'}</div><div class="d">Ob mrazu se pogosteje kuri.</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Kako se spreminja zrak")
    zad = s25.loc[t0 - 11 * H:t0].rename("pm25").reset_index().rename(columns={"index": "time_utc"})
    zad["cas"] = zad.time_utc.dt.tz_convert(TZ)
    zad["vrsta"] = "Izmerjeno"
    nap = pd.concat([zad.tail(1)[["cas", "pm25"]], X[["cas", "ocena_pm25"]].rename(columns={"ocena_pm25": "pm25"})])
    nap["vrsta"] = "Napoved"
    deli = [zad[["cas", "pm25", "vrsta"]], nap]
    if dejansko is not None:                                                 # pretekli dan: kaj je bilo res izmerjeno
        dj = dejansko.rename("pm25").reset_index().rename(columns={"index": "time_utc"})
        dj["cas"] = dj.time_utc.dt.tz_convert(TZ)
        dj = pd.concat([zad.tail(1)[["cas", "pm25"]], dj[["cas", "pm25"]]])
        dj["vrsta"] = "Res izmerjeno"
        deli.append(dj)
    L = pd.concat(deli, ignore_index=True)
    ymax = max(60.0, float(np.ceil(np.nanmax(L.pm25) / 10) * 10))          # fiksna skala 0–60 µg/m³ (razširi se le nad 60)
    ys = alt.Scale(domain=[0, ymax])
    xs = alt.Scale(domain=[L.cas.min(), L.cas.max()])
    bands = pd.DataFrame([dict(od=a_, do=min(b_, ymax), barva=SVETLE[k]) for a_, b_, k in MEJE_PM25 + [(120, 1000, 5)] if a_ < ymax])
    band = alt.Chart(bands).mark_rect().encode(y=alt.Y("od:Q", scale=ys), y2="do:Q", color=alt.Color("barva:N", scale=None, legend=None))
    line = alt.Chart(L).mark_line(strokeWidth=3).encode(
        x=alt.X("cas:T", title=None, scale=xs, axis=alt.Axis(format="%H:%M", tickCount=8, grid=False)),
        y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)", scale=ys, axis=alt.Axis(grid=False, values=list(range(0, int(ymax) + 1, 10)))),
        color=alt.Color("vrsta:N", title=None, scale=alt.Scale(domain=["Izmerjeno", "Napoved", "Res izmerjeno"],
                                                               range=["#1f4e79", "#8a8f98", "#d9480f"]), legend=None),
        strokeDash=alt.StrokeDash("vrsta:N", scale=alt.Scale(domain=["Izmerjeno", "Napoved", "Res izmerjeno"],
                                                              range=[[1, 0], [6, 4], [1, 0]]), legend=None),
        tooltip=[alt.Tooltip("cas:T", title="ura", format="%H:%M"), alt.Tooltip("pm25:Q", title="PM2.5", format=".0f"),
                 alt.Tooltip("vrsta:N", title="")])
    zdaj = pd.DataFrame({"cas": [t_meritve], "txt": ["ZDAJ"], "y": [ymax]})
    zdaj_rule = alt.Chart(zdaj).mark_rule(color="#14213d", strokeDash=[2, 3]).encode(x=alt.X("cas:T", scale=xs))
    zdaj_txt = alt.Chart(zdaj).mark_text(dy=8, dx=4, align="left", fontWeight="bold", fontSize=12, color="#14213d").encode(
        x=alt.X("cas:T", scale=xs), y=alt.Y("y:Q", scale=ys), text="txt:N")
    naj = pd.DataFrame({"cas": [best.cas], "pm25": [best.ocena_pm25], "txt": ["najbolje"]})
    naj_pt = alt.Chart(naj).mark_point(filled=True, size=220, color="#1e7a3c").encode(x=alt.X("cas:T", scale=xs), y=alt.Y("pm25:Q", scale=ys))
    naj_txt = alt.Chart(naj).mark_text(dy=-16, fontWeight="bold", fontSize=12, color="#1e7a3c").encode(
        x=alt.X("cas:T", scale=xs), y=alt.Y("pm25:Q", scale=ys), text="txt:N")
    # povprečna napaka napovedi po urah vnaprej (test zime 2025/26, vse tri postaje) — pas se z oddaljenostjo širi
    NAPAKA = {1: 2.6, 2: 3.3, 3: 3.8, 4: 4.3, 5: 4.7, 6: 5.0, 7: 5.3, 8: 5.5, 9: 5.7, 10: 5.9, 11: 6.0, 12: 6.2}
    pas = X[["cas", "ocena_pm25", "h"]].copy()
    napaka = pas.h.map(NAPAKA)
    pas["spodaj"] = (pas.ocena_pm25 - napaka).clip(lower=0)
    pas["zgoraj"] = pas.ocena_pm25 + napaka
    pas_area = alt.Chart(pas).mark_area(color="#8a8f98", opacity=0.25).encode(
        x=alt.X("cas:T", scale=xs), y=alt.Y("spodaj:Q", scale=ys), y2="zgoraj:Q")
    st.altair_chart((band + pas_area + line + zdaj_rule + zdaj_txt + naj_pt + naj_txt).properties(height=300).configure_view(strokeWidth=0),
                    width="stretch")
    def _crta(barva, crtkano=False):
        slog = "dashed" if crtkano else "solid"
        return f"<span style='display:inline-block;width:26px;border-top:3px {slog} {barva};vertical-align:middle;margin-right:6px'></span>"
    def _kvad(barva):
        return f"<span style='display:inline-block;width:14px;height:12px;background:{barva};border-radius:3px;vertical-align:middle;margin-right:6px'></span>"
    def _pika(barva):
        return f"<span style='display:inline-block;width:12px;height:12px;background:{barva};border-radius:50%;vertical-align:middle;margin-right:6px'></span>"
    leg = [(_crta("#1f4e79"), "Izmerjeno (ARSO)"), (_crta("#8a8f98", True), "Napoved"), (_kvad("rgba(138,143,152,.35)"), "običajna napaka (od ±3 µg/m³ čez 1 uro do ±6 µg/m³ čez 12 ur)")]
    if dejansko is not None:
        leg.append((_crta("#d9480f"), "Pozneje res izmerjeno"))
    leg.append((_pika("#1e7a3c"), "Predvidoma najčistejša ura"))
    ozadje_leg = "".join(f"<span style='margin-right:12px;white-space:nowrap'>{_kvad(SVETLE[k])}{RAZRED_IME[k]}</span>" for k in (1, 2, 3, 4, 5))
    st.markdown("<div style='font-size:13px;color:#33404f;line-height:2'>"
                + "".join(f"<span style='margin-right:16px;white-space:nowrap'>{s}{tx}</span>" for s, tx in leg)
                + f"<br><span style='color:#6b7480'>Ozadje:</span> {ozadje_leg}</div>", unsafe_allow_html=True)

st.markdown("<div class='small' style='margin-top:10px'>Vir: ARSO. Prototip za ADA Hack 2026. Ni zdravniški nasvet.</div>",
            unsafe_allow_html=True)
