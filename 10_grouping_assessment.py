"""Assessment of regional groupings: daily spending shortfalls of each city, groups formed by province, city cluster, demand ties, nationwide draws and distance beyond the rainfall system, East-West cooperation groups, the group shared-rain share, synchrony of group shortfalls (variance ratio), city-specific decoupling distances and the excess peak shortfall."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.stats import norm, spearmanr
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = r"E:\Extreme Rainfall and UnionPay Spending"
DATASET = os.path.join(ROOT, "Dataset")
EXTERNAL = os.path.join(ROOT, "External data")
INTERMEDIATE = os.path.join(ROOT, "Intermediate")


def group_loss_inputs():
    """Inputs for the group shortfall analysis: per-cent abnormal spending from a fixed-effect counterfactual in logs,
    event-adjusted daily loss matrices for the main specification and its variants, spending weights, and five ex-ante
    group types (city cluster, province, national random, demand partners, cross-rain-belt). Also writes the event-level
    comparison with the canonical current shock (G0prime_event_table.csv) and the rainfall dose response of the loss."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
    SEED = 20261110
    NMAX, ND = 20, 200
    warnings.filterwarnings("ignore")

    def strip(s):
        return s.astype(str).str.strip()

    def load_panel():
        P = pd.read_csv(os.path.join(INTERMEDIATE, "unionpay_city_daily_2017_2019.csv"),
                        encoding=ENC, dtype={"city_id": str})
        P["city_name"] = strip(P.city_name)
        P["date"] = pd.to_datetime(P.date)
        hosts = sorted(P.city_name.unique())
        dates = pd.DatetimeIndex(sorted(P.date.unique()))
        W = {c: P.pivot(index="date", columns="city_name", values=c).reindex(index=dates, columns=hosts).to_numpy(float)
             for c in ("merchant_place_spending", "local_spending", "inbound_spending")}
        cal = pd.read_csv(os.path.join(DATASET, "China_calendar_2017_2019", "China_calendar_2017_2019.csv"), encoding=ENC)
        cal["date"] = pd.to_datetime(cal.date)
        return hosts, dates, W, cal.set_index("date").reindex(dates)

    def load_catalogue(hosts):
        cat = pd.read_csv(os.path.join(INTERMEDIATE, "extreme_rainfall_events_2017_2019.csv"), encoding=ENC)
        cat["city_name"] = strip(cat.city_name)
        for c in ("city_event_start", "city_event_end"):
            cat[c] = pd.to_datetime(cat[c])
        return cat[cat.city_name.isin(hosts)].reset_index(drop=True)

    def day_mask(ev, hosts, dates, pre=0, post=0):
        """T x N: host-day within [start - pre, end + post] of any event in ev."""
        hix = {c: i for i, c in enumerate(hosts)}
        d0 = dates[0]
        T = len(dates)
        M = np.zeros((T, len(hosts)), bool)
        for c, s, e in zip(ev.city_name, ev.city_event_start, ev.city_event_end):
            a = max((s - d0).days - pre, 0)
            b = min((e - d0).days + post, T - 1)
            if b >= 0 and a <= T - 1:
                M[a:b + 1, hix[c]] = True
        return M

    def fe_codes(dates, cal, N, variant="ym"):
        T = len(dates)
        h = np.broadcast_to(np.arange(N)[None, :], (T, N))
        ym = ((dates.year - 2017) * 12 + dates.month - 1).to_numpy()[:, None]
        yr = (dates.year - 2017).to_numpy()[:, None]
        mo = (dates.month - 1).to_numpy()[:, None]
        dow = dates.dayofweek.to_numpy()[:, None]
        hol = cal.is_public_holiday.to_numpy(int)[:, None]
        adj = cal.is_adjusted_workday.to_numpy(int)[:, None]
        dt = np.broadcast_to(np.arange(T)[:, None], (T, N))
        first = [h * 36 + ym] if variant == "ym" else [h * 3 + yr, h * 12 + mo]
        return first + [h * 7 + dow, h * 2 + hol, h * 2 + adj, dt]

    def fit_fe(Y, train, codes, tol=1e-9, maxit=20000):
        """Backfitting OLS with additive fixed effects on training cells; NaN where a level has no training cell."""
        y = Y[train]
        cs = [c[train] for c in codes]
        nl = [int(c.max()) + 1 for c in codes]
        cnt = [np.bincount(c, minlength=k).astype(float) for c, k in zip(cs, nl)]
        fe = [np.zeros(k) for k in nl]
        r = y - sum(f[c] for f, c in zip(fe, cs))
        for it in range(maxit):
            mx = 0.0
            for j, (c, k) in enumerate(zip(cs, nl)):
                adj = np.divide(np.bincount(c, weights=r, minlength=k), cnt[j], out=np.zeros(k), where=cnt[j] > 0)
                fe[j] += adj
                r -= adj[c]
                mx = max(mx, float(np.abs(adj).max()))
            if mx < tol:
                break
        sup = np.ones(Y.shape, bool)
        yhat = np.zeros(Y.shape)
        for f, c, k_cnt in zip(fe, codes, cnt):
            sup &= k_cnt[c] > 0
            yhat += f[c]
        yhat[~sup] = np.nan
        return yhat, sup, it + 1, r

    def codes_for(kind, dates, cal, N):
        T = len(dates)
        h = np.broadcast_to(np.arange(N)[None, :], (T, N))
        dt = np.broadcast_to(np.arange(T)[:, None], (T, N))
        mo = (dates.month - 1).to_numpy()[:, None]
        if kind == "moy":
            return [h * 12 + mo, dt]
        if kind == "host":
            return [h + 0 * mo, dt]
        if kind == "ym":
            return fe_codes(dates, cal, N, "ym")
        if kind == "ym_fallback":
            return fe_codes(dates, cal, N, "fallback")
        raise ValueError(kind)

    def abn_pct(Wlev, ev_hold, hosts, dates, cal, spec="main"):
        """Per-cent abnormal spending a = 100 (Y / exp(yhat) - 1); cells without support are filled along a fallback chain.
        main: city x month-of-year + date FE, 28-day hold-out; short_holdout: same with [-3, +7] hold-out; calendar_fe: calendar FE, [-3, +7]."""
        Ylog = 100 * np.log(Wlev)
        N = Ylog.shape[1]
        hold28 = ~day_mask(ev_hold, hosts, dates, 28, 28)
        hold37 = ~day_mask(ev_hold, hosts, dates, 3, 7)
        chain = {"main": [("moy", hold28), ("moy", hold37), ("host", hold37)], "short_holdout": [("moy", hold37), ("host", hold37)],
                 "calendar_fe": [("ym", hold37), ("ym_fallback", hold37)]}[spec]
        yhat = np.full(Ylog.shape, np.nan)
        src = np.full(Ylog.shape, -1, int)
        for lvl, (kind, train) in enumerate(chain):
            yh, sup, it, r = fit_fe(Ylog, train, codes_for(kind, dates, cal, N))
            fill = np.isnan(yhat) & sup
            yhat[fill] = yh[fill]
            src[fill] = lvl
            if not np.isnan(yhat).any():
                break
        return 100 * (np.exp((Ylog - yhat) / 100) - 1), src

    def adjusted_loss(a, ev, expo_any, hosts, dates, src=None):
        """L = -(a - baseline) on each event's exposure days, 0 elsewhere. Baseline: mean a over the clean pre-event days
        (14 days, >= 7 clean), else the host's clean days of that year-month, else of that year."""
        hix = {c: i for i, c in enumerate(hosts)}
        d0 = dates[0]
        ymid = ((dates.year - 2017) * 12 + dates.month - 1).to_numpy()
        yid = (dates.year - 2017).to_numpy()
        L = np.zeros(a.shape)
        rows = []
        for r in ev.itertuples(index=False):
            n = hix[r.city_name]
            s = (r.city_event_start - d0).days
            e = (r.city_event_end - d0).days
            pre = np.arange(max(s - 14, 0), s)
            pre = pre[~expo_any[pre, n]]
            if len(pre) >= 7:
                base, kind = a[pre, n].mean(), "event pre-window"
            else:
                d = np.where((ymid == ymid[s]) & ~expo_any[:, n])[0]
                kind = "host year-month"
                if len(d) == 0:
                    d = np.where((yid == yid[s]) & ~expo_any[:, n])[0]
                    kind = "host year"
                base = a[d, n].mean()
            L[s:e + 1, n] = -(a[s:e + 1, n] - base)
            rows.append(dict(city_event_id=r.city_event_id, city_name=r.city_name, meteorological_event_id=getattr(r, "meteorological_event_id", ""),
                             start=s, end=e, n_pre=len(pre), baseline_kind=kind, baseline=base, exposure_mean_a=a[s:e + 1, n].mean(),
                             shock_adjusted=a[s:e + 1, n].mean() - base, sum_L=L[s:e + 1, n].sum(),
                             n_fallback_cells=int((src[s:e + 1, n] > 0).sum()) if src is not None else 0))
        return L, pd.DataFrame(rows)

    def hav(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        x = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(x))

    def polygonal_only(geom):
        if geom is None or geom.is_empty:
            return geom
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom
        if isinstance(geom, GeometryCollection):
            parts = [polygonal_only(x) for x in geom.geoms]
            parts = [p for p in parts if p is not None and not p.is_empty]
            return unary_union(parts) if parts else Polygon()
        return Polygon()

    def centroids():
        g = gpd.read_file(os.path.join(DATASET, "GIS", "Prefecture City.shp"))
        if int((~g.geometry.is_valid).sum()):
            rep = shapely.make_valid(g.geometry.array)
            g.geometry = [polygonal_only(x) for x in rep]
        g = gpd.GeoDataFrame({"city_name": g.city_name.astype("string").str.strip(), "geometry": g.geometry},
                             crs=g.crs).dissolve(by="city_name", as_index=False).to_crs(4326)
        pr = g.to_crs("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs")
        cen = gpd.GeoSeries(pr.geometry.centroid, crs=pr.crs).to_crs(4326)
        return pd.DataFrame({"city_name": g.city_name.astype(str).values, "lon": cen.x.values, "lat": cen.y.values}).set_index("city_name")

    def ibtracs_fixes():
        ib = pd.read_csv(os.path.join(EXTERNAL, "IBTrACS", "ibtracs.WP.list.v04r01.csv"), skiprows=[1], low_memory=False,
                         usecols=["SID", "SEASON", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "TRACK_TYPE", "USA_SSHS"])
        ib["t"] = pd.to_datetime(ib.ISO_TIME) + pd.Timedelta(hours=8)
        ib = ib[(ib.t >= "2016-11-01") & (ib.t < "2020-02-01") & (ib.TRACK_TYPE == "main")].copy()
        ib["bjt_date"] = pd.to_datetime(ib.t.dt.strftime("%Y-%m-%d"))
        ib["LAT"] = pd.to_numeric(ib.LAT)
        ib["LON"] = pd.to_numeric(ib.LON)
        return ib.reset_index(drop=True)

    hosts225, dates, W, cal = load_panel()
    cat = load_catalogue(hosts225)
    T = len(dates)
    expo225 = day_mask(cat, hosts225, dates, 0, 0)

    # Event-level comparison of the adjusted shock with the canonical current shock (merchant-place spending)
    a, src = abn_pct(W["merchant_place_spending"], cat, hosts225, dates, cal, "main")
    L, et = adjusted_loss(a, cat, expo225, hosts225, dates, src)
    ev = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, dtype={"city_id": str})
    g = et.merge(ev[["city_event_id", "CurrentShock_merchant", "meteorological_event_id"]].rename(columns={"meteorological_event_id": "sys_can"}),
                 on="city_event_id", how="left")
    g["canonical643"] = g.CurrentShock_merchant.notna()
    g.to_csv(os.path.join(INTERMEDIATE, "G0prime_event_table.csv"), index=False, encoding=ENC)

    # Loss matrices (hosts with weather data)
    units = set(strip(pd.read_csv(os.path.join(WEATHER, "city_daily_weather_2017.csv"), encoding=ENC, usecols=["city_name"]).city_name))
    keep = np.array([h in units for h in hosts225])
    hosts = [h for h in hosts225 if h in units]
    N = len(hosts)
    hix = {c: i for i, c in enumerate(hosts)}

    def build(series, ev_expo, ev_hold, spec, expo_any=None):
        a, src = abn_pct(W[series], ev_hold, hosts225, dates, cal, spec)
        ea = day_mask(ev_expo, hosts225, dates, 0, 0) if expo_any is None else expo_any
        L, et = adjusted_loss(a, ev_expo, ea, hosts225, dates, src)
        return a, L, et

    LOSS, EXPO = {}, {}
    A_MAIN, L, ET = a, L, et.copy()
    LOSS["main"] = L[:, keep]
    EXPO["main"] = expo225[:, keep]
    for spec in ("calendar_fe", "short_holdout"):
        _, L2, _ = build("merchant_place_spending", cat, cat, spec)
        LOSS[spec] = L2[:, keep]
        EXPO[spec] = EXPO["main"]
    for ser, nm in (("local_spending", "local"), ("inbound_spending", "inbound")):
        _, L2, _ = build(ser, cat, cat, "main")
        LOSS[nm] = L2[:, keep]
        EXPO[nm] = EXPO["main"]
    LOSS["all"] = -A_MAIN[:, keep]  # -a on every day, no event adjustment
    EXPO["all"] = EXPO["main"]

    # Fitted loss: dose response of abnormal spending on own same-day rainfall
    wx = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC, usecols=["city_name", "date", "rain_mm"])
                    for y in (2017, 2018, 2019)])
    wx["city_name"] = strip(wx.city_name)
    wx["date"] = pd.to_datetime(wx.date)
    RAIN = wx[wx.city_name.isin(hosts)].pivot(index="date", columns="city_name", values="rain_mm").reindex(index=dates, columns=hosts).to_numpy(float)
    a224 = A_MAIN[:, keep]
    E = EXPO["main"]
    rb = np.where(RAIN <= 0, 0, np.where(RAIN <= 5, 1, np.where(RAIN <= 10, 2, np.where(RAIN <= 25, 3, np.where(RAIN < 50, 4, np.where(
        RAIN < 60, 5, np.where(RAIN < 75, 6, np.where(RAIN < 100, 7, np.where(RAIN < 150, 8, 9)))))))))
    base = a224[~E].mean()
    dr = np.array([a224[rb == b].mean() - base if (rb == b).any() else np.nan for b in range(10)])
    LOSS["fit"] = np.where(E, -dr[rb], 0.0)
    EXPO["fit"] = E
    pd.DataFrame(dict(rain_bin=["0", "(0,5]", "(5,10]", "(10,25]", "(25,50)", "[50,60)", "[60,75)", "[75,100)", "[100,150)", ">=150"],
                      n_host_days=[int((rb == b).sum()) for b in range(10)], n_exposure_days=[int(((rb == b) & E).sum()) for b in range(10)],
                      dose_response_pct=dr)).to_csv(os.path.join(INTERMEDIATE, "T4_Lfit_dose_response.csv"), index=False, encoding=ENC)

    # Tropical-cyclone flag for every host event: nearest IBTrACS fix within [start - 1, end + 1] days of the city centroid
    ib = ibtracs_fixes()
    cen = centroids()
    tc = []
    for r in cat.itertuples(index=False):
        f = ib[(ib.bjt_date >= r.city_event_start - pd.Timedelta(days=1)) & (ib.bjt_date <= r.city_event_end + pd.Timedelta(days=1))]
        if len(f):
            d = hav(cen.loc[r.city_name, "lat"], cen.loc[r.city_name, "lon"], f.LAT.to_numpy(), f.LON.to_numpy())
            k = int(np.argmin(d))
            tc.append((float(d[k]), f.NAME.iloc[k]))
        else:
            tc.append((np.nan, ""))
    cat["TCdist_km"] = [x[0] for x in tc]
    cat["TCname"] = [x[1] for x in tc]
    cat["TC500"] = (cat.TCdist_km <= 500).astype(int)
    cat["canonical643"] = cat.city_event_id.isin(set(ev.city_event_id))
    ET = ET.merge(cat[["city_event_id", "TC500", "TCname", "TCdist_km", "canonical643", "city_event_start", "city_event_end", "peak_rain_mm"]], on="city_event_id")

    def subset_loss(evsub):
        msk = day_mask(evsub, hosts, dates, 0, 0)
        return np.where(msk, LOSS["main"], 0.0), msk

    LOSS["canon643"], EXPO["canon643"] = subset_loss(cat[cat.canonical643])
    LOSS["noTC"], EXPO["noTC"] = subset_loss(cat[cat.TC500 == 0])
    sysloss = ET.groupby("meteorological_event_id").agg(sum_L=("sum_L", "sum")).sort_values("sum_L", ascending=False)
    top3 = list(sysloss.index[:3])
    LOSS["noTop3"], EXPO["noTop3"] = subset_loss(cat[~cat.meteorological_event_id.isin(top3)])

    # CPC-defined events: CPC rain >= the CPC threshold with the ERA5-Land 1981-2016 extreme-day frequency
    era = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC, usecols=["city_name", "date", "rain_mm"])
                     for y in range(1981, 2017)], ignore_index=True)
    era["city_name"] = era.city_name.str.strip()
    rl = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC).set_index("city_name").return_level_3yr_mm
    era["ext"] = (era.rain_mm >= 50) | (era.rain_mm >= era.city_name.map(rl).fillna(np.inf))
    freq = era.groupby("city_name").ext.mean()
    cp = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_cpc_1981_2019.parquet"))
    cp["city_name"] = cp.city_name.astype(str).str.strip()
    cp["date"] = pd.to_datetime(cp.date)
    xh = cp[(cp.date < "2017-01-01")].dropna(subset=["rain_mm"])
    th = pd.Series({c: s.quantile(1 - freq[c]) for c, s in xh.groupby("city_name").rain_mm if c in freq.index})
    cp = cp[cp.date >= "2016-12-01"]
    CPC = cp[cp.city_name.isin(hosts225)].pivot(index="date", columns="city_name", values="rain_mm").reindex(index=dates, columns=hosts225)
    thr = np.array([th.get(h, np.nan) for h in hosts225])
    XC = (CPC.to_numpy() >= thr[None, :]) & np.isfinite(thr)[None, :] & (thr[None, :] > 0)
    recs = []
    for j, h in enumerate(hosts225):
        d = np.where(XC[:, j])[0]
        if len(d) == 0:
            continue
        for run in np.split(d, np.where(np.diff(d) != 1)[0] + 1):
            recs.append(dict(city_event_id=f"CPC_{h}_{run[0]}", city_name=h, city_event_start=dates[run[0]], city_event_end=dates[run[-1]], meteorological_event_id=""))
    cpc_ev = pd.DataFrame(recs)
    both = pd.concat([cat[["city_name", "city_event_start", "city_event_end"]], cpc_ev[["city_name", "city_event_start", "city_event_end"]]])
    expo_cpc = day_mask(cpc_ev, hosts225, dates, 0, 0)
    a_c, src_c = abn_pct(W["merchant_place_spending"], both, hosts225, dates, cal, "main")
    L_c, _ = adjusted_loss(a_c, cpc_ev, expo_cpc | expo225, hosts225, dates, src_c)
    LOSS["cpc"] = L_c[:, keep]
    EXPO["cpc"] = expo_cpc[:, keep]

    # Spending weights: mean 2017 merchant-place spending on days outside [-3, +7] event windows
    clean37 = ~day_mask(cat, hosts225, dates, 3, 7)
    y17 = (dates.year == 2017)[:, None]
    WSP = np.nanmean(np.where(clean37 & y17, W["merchant_place_spending"], np.nan), axis=0)[keep]

    # Groups
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC, usecols=["focal_city", "partner_city", "distance_km", "pi"])
    hp = hp[hp.focal_city.isin(hosts) & hp.partner_city.isin(hosts)]
    PI = np.full((N, N), np.nan)
    DIST = np.zeros((N, N))
    fi = hp.focal_city.map(hix).to_numpy()
    pj = hp.partner_city.map(hix).to_numpy()
    PI[fi, pj] = hp.pi.to_numpy()
    DIST[fi, pj] = hp.distance_km.to_numpy()
    np.fill_diagonal(PI, np.nan)
    PS = np.where(np.isfinite(PI) & np.isfinite(PI.T), (PI + PI.T) / 2, np.where(np.isfinite(PI), PI, np.where(np.isfinite(PI.T), PI.T, 0.0)))
    attrs = pd.read_csv(os.path.join(EXTERNAL, "city_attributes.csv"), encoding=ENC).drop_duplicates("city_name")
    prov = dict(zip(strip(attrs.city_name), attrs.province))
    cl = pd.read_csv(os.path.join(INTERMEDIATE, "city_clusters_19.csv"), encoding=ENC)
    cl = cl[cl.unionpay_host == 1]
    cl["city_name"] = strip(cl.city_name)
    pn = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC, usecols=["host_name", "guest_name", "network_year", "w"])
    pn = pn[pn.network_year == 2018].sort_values(["host_name", "w"], ascending=[True, False])

    def build_pools(univ):
        """univ: sorted host indices allowed as focal hosts and members."""
        us = set(univ)
        P = {}

        def random_type(groups, code):
            rng = np.random.default_rng([SEED, code])
            Ms, rows = [], []
            for i, g, lab in groups:
                oth = np.array([m for m in g if m != i])
                k = min(len(oth), NMAX - 1)
                order = np.argsort(rng.random((ND, len(oth))), axis=1)[:, :k]
                M = np.full((ND, NMAX), -1, int)
                M[:, 0] = i
                M[:, 1:k + 1] = oth[order]
                Ms.append(M)
                rows.append((i, lab, k + 1))
            return dict(M=np.vstack(Ms), focal=np.array([r[0] for r in rows]), label=[r[1] for r in rows], size=np.array([r[2] for r in rows]), draws=ND)

        memb = {c: sorted(hix[x] for x in g.city_name if x in hix and hix[x] in us) for c, g in cl.groupby("cluster_en")}
        P["cluster"] = random_type([(i, m, c) for c, m in memb.items() if len(m) >= 2 for i in m], 1)
        pm = {}
        for i in univ:
            pm.setdefault(prov[hosts[i]], []).append(i)
        P["province"] = random_type([(i, m, p) for p, m in pm.items() if len(m) >= 2 for i in m], 2)
        P["random"] = random_type([(i, univ, "all") for i in univ], 3)
        Ms, sz = [], []
        for i in univ:  # demand partners: 2018 inbound partners that are hosts, by weight
            g = [hix[x] for x in pn.loc[pn.host_name == hosts[i], "guest_name"] if x in hix and hix[x] in us and hix[x] != i][:NMAX - 1]
            M = np.full(NMAX, -1, int)
            M[0] = i
            M[1:len(g) + 1] = g
            Ms.append(M)
            sz.append(len(g) + 1)
        P["demand"] = dict(M=np.array(Ms), focal=np.array(univ), label=["demand"] * len(univ), size=np.array(sz), draws=1)
        Ms, nfb, first_fb = [], 0, []
        for i in univ:  # cross-rain-belt: greedy, each new member >= 500 km from all members and least co-exposed
            mem = [i]
            ffb = NMAX + 1
            while len(mem) < NMAX:
                cand = [k for k in univ if k not in mem and all(DIST[k, m] >= 500 for m in mem)]
                if not cand:
                    cand = [k for k in univ if k not in mem]
                    nfb += 1
                    ffb = min(ffb, len(mem) + 1)
                sc = np.array([PS[k, mem].mean() for k in cand])
                mem.append(cand[int(np.argmin(sc))])
            Ms.append(mem)
            first_fb.append(ffb)
        P["crossbelt"] = dict(M=np.array(Ms), focal=np.array(univ), label=["crossbelt"] * len(univ), size=np.full(len(univ), NMAX), draws=1,
                              first_fallback_n=np.array(first_fb), n_fallback_steps=nfb)
        P["whole"] = {c: np.array(m) for c, m in memb.items() if len(m) >= 10}
        P["members"] = memb
        P["univ"] = np.array(univ)
        return P

    POOLS = build_pools(list(range(N)))
    prd = {hix[x] for x in pd.read_csv(os.path.join(EXTERNAL, "pearl_river_delta_cities.csv"), encoding=ENC).city_name}
    POOLS_NOPRD = build_pools([i for i in range(N) if i not in prd])
    ymid = ((dates.year - 2017) * 12 + dates.month - 1).to_numpy()
    pickle.dump(dict(hosts=hosts, dates=dates, LOSS=LOSS, EXPO=EXPO, WSP=WSP, POOLS=POOLS, POOLS_NOPRD=POOLS_NOPRD, DIST=DIST, PI=PI, block=ymid,
                     year=dates.year.to_numpy(), NMAX=NMAX, prov=[prov[h] for h in hosts]),
                open(os.path.join(INTERMEDIATE, "t4_inputs.pkl"), "wb"))
    ET.to_csv(os.path.join(INTERMEDIATE, "T4_event_losses.csv"), index=False, encoding=ENC)
    print("hosts", N, "loss variants", sorted(LOSS))


def group_loss_tests(argv=None):
    """Group shortfall statistics: variance ratio VR, risk ratio RR, average pairwise correlation and co-loss shares by
    group type and group size, with year-month block bootstrap intervals and circular-shift permutation nulls; whole-cluster
    RR; the worst-1% tail ratio; distances at which pairwise co-loss and co-hit shares fall below 5%; and the five worst
    group days of the five largest city clusters."""
    ENC = "utf-8-sig"
    SEED = 20261110
    NB = 999
    TYPES = ["cluster", "province", "random", "demand", "crossbelt"]
    FYP5 = ["JJJ", "YRD", "PRD", "MYR", "CY"]  # the five clusters to be optimised and upgraded in the 14th Five-Year Plan
    LAB = {"cluster": "City cluster", "province": "Province", "demand": "Demand partners", "random": "National random", "crossbelt": "Cross-rain-belt"}
    # name: (shortfall key, groups key, spending weights, n main, n max, size curve, permutation n max)
    CONFIGS = {"main": ("main", "POOLS", False, 5, 20, True, 10), "spw": ("main", "POOLS", True, 5, 5, False, None),
               "n10": ("main", "POOLS", False, 10, 10, False, None)}
    warnings.filterwarnings("ignore")

    def load_inputs():
        return pickle.load(open(os.path.join(INTERMEDIATE, "t4_inputs.pkl"), "rb"))

    class Moments:
        """Per-block sufficient statistics of a loss matrix L (T x N) and exposure mask E (T x N)."""

        def __init__(self, L, E, block):
            ub = np.unique(block)
            bid = np.searchsorted(ub, block)
            B = len(ub)
            N = L.shape[1]
            Ef = E.astype(float)
            self.n = np.bincount(bid, minlength=B).astype(float)
            self.B = B
            self.S1 = np.zeros((B, N))
            self.S2 = np.zeros((B, N, N))
            self.G = np.zeros((B, N, N))
            self.H = np.zeros((B, N, N))
            for b in range(B):
                Lb = L[bid == b]
                Eb = Ef[bid == b]
                self.S1[b] = Lb.sum(0)
                self.S2[b] = Lb.T @ Lb
                self.G[b] = Eb.T @ Lb
                self.H[b] = Eb.T @ Eb

        def at(self, k=None):
            k = np.ones(self.B) if k is None else k
            n = k @ self.n
            m = (k @ self.S1) / n
            return np.tensordot(k, self.S2, 1) / n - np.outer(m, m), np.tensordot(k, self.G, 1), np.tensordot(k, self.H, 1)

    def prep(pool, N, nmax):
        M = pool["M"][:, :nmax]
        d = pool["draws"]
        size = np.repeat(np.minimum(pool["size"], nmax), d)
        M0 = np.where(M >= 0, M, 0)
        pa = np.array([a for b in range(1, nmax) for a in range(b)])
        pb = np.array([b for b in range(1, nmax) for a in range(b)])
        uf = np.unique(pool["focal"])
        RF = (pool["focal"][:, None] == uf[None, :]).astype(float)
        return dict(idx=(M0[:, pa] * N + M0[:, pb]).T.astype(np.int32).copy(), didx=(M0 * N + M0).T.astype(np.int32).copy(), M0T=M0.T.copy(),
                    pa=pa, pb=pb, size=size, nmax=nmax, P=len(M), R=len(pool["focal"]), draws=d, focal=uf, RF=RF)

    def nested(C, pp, w=None):
        """VR, RR and average correlation for groups made of the first 2..nmax members; arrays (nmax - 1, P)."""
        Cf = C.ravel()
        sd = np.sqrt(np.clip(np.diag(C), 0, None))
        nmax, P = pp["nmax"], pp["P"]
        X = Cf.take(pp["idx"])
        Dg = Cf.take(pp["didx"])
        SD = sd.take(pp["M0T"])
        if w is not None:
            wm = w.take(pp["M0T"])
            X = X * wm[pp["pa"]] * wm[pp["pb"]]
            Dg = Dg * wm ** 2
            SD = SD * wm
        S = Dg[0].copy()
        D = Dg[0].copy()
        Pf = SD[0].copy()
        pos = 0
        VR = np.full((nmax - 1, P), np.nan)
        RR = np.full((nmax - 1, P), np.nan)
        RHO = np.full((nmax - 1, P), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            for b in range(1, nmax):
                S = S + Dg[b] + 2 * X[pos:pos + b].sum(0)
                pos += b
                D = D + Dg[b]
                Pf = Pf + SD[b]
                ok = pp["size"] >= b + 1
                sP = np.sqrt(np.clip(S, 0, None))
                den = Pf ** 2 - D
                VR[b - 1] = np.where(ok & (D > 0), sP / np.sqrt(D), np.nan)
                RR[b - 1] = np.where(ok & (Pf > 0), sP / Pf, np.nan)
                RHO[b - 1] = np.where(ok & (Pf > 0) & (den > 1e-9 * Pf ** 2), (S - D) / den, np.nan)
        return VR, RR, RHO

    def to_focal(x, pp):
        """Group-level (nn, P) to focal-level (nn, F): mean over draws within focal x group row, then over the focal's rows."""
        nn = x.shape[0]
        xr = x.reshape(nn, pp["R"], pp["draws"])
        ok = np.isfinite(xr)
        cnt = ok.sum(2)
        rowm = np.where(cnt > 0, np.where(ok, xr, 0).sum(2) / np.maximum(cnt, 1), np.nan)
        okr = np.isfinite(rowm)
        num = np.where(okr, rowm, 0) @ pp["RF"]
        den = okr.astype(float) @ pp["RF"]
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    def ecl_es(G, H, pp, n, w=None):
        """Focal-level expected co-loss share (ECL) and realised co-exposure share (ES_real) for groups of size n."""
        i = pp["M0T"][0]
        mem = pp["M0T"][1:n]
        g = G[i[None, :], mem]
        h = H[i[None, :], mem]
        if w is None:
            gb, hb = g.mean(0), h.mean(0)
        else:
            wm = w[mem]
            gb, hb = (g * wm).sum(0) / wm.sum(0), (h * wm).sum(0) / wm.sum(0)
        ok = pp["size"] >= n
        gb = np.where(ok, gb, np.nan)
        hb = np.where(ok, hb, np.nan)
        Gb = to_focal(gb[None, :], pp)[0]
        Hb = to_focal(hb[None, :], pp)[0]
        f = pp["focal"]
        Gff, Hff = G[f, f], H[f, f]
        v = np.isfinite(Gb) & (Gff > 0) & (Hff > 0)
        return np.where(v, Gb / np.where(v, Gff, 1), np.nan), np.where(v, Hb / np.where(v, Hff, 1), np.nan), Gb, Hb, Gff, Hff, v

    def whole_rr(C, whole, w=None):
        sd = np.sqrt(np.clip(np.diag(C), 0, None))
        out = {}
        for c, m in whole.items():
            ww = np.ones(len(m)) if w is None else w[m]
            out[c] = float(np.sqrt(max(ww @ C[np.ix_(m, m)] @ ww, 0)) / (ww @ sd[m]))
        return out

    def med(x):
        return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan

    def mean(x):
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    def evaluate(C, G, H, PP, POOLS, w=None, nmain=5, ecl=True, curve=True):
        out, RRf = {}, {}
        for t in TYPES:
            pp = PP[t]
            VR, RR, RHO = nested(C, pp, w)
            F = np.sqrt(np.clip(RHO, 0, None))
            fo = {s: to_focal(x, pp) for s, x in (("VR", VR), ("RR", RR), ("rho", RHO), ("F", F))}
            ns = range(2, pp["nmax"] + 1) if curve else [nmain]
            for n in ns:
                for s in ("VR", "RR", "rho", "F"):
                    x = fo[s][n - 2]
                    out[f"{t}|{s}|med|{n}"] = med(x)
                    out[f"{t}|{s}|mean|{n}"] = mean(x)
                out[f"{t}|N|{n}"] = float(np.isfinite(fo["RR"][n - 2]).sum())
            RRf[t] = dict(zip(pp["focal"], fo["RR"][nmain - 2]))
            if ecl and G is not None:
                e, s_, Gb, Hb, Gff, Hff, v = ecl_es(G, H, pp, nmain, w)
                out[f"{t}|ECL|med"] = med(e)
                out[f"{t}|ECL|mean"] = mean(e)
                out[f"{t}|ES|med"] = med(s_)
                out[f"{t}|ES|mean"] = mean(s_)
                out[f"{t}|N_ecl"] = float(v.sum())
                out[f"{t}|ECLratio|med"] = out[f"{t}|ECL|med"] / out[f"{t}|ES|med"] if out[f"{t}|ES|med"] > 0 else np.nan
                out[f"{t}|ECLratio|mean"] = out[f"{t}|ECL|mean"] / out[f"{t}|ES|mean"] if out[f"{t}|ES|mean"] > 0 else np.nan
                pe = Gb[v].sum() / Gff[v].sum() if v.any() else np.nan
                ps = Hb[v].sum() / Hff[v].sum() if v.any() else np.nan
                out[f"{t}|ECL|pooled"] = float(pe)
                out[f"{t}|ES|pooled"] = float(ps)
                out[f"{t}|ECLratio|pooled"] = float(pe / ps) if ps and ps > 0 else np.nan
        out["K2|diff_own"] = out[f"crossbelt|RR|med|{nmain}"] - out[f"cluster|RR|med|{nmain}"]
        fc = PP["cluster"]["focal"]
        rc = np.array([RRf["cluster"][i] for i in fc])
        rb = np.array([RRf["crossbelt"].get(i, np.nan) for i in fc])
        cm = np.isfinite(rc) & np.isfinite(rb)
        out["K2|diff_common"] = med(rb[cm]) - med(rc[cm])
        out["K2|N_common"] = float(cm.sum())
        out["K2|diff_own_mean"] = out[f"crossbelt|RR|mean|{nmain}"] - out[f"cluster|RR|mean|{nmain}"]
        wr = whole_rr(C, POOLS["whole"], w)
        dif = {}
        for c, m in POOLS["whole"].items():
            out[f"whole|RR|{c}"] = wr[c]
            for i in m:
                dif.setdefault(i, []).append(wr[c])
        d = np.array([np.mean(v) - RRf["crossbelt"].get(i, np.nan) for i, v in dif.items()])
        out["K3|med"] = med(d)
        out["K3|mean"] = mean(d)
        out["K3|N"] = float(np.isfinite(d).sum())
        out["K3|cluster_mean"] = float(np.mean([wr[c] - np.nanmean([RRf["crossbelt"].get(i, np.nan) for i in m]) for c, m in POOLS["whole"].items()])) if POOLS["whole"] else np.nan
        out["K3|whole_RR_med_focal"] = med(np.array([np.mean(v) for v in dif.values()])) if dif else np.nan
        out["K3|cb_RR_med_focal"] = med(np.array([RRf["crossbelt"].get(i, np.nan) for i in dif])) if dif else np.nan
        return out

    def run_config(L, E, block, POOLS, w=None, nmain=5, nmax=None, KB=None, OFF=None, curve=False, perm_nmax=None):
        """Point estimate, block bootstrap (rows of KB) and circular-shift permutations (rows of OFF)."""
        N = L.shape[1]
        nmax = nmax or nmain
        M = Moments(L, E, block)
        PP = {t: prep(POOLS[t], N, nmax) for t in TYPES}
        C, G, H = M.at()
        pt = evaluate(C, G, H, PP, POOLS, w, nmain, True, curve)
        boot = []
        for k in KB:
            C, G, H = M.at(k.astype(float))
            boot.append(evaluate(C, G, H, PP, POOLS, w, nmain, True, curve))
        perm = []
        T = L.shape[0]
        ar = np.arange(T)[:, None]
        col = np.arange(N)[None, :]
        Lc = L - L.mean(0)
        PPp = PP if (perm_nmax is None or perm_nmax == nmax) else {t: prep(POOLS[t], N, perm_nmax) for t in TYPES}
        for off in OFF:
            Lp = Lc[(ar - off[None, :]) % T, col]
            Cp = Lp.T @ Lp / T
            o = evaluate(Cp, None, None, PPp, POOLS, w, nmain, False, perm_nmax is not None and perm_nmax > nmain)
            perm.append({k: v for k, v in o.items() if "|VR|" in k or "|RR|" in k or k.startswith("K")})
        return pt, pd.DataFrame(boot), pd.DataFrame(perm)

    def ci(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return (float(np.quantile(x, .025)), float(np.quantile(x, .975))) if len(x) else (np.nan, np.nan)

    def perm_p(obs, null):
        null = np.asarray(null, float)
        null = null[np.isfinite(null)]
        return float((1 + (null >= obs).sum()) / (len(null) + 1))

    def draws(T, N, B=36):
        KB = np.random.default_rng([SEED, 1, B]).multinomial(B, np.full(B, 1 / B), size=NB)
        OFF = np.random.default_rng([SEED, 2, T]).integers(30, T - 30 + 1, size=(NB, N))
        return KB, OFF

    def run_worker(name, work):
        lk, pk, spw, nmain, nmax, curve, pn = CONFIGS[name]
        I = load_inputs()
        L, E, block = I["LOSS"][lk], I["EXPO"][lk], I["block"]
        T, N = L.shape
        KB, OFF = draws(T, N, len(np.unique(block)))
        pt, boot, perm = run_config(L, E, block, I[pk], I["WSP"] if spw else None, nmain, nmax, KB, OFF, curve, pn)
        pt["_T"] = T
        pt["_B"] = int(len(np.unique(block)))
        pt["_nmain"] = nmain
        json.dump(pt, open(os.path.join(work, f"res_{name}_point.json"), "w"))
        boot.to_parquet(os.path.join(work, f"res_{name}_boot.parquet"))
        perm.to_parquet(os.path.join(work, f"res_{name}_perm.parquet"))

    def pava_dec(y, w):
        """Weighted non-increasing isotonic regression."""
        y = -np.asarray(y, float)
        w = np.asarray(w, float)
        vals, wts, cnt = [], [], []
        for yi, wi in zip(y, w):
            vals.append(yi)
            wts.append(wi)
            cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
                ww = w1 + w2
                vals.append((v1 * w1 + v2 * w2) / ww if ww > 0 else (v1 + v2) / 2)
                wts.append(ww)
                cnt.append(c1 + c2)
        return -np.repeat(vals, cnt)

    def Lcross(ctr, fit, c):
        """Distance (km) at which a non-increasing curve over log-distance bin centres first falls below c."""
        g = np.isfinite(fit)
        x, f = ctr[g], fit[g]
        k = np.where(f < c)[0]
        if len(k) == 0:
            return np.inf
        k = k[0]
        if k == 0:
            return 0.0
        return float(np.exp(x[k - 1] + (f[k - 1] - c) * (x[k] - x[k - 1]) / (f[k - 1] - f[k])))

    def tail_table(I):
        """Mean of the worst 1% of group-shortfall days relative to the members' mean worst-1% loss (n = 5, equal weights)."""
        L = I["LOSS"]["main"]
        T, N = L.shape
        block = I["block"]
        KB, _ = draws(T, N, 36)
        nblk = np.bincount(block).astype(float)

        def top_pos(X):
            npos = (X > 0).sum(0)
            K = max(int(npos.max()), 1)
            order = np.argsort(-X, axis=0)[:K]
            v = np.take_along_axis(X, order, 0)
            order = np.where(v > 0, order, T)
            v = np.where(v > 0, v, 0.0)
            return v.T.copy(), order.T.copy()

        def tail_from(v, di, mult, kb):
            m = mult[di]
            cum = np.cumsum(m, 1)
            take = np.clip(kb - (cum - m), 0, m)
            return (take * v).sum(1) / kb

        vm, dm = top_pos(L)
        PREP = {}
        for t in TYPES:
            pp = prep(I["POOLS"][t], N, 5)
            M = pp["M0T"].T
            ok = pp["size"] >= 5
            vs, ds = [], []
            for c in range(0, len(M), 4000):
                v, d = top_pos(L[:, M[c:c + 4000]].mean(-1))
                vs.append(v)
                ds.append(d)
            K = max(x.shape[1] for x in vs)
            V = np.vstack([np.pad(x, ((0, 0), (0, K - x.shape[1]))) for x in vs])
            D = np.vstack([np.pad(x, ((0, 0), (0, K - x.shape[1])), constant_values=T) for x in ds])
            PREP[t] = (pp, M, ok, V, D)

        def stat(mult, kb):
            tm = tail_from(vm, dm, mult, kb)
            out = {}
            for t, (pp, M, ok, V, D) in PREP.items():
                tp = tail_from(V, D, mult, kb)
                den = tm[M].mean(1)
                r = np.where(ok & (den > 0), tp / np.where(den > 0, den, 1), np.nan)
                f = to_focal(r[None, :], pp)[0]
                ta = to_focal(np.where(ok, tp, np.nan)[None, :], pp)[0]
                out[f"{t}|tail_ratio|med"] = med(f)
                out[f"{t}|tail_ratio|mean"] = mean(f)
                out[f"{t}|tail_abs|med"] = med(ta)
                out[f"{t}|N"] = float(np.isfinite(f).sum())
            out["member_tail|med"] = float(np.median(tm))
            return out

        mult0 = np.ones(T + 1, int)
        mult0[T] = 10 ** 6
        k0 = int(np.ceil(0.01 * T))
        pt = stat(mult0, k0)
        boot = pd.DataFrame([stat(np.append(k[block], 10 ** 6), int(np.ceil(0.01 * (k @ nblk)))) for k in KB])
        res = [dict(pool_type=t, n=5, weights="equal", k_days=k0, N_focal=int(pt[f"{t}|N"]), tail_ratio_median=pt[f"{t}|tail_ratio|med"],
                    ci_lo=ci(boot[f"{t}|tail_ratio|med"])[0], ci_hi=ci(boot[f"{t}|tail_ratio|med"])[1],
                    tail_ratio_mean=pt[f"{t}|tail_ratio|mean"], mean_ci_lo=ci(boot[f"{t}|tail_ratio|mean"])[0], mean_ci_hi=ci(boot[f"{t}|tail_ratio|mean"])[1],
                    pooled_worst1pct_loss_median=pt[f"{t}|tail_abs|med"], member_worst1pct_loss_median=pt["member_tail|med"]) for t in TYPES]
        pd.DataFrame(res).to_csv(os.path.join(INTERMEDIATE, "T4_tail.csv"), index=False, encoding=ENC)

    def distance_table(I):
        """Distance at which the pairwise co-loss share and the co-hit share first fall below 5% (ERA5-Land and CPC events)."""
        N = len(I["hosts"])
        DIST = I["DIST"]
        block = I["block"]
        KB, _ = draws(len(block), N, 36)
        ii, jj = np.where(~np.eye(N, dtype=bool))
        LD = np.log(np.clip(DIST[ii, jj], 5, None))
        nb = int(min(60, len(ii) // 150))
        ed = np.unique(np.quantile(LD, np.linspace(0, 1, nb + 1)))
        bn = np.clip(np.searchsorted(ed, LD, side="right") - 1, 0, len(ed) - 2)
        nbin = len(ed) - 1
        ctr = np.bincount(bn, weights=LD, minlength=nbin) / np.bincount(bn, minlength=nbin)
        C0, BIG = 0.05, 1e7

        def curves(G, H):
            gd, hd = np.diag(G), np.diag(H)
            v = (gd[ii] > 0) & (hd[ii] > 0)
            b = bn[v]
            cnt = np.bincount(b, minlength=nbin).astype(float)
            e = G[ii, jj][v] / gd[ii][v]
            h = H[ii, jj][v] / hd[ii][v]
            out = dict(n=cnt, kappa=np.bincount(b, weights=e, minlength=nbin) / cnt, hit=np.bincount(b, weights=h, minlength=nbin) / cnt,
                       kappa_pooled=np.bincount(b, weights=G[ii, jj][v], minlength=nbin) / np.bincount(b, weights=gd[ii][v], minlength=nbin),
                       hit_pooled=np.bincount(b, weights=H[ii, jj][v], minlength=nbin) / np.bincount(b, weights=hd[ii][v], minlength=nbin),
                       n_focal=int(((gd > 0) & (hd > 0)).sum()))
            for k in ("kappa", "hit", "kappa_pooled", "hit_pooled"):
                out[k + "_iso"] = pava_dec(out[k], cnt)
                d = Lcross(ctr, out[k + "_iso"], C0)
                out["d_" + k] = BIG if np.isinf(d) else d
            return out

        def q(x, p):
            x = np.sort(np.asarray(x, float))
            return float(x[int(np.rint(p * (len(x) - 1)))])

        rows = []
        for prod, key in (("ERA5", "main"), ("CPC", "cpc")):
            M = Moments(I["LOSS"][key], I["EXPO"][key], block)
            _, G, H = M.at()
            pt = curves(G, H)
            bs = []
            for k in KB:
                _, Gb, Hb = M.at(k.astype(float))
                bs.append(curves(Gb, Hb))
            for ver, kk, hk in (("mean of pair ratios (plan wording)", "kappa", "hit"), ("pooled ratio of sums", "kappa_pooled", "hit_pooled")):
                dk = np.array([b["d_" + kk] for b in bs])
                dh = np.array([b["d_" + hk] for b in bs])
                dd = dk - dh
                fmt = lambda x: np.nan if x >= BIG / 2 else x
                rows.append(dict(product=prod, version=ver, n_focal=pt["n_focal"], threshold=C0, d_coloss_km=fmt(pt["d_" + kk]),
                                 d_coloss_ci_lo=fmt(q(dk, .025)), d_coloss_ci_hi=fmt(q(dk, .975)), d_cohit_km=fmt(pt["d_" + hk]),
                                 d_cohit_ci_lo=fmt(q(dh, .025)), d_cohit_ci_hi=fmt(q(dh, .975)), diff_km=fmt(pt["d_" + kk]) - fmt(pt["d_" + hk]),
                                 diff_ci_lo=q(dd, .025) if abs(q(dd, .025)) < BIG / 2 else np.nan, diff_ci_hi=q(dd, .975) if abs(q(dd, .975)) < BIG / 2 else np.nan,
                                 share_draws_coloss_never_below=float((dk >= BIG / 2).mean()), share_draws_cohit_never_below=float((dh >= BIG / 2).mean()),
                                 share_draws_diff_gt0=float((dd > 0).mean()), coloss_first_bin_value=float(pt[kk + "_iso"][0]),
                                 cohit_first_bin_value=float(pt[hk + "_iso"][0])))
        pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "T4_K5_distances.csv"), index=False, encoding=ENC)

    def magnitude_table(I):
        """Five worst group days (spending-weighted main loss) of the five clusters."""
        hosts = I["hosts"]
        hix = {c: i for i, c in enumerate(hosts)}
        L, E, w, dates = I["LOSS"]["main"], I["EXPO"]["main"], I["WSP"], I["dates"]
        et = pd.read_csv(os.path.join(INTERMEDIATE, "T4_event_losses.csv"), encoding=ENC)
        et["h"] = et.city_name.map(hix)
        rows = []
        for c in FYP5:
            m = np.array(I["POOLS"]["members"][c])
            ww = w[m] / w[m].sum()
            l = L[:, m] @ ww
            le = L[:, m].mean(1)
            sd = l.std()
            for rank, t in enumerate(np.argsort(-l)[:5], 1):
                act = et[et.h.isin(m) & (et.start <= t) & (et.end >= t)]
                sysc = act.meteorological_event_id.value_counts()
                top = sysc.index[0] if len(sysc) else ""
                tcn = act[(act.meteorological_event_id == top) & (act.TC500 == 1)].TCname
                rows.append(dict(cluster=c, n_host_members=len(m), rank=rank, date=str(dates[t].date()), share_members_in_event=float(E[t, m].mean()),
                                 n_members_in_event=int(E[t, m].sum()), spending_share_of_members_in_event=float(ww[E[t, m]].sum()),
                                 pooled_loss_pct_spending_weighted=float(l[t]), pooled_loss_pct_equal_weighted=float(le[t]),
                                 pooled_loss_in_sd_of_daily_pooled_loss=float(l[t] / sd), main_system_id=top, n_systems_active=int(len(sysc)),
                                 members_in_main_system=int(sysc.iloc[0]) if len(sysc) else 0, IBTrACS_name=(tcn.mode().iloc[0] if len(tcn) else ""),
                                 n_members_within_500km_of_TC=int((act.TC500 == 1).sum())))
        pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "T4_magnitudes_worst_days.csv"), index=False, encoding=ENC)

    def collect(work):
        def load(name):
            return (json.load(open(os.path.join(work, f"res_{name}_point.json"))), pd.read_parquet(os.path.join(work, f"res_{name}_boot.parquet")),
                    pd.read_parquet(os.path.join(work, f"res_{name}_perm.parquet")))

        def est(pt, bt, key, flip=False):
            lo, hi = ci(bt[key])
            v = pt[key]
            return (1 - v, 1 - hi, 1 - lo) if flip else (v, lo, hi)

        rows = []
        for cfg, wlab, ns in (("main", "equal", (5, 10)), ("spw", "spending", (5,)), ("n10", "equal", (10,))):
            pt, bt, pm = load(cfg)
            for n in ns:
                for t in TYPES:
                    if cfg == "n10":  # only ECL and ES_real at n = 10 come from this run
                        stats = []
                    else:
                        stats = [("VR", f"{t}|VR|{{a}}|{n}", False), ("RR", f"{t}|RR|{{a}}|{n}", False), ("DA", f"{t}|RR|{{a}}|{n}", True),
                                 ("rho", f"{t}|rho|{{a}}|{n}", False), ("F", f"{t}|F|{{a}}|{n}", False)]
                    if pt["_nmain"] == n:
                        stats += [("ECL", f"{t}|ECL|{{a}}", False), ("ES_real", f"{t}|ES|{{a}}", False), ("ECL/ES_real", f"{t}|ECLratio|{{a}}", False)]
                    for s, k, flip in stats:
                        r = dict(weights=wlab, n=n, pool_type=LAB[t], statistic=s, N_focal=int(pt[f"{t}|N_ecl"] if s.startswith("E") else pt[f"{t}|N|{n}"]))
                        for a, al in (("med", "median"), ("mean", "mean")):
                            v, lo, hi = est(pt, bt, k.format(a=a), flip)
                            r[al], r[al + "_ci_lo"], r[al + "_ci_hi"] = v, lo, hi
                        if s.startswith("E"):
                            v, lo, hi = est(pt, bt, k.format(a="pooled"))
                            r["pooled"], r["pooled_ci_lo"], r["pooled_ci_hi"] = v, lo, hi
                        if s == "VR":
                            kk = k.format(a="med")
                            r["perm_P_median"] = perm_p(pt[kk], pm[kk])
                            r["perm_null_median_mean"] = float(pm[kk].mean())
                            r["perm_null_median_p95"] = float(pm[kk].quantile(.95))
                            km = k.format(a="mean")
                            r["perm_P_mean"] = perm_p(pt[km], pm[km])
                        if s == "RR":
                            r["perm_null_median_mean"] = float(pm[k.format(a="med")].mean())
                        rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "T4_pool_statistics_main.csv"), index=False, encoding=ENC)
        pt, bt, pm = load("main")
        pd.DataFrame([dict(cluster=k.split("|")[2], RR_whole=pt[k], ci_lo=ci(bt[k])[0], ci_hi=ci(bt[k])[1]) for k in pt if k.startswith("whole|RR|")]).to_csv(
            os.path.join(INTERMEDIATE, "T4_whole_cluster_RR.csv"), index=False, encoding=ENC)
        cr = []
        for t in TYPES:
            for n in range(2, 21):
                r = dict(pool_type=LAB[t], n=n, N_focal=int(pt[f"{t}|N|{n}"]), independent_equal_benchmark=1 / np.sqrt(n))
                for s in ("RR", "VR", "F"):
                    for a, al in (("med", "median"), ("mean", "mean")):
                        v, lo, hi = est(pt, bt, f"{t}|{s}|{a}|{n}")
                        r[f"{s}_{al}"], r[f"{s}_{al}_ci_lo"], r[f"{s}_{al}_ci_hi"] = v, lo, hi
                cr.append(r)
        pd.DataFrame(cr).to_csv(os.path.join(INTERMEDIATE, "T4_size_span_curve.csv"), index=False, encoding=ENC)

    def main():
        work = tempfile.mkdtemp(prefix="group_loss_")
        me = os.path.abspath(__file__)
        procs = {n: subprocess.Popen([sys.executable, me, "group_loss_tests", "worker", n, work]) for n in CONFIGS}
        I = load_inputs()
        tail_table(I)
        distance_table(I)
        magnitude_table(I)
        while any(p.poll() is None for p in procs.values()):
            time.sleep(5)
        failed = [n for n, p in procs.items() if p.returncode != 0]
        if failed:
            raise RuntimeError(f"workers failed: {failed}")
        collect(work)
        print("done")

    argv = argv or [sys.argv[0]]
    if len(argv) > 1 and argv[1] == "worker":
        run_worker(argv[2], argv[3])
    else:
        main()


def decoupling_distance_by_city():
    """City-specific decoupling distance L05: the distance at which the isotonic (non-increasing) fit of binned pairwise
    co-extreme probability pi first falls below 0.05, with partner-resampling bootstrap intervals, an independent CPC
    version, and summaries by city cluster and NBS region."""
    ENC = "utf-8-sig"
    BW = 50.0
    NBOOT = 999
    rng = np.random.default_rng(20261111)

    def pava_dec(y, w):
        y = -np.asarray(y, float)
        w = np.asarray(w, float)
        vals, wts, cnt = [], [], []
        for yi, wi in zip(y, w):
            vals.append(yi)
            wts.append(wi)
            cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
                ww = w1 + w2
                vals.append((v1 * w1 + v2 * w2) / ww)
                wts.append(ww)
                cnt.append(c1 + c2)
        return -np.repeat(vals, cnt)

    def lcross_lin(x, f, c=0.05):
        k = np.where(f < c)[0]
        if len(k) == 0:
            return np.inf
        k = k[0]
        if k == 0:
            return 0.0
        return float(x[k - 1] + (f[k - 1] - c) * (x[k] - x[k - 1]) / (f[k - 1] - f[k]))

    def L05_binned(d, p, wt=None):
        wt = np.ones_like(d) if wt is None else wt
        b = (d // BW).astype(int)
        nb = b.max() + 1
        n = np.bincount(b, weights=wt, minlength=nb)
        s = np.bincount(b, weights=wt * p, minlength=nb)
        sd = np.bincount(b, weights=wt * d, minlength=nb)
        g = n > 0
        ctr = sd[g] / n[g]
        mean = s[g] / n[g]
        fit = pava_dec(mean, n[g])
        return lcross_lin(ctr, fit), ctr, fit

    def L05_pair(d, p):
        o = np.argsort(d, kind="stable")
        fit = pava_dec(p[o], np.ones(len(o)))
        return lcross_lin(d[o], fit)

    def city_L05(df, pcol):
        rows = []
        for c, g in df.groupby("focal_city"):
            d = g.distance_km.to_numpy()
            p = g[pcol].to_numpy()
            ok = np.isfinite(p)
            d, p = d[ok], p[ok]
            L, ctr, fit = L05_binned(d, p)
            Lp = L05_pair(d, p)
            npair = len(d)
            bs = np.empty(NBOOT)
            for b in range(NBOOT):
                cnt = np.bincount(rng.integers(0, npair, npair), minlength=npair).astype(float)
                m = cnt > 0
                bs[b] = L05_binned(d[m], p[m], cnt[m])[0]
            rows.append(dict(city=c, L05_km=L, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), L05_pairiso_km=Lp, n_pairs=npair,
                             pi_first_bin=fit[0], nearest_partner_km=d.min(), boot_inf=int(np.isinf(bs).sum()), boot_zero=int((bs == 0).sum())))
        return pd.DataFrame(rows)

    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    for c in ["focal_city", "partner_city"]:
        hp[c] = hp[c].astype(str).str.strip()
    ch = pd.read_csv(os.path.join(INTERMEDIATE, "city_hazard.csv"), encoding=ENC)
    chh = ch.assign(city_name=ch.city_name.str.strip()).set_index("city_name")
    cl = pd.read_csv(os.path.join(INTERMEDIATE, "city_clusters_19.csv"), encoding=ENC)
    pn = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC)
    for c in ["host_name", "guest_name"]:
        pn[c] = pn[c].astype(str).str.strip()
    attr = pd.read_csv(os.path.join(EXTERNAL, "city_attributes.csv"), encoding=ENC)
    attr["city_name"] = attr.city_name.str.strip()
    prov = attr.drop_duplicates("city_name").set_index("city_name").province
    regions = pd.read_csv(os.path.join(EXTERNAL, "province_regions.csv"), encoding=ENC)
    p2r = dict(zip(regions.province, regions.region4.str.lower()))

    L1 = city_L05(hp, "pi")
    p18 = pn[(pn.network_year == 2018) & (pn.usable == 1)].copy()
    p18["wn"] = p18.w / p18.groupby("host_name").w.transform("sum")
    p18["L05_host"] = p18.host_name.map(L1.set_index("city").L05_km)
    g = p18.groupby("host_name")
    S18 = g.apply(lambda x: (x.wn * x.pi).sum())
    inside = g.apply(lambda x: x.wn[x.distance_km < x.L05_host].sum())
    in451 = g.apply(lambda x: x.wn[x.distance_km < 451].sum())

    clh = cl[cl.cluster_en.isin(["JJJ", "YRD", "PRD", "MYR", "CY"])].copy()
    clh["city_name"] = clh.city_name.str.strip()
    cmap = clh.groupby("city_name").cluster_en.agg(lambda s: "+".join(sorted(s)))
    F = L1.copy()
    F["lat"] = F.city.map(chh.lat)
    F["lon"] = F.city.map(chh.lon)
    F["S_2018"] = F.city.map(S18)
    F["demand_share_inside_L05"] = F.city.map(inside)
    F["demand_share_inside_451"] = F.city.map(in451)
    F["cluster"] = F.city.map(cmap).fillna("none")
    F["sample_host"] = F.city.isin(S18.index).astype(int)
    F["province"] = F.city.map(prov)
    F["region"] = F.province.map(p2r)
    F["region_source"] = np.where(F.region.notna(), "province_NBS", "nearest_host_approx")
    hh = F[F.region.notna()]
    for i in F.index[F.region.isna()]:
        dd = (hh.lat - F.lat[i]) ** 2 + ((hh.lon - F.lon[i]) * np.cos(np.radians(F.lat[i]))) ** 2
        F.loc[i, "region"] = hh.region[dd.idxmin()]
    F["L05_flag"] = np.where(F.L05_km == 0, "first_bin_below_0.05", "")

    # CPC pairwise co-extreme probability, 1981-2016, top n_i days per unit
    cp = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_cpc_1981_2019.parquet"))
    cp["city_name"] = cp.city_name.str.strip()
    cp["date"] = pd.to_datetime(cp.date)
    units = sorted(chh.index)
    cpp = cp[cp.date.dt.year <= 2016].pivot(index="date", columns="city_name", values="rain_mm").reindex(columns=units)
    ni = np.rint(chh.extreme_days_per_year.reindex(units) * 36).astype(int).to_numpy()
    Xv = cpp.to_numpy()
    Hc = np.zeros(Xv.shape, bool)
    for j in range(len(units)):
        if ni[j] == 0:
            continue
        v = np.where(np.isfinite(Xv[:, j]), Xv[:, j], -1)
        thr = np.sort(v)[-ni[j]]
        Hc[:, j] = v >= thr
    b = Hc.astype(float)
    own = b.sum(0)
    co = b.T @ b
    with np.errstate(invalid="ignore", divide="ignore"):
        PIc = co / own[:, None]
    uix = {u: i for i, u in enumerate(units)}
    hp["pi_cpc"] = PIc[hp.focal_city.map(uix).to_numpy(), hp.partner_city.map(uix).to_numpy()]
    L1c = city_L05(hp, "pi_cpc").set_index("city")
    F["L05_cpc_km"] = F.city.map(L1c.L05_km)
    F["L05_cpc_lo"] = F.city.map(L1c.lo)
    F["L05_cpc_hi"] = F.city.map(L1c.hi)

    def summ(df, label, grp):
        v = df.L05_km
        return dict(set=label, group=grp, n=len(df), L05_median=v.median(), L05_p25=v.quantile(.25), L05_p75=v.quantile(.75),
                    share_below_451=(v < 451).mean(), share_CI_above_451=(df.lo > 451).mean(), share_CI_below_451=(df.hi < 451).mean(),
                    L05_cpc_median=df.L05_cpc_km.median(), S2018_median=df.S_2018.median(),
                    demand_inside_L05i_median=df.demand_share_inside_L05.median(), demand_inside_451_median=df.demand_share_inside_451.median())

    rows = [summ(F, "all_367_units", "all"), summ(F[F.sample_host == 1], "sample_hosts_224", "all")]
    H = F[F.sample_host == 1]
    for k in ["JJJ", "YRD", "PRD", "MYR", "CY"]:
        rows.append(summ(H[H.cluster.str.contains(k)], "sample_hosts_224", "cluster:" + k))
    rows.append(summ(H[H.cluster == "none"], "sample_hosts_224", "cluster:none"))
    for r_ in ["east", "central", "west", "northeast"]:
        rows.append(summ(H[H.region == r_], "sample_hosts_224", "region:" + r_))
    for r_ in ["east", "central", "west", "northeast"]:
        rows.append(summ(F[F.region == r_], "all_367_units", "region:" + r_))
    SUM = pd.DataFrame(rows)
    sp = [("L05_vs_S2018",) + tuple(spearmanr(H.L05_km, H.S_2018)), ("L05_vs_demand_inside",) + tuple(spearmanr(H.L05_km, H.demand_share_inside_L05)),
          ("L05_ERA5_vs_CPC_all367",) + tuple(spearmanr(F.L05_km, F.L05_cpc_km)), ("L05_vs_lat_all367",) + tuple(spearmanr(F.L05_km, F.lat))]
    SP = pd.DataFrame(sp, columns=["pair", "spearman", "p"])
    cols = ["city", "lat", "lon", "L05_km", "lo", "hi", "n_pairs", "S_2018", "demand_share_inside_L05", "cluster", "region", "region_source", "sample_host", "province",
            "L05_pairiso_km", "pi_first_bin", "nearest_partner_km", "L05_flag", "demand_share_inside_451", "L05_cpc_km", "L05_cpc_lo", "L05_cpc_hi", "boot_inf", "boot_zero"]
    F[cols].to_csv(os.path.join(INTERMEDIATE, "fig5a_city_L05.csv"), index=False, encoding=ENC)
    SUM.to_csv(os.path.join(INTERMEDIATE, "fig5a_L05_summary.csv"), index=False, encoding=ENC)
    SP.to_csv(os.path.join(INTERMEDIATE, "fig5a_L05_correlations.csv"), index=False, encoding=ENC)
    print(SUM[["set", "group", "n", "L05_median"]].to_string(index=False))


def group_exposure():
    """Ex-ante shared-rain exposure of the city groups (mean pi between the focal city and the other members, ERA5-Land
    1981-2016, early and late periods and CPC) against the realised variance ratio VR of their combined spending
    shortfall: focal-level table, binned medians with focal-city bootstrap CIs, exposure by group design, and
    diagnostics of the daily loss matrix."""
    ENC = "utf-8-sig"
    TY = ["cluster", "province", "random", "demand", "crossbelt"]
    B = 999
    warnings.filterwarnings("ignore")

    class Moments:
        """Per-block sufficient statistics of a loss matrix L (T x N) and exposure mask E (T x N)."""

        def __init__(self, L, E, block):
            ub = np.unique(block)
            bid = np.searchsorted(ub, block)
            nb = len(ub)
            N = L.shape[1]
            Ef = E.astype(float)
            self.n = np.bincount(bid, minlength=nb).astype(float)
            self.B = nb
            self.S1 = np.zeros((nb, N))
            self.S2 = np.zeros((nb, N, N))
            self.G = np.zeros((nb, N, N))
            self.H = np.zeros((nb, N, N))
            for b in range(nb):
                Lb = L[bid == b]
                Eb = Ef[bid == b]
                self.S1[b] = Lb.sum(0)
                self.S2[b] = Lb.T @ Lb
                self.G[b] = Eb.T @ Lb
                self.H[b] = Eb.T @ Eb

        def at(self, k=None):
            k = np.ones(self.B) if k is None else k
            n = k @ self.n
            m = (k @ self.S1) / n
            return np.tensordot(k, self.S2, 1) / n - np.outer(m, m), np.tensordot(k, self.G, 1), np.tensordot(k, self.H, 1)

    def prep(pool, N, nmax):
        M = pool["M"][:, :nmax]
        d = pool["draws"]
        size = np.repeat(np.minimum(pool["size"], nmax), d)
        M0 = np.where(M >= 0, M, 0)
        pa = np.array([a for b in range(1, nmax) for a in range(b)])
        pb = np.array([b for b in range(1, nmax) for a in range(b)])
        uf = np.unique(pool["focal"])
        RF = (pool["focal"][:, None] == uf[None, :]).astype(float)
        return dict(idx=(M0[:, pa] * N + M0[:, pb]).T.astype(np.int32).copy(), didx=(M0 * N + M0).T.astype(np.int32).copy(), M0T=M0.T.copy(),
                    pa=pa, pb=pb, size=size, nmax=nmax, P=len(M), R=len(pool["focal"]), draws=d, focal=uf, RF=RF)

    def nested(C, pp):
        Cf = C.ravel()
        sd = np.sqrt(np.clip(np.diag(C), 0, None))
        nmax, P = pp["nmax"], pp["P"]
        X = Cf.take(pp["idx"])
        Dg = Cf.take(pp["didx"])
        SD = sd.take(pp["M0T"])
        S = Dg[0].copy()
        D = Dg[0].copy()
        Pf = SD[0].copy()
        pos = 0
        VR = np.full((nmax - 1, P), np.nan)
        RR = np.full((nmax - 1, P), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            for b in range(1, nmax):
                S = S + Dg[b] + 2 * X[pos:pos + b].sum(0)
                pos += b
                D = D + Dg[b]
                Pf = Pf + SD[b]
                ok = pp["size"] >= b + 1
                sP = np.sqrt(np.clip(S, 0, None))
                VR[b - 1] = np.where(ok & (D > 0), sP / np.sqrt(D), np.nan)
                RR[b - 1] = np.where(ok & (Pf > 0), sP / Pf, np.nan)
        return VR, RR

    def to_focal(x, pp):
        nn = x.shape[0]
        xr = x.reshape(nn, pp["R"], pp["draws"])
        ok = np.isfinite(xr)
        cnt = ok.sum(2)
        rowm = np.where(cnt > 0, np.where(ok, xr, 0).sum(2) / np.maximum(cnt, 1), np.nan)
        okr = np.isfinite(rowm)
        num = np.where(okr, rowm, 0) @ pp["RF"]
        den = okr.astype(float) @ pp["RF"]
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    def es_real(G, H, pp, n):
        i = pp["M0T"][0]
        mem = pp["M0T"][1:n]
        g = G[i[None, :], mem]
        h = H[i[None, :], mem]
        gb, hb = g.mean(0), h.mean(0)
        ok = pp["size"] >= n
        Gb = to_focal(np.where(ok, gb, np.nan)[None, :], pp)[0]
        Hb = to_focal(np.where(ok, hb, np.nan)[None, :], pp)[0]
        f = pp["focal"]
        Gff, Hff = G[f, f], H[f, f]
        v = np.isfinite(Gb) & (Gff > 0) & (Hff > 0)
        return np.where(v, Hb / np.where(v, Hff, 1), np.nan)

    def agg(P, val):
        d = P["draws"]
        R_ = len(P["focal"])
        x = val.reshape(R_, d)
        with np.errstate(all="ignore"):
            rm = np.nanmean(x, 1)
        return pd.DataFrame(dict(f=P["focal"], v=rm)).groupby("f").v.mean()

    def pool_vals(P, n, fun):
        M = P["M"]
        size = np.repeat(P["size"], P["draws"])
        ok = size >= n
        return np.where(ok, fun(M[:, :n]), np.nan)

    I = pickle.load(open(os.path.join(INTERMEDIATE, "t4_inputs.pkl"), "rb"))
    hosts = I["hosts"]
    N = len(hosts)
    hix = {c: i for i, c in enumerate(hosts)}
    DIST = I["DIST"]
    POOLS = I["POOLS"]
    L = I["LOSS"]["main"]
    Ex = I["EXPO"]["main"]

    # focal-level VR, RR and realised co-exposure share (equal weights, point estimates)
    C, G, H = Moments(L, Ex, I["block"]).at()
    fl = []
    for t in TY:
        pp = prep(POOLS[t], N, 10)
        VR, RR = nested(C, pp)
        VRf, RRf = to_focal(VR, pp), to_focal(RR, pp)
        fl.append(pd.DataFrame({"pool_type": t, "focal": pp["focal"], "VR_n5": VRf[3], "VR_n10": VRf[8], "RR_n5": RRf[3],
                                "ES_real_n5": es_real(G, H, pp, 5)}))
    fl = pd.concat(fl, ignore_index=True)

    # pairwise co-extreme probability matrices (ERA5-Land and CPC)
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    for c in ["focal_city", "partner_city"]:
        hp[c] = hp[c].astype(str).str.strip()
    ch = pd.read_csv(os.path.join(INTERMEDIATE, "city_hazard.csv"), encoding=ENC)
    chh = ch.assign(city_name=ch.city_name.str.strip()).set_index("city_name")
    cp = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_cpc_1981_2019.parquet"))
    cp["city_name"] = cp.city_name.str.strip()
    cp["date"] = pd.to_datetime(cp.date)
    units = sorted(chh.index)
    cpp = cp[cp.date.dt.year <= 2016].pivot(index="date", columns="city_name", values="rain_mm").reindex(columns=units)
    ni = np.rint(chh.extreme_days_per_year.reindex(units) * 36).astype(int).to_numpy()
    Xv = cpp.to_numpy()
    Hc = np.zeros(Xv.shape, bool)
    for j in range(len(units)):
        if ni[j] == 0:
            continue
        v = np.where(np.isfinite(Xv[:, j]), Xv[:, j], -1)
        thr = np.sort(v)[-ni[j]]
        Hc[:, j] = v >= thr
    b = Hc.astype(float)
    own = b.sum(0)
    co = b.T @ b
    with np.errstate(invalid="ignore", divide="ignore"):
        PIc = co / own[:, None]
    uix = {u: i for i, u in enumerate(units)}
    uix = {u: i for i, u in enumerate(units)}
    hp["pi_cpc"] = PIc[hp.focal_city.map(uix).to_numpy(), hp.partner_city.map(uix).to_numpy()]

    def mat(col):
        Mx = np.full((N, N), np.nan)
        h = hp[hp.focal_city.isin(hix) & hp.partner_city.isin(hix)]
        Mx[h.focal_city.map(hix), h.partner_city.map(hix)] = h[col].to_numpy()
        return Mx

    PIM = {"S_pool": mat("pi"), "S_pool_early": mat("pi_early"), "S_pool_late": mat("pi_late"), "S_pool_cpc": mat("pi_cpc")}
    recs = []
    for t_ in TY:
        P = POOLS[t_]
        cols = {}
        for n in (5, 10):
            for k, Mx in PIM.items():
                cols[f"{k}_n{n}"] = agg(P, pool_vals(P, n, lambda Mn: np.nanmean(Mx[Mn[:, :1], Mn[:, 1:]], 1)))
            cols[f"mean_member_dist_km_n{n}"] = agg(P, pool_vals(P, n, lambda Mn: DIST[Mn[:, :1], Mn[:, 1:]].mean(1)))
        d = pd.DataFrame(cols)
        d["pool_type"] = t_
        d["focal"] = d.index
        recs.append(d)
    SP_ = pd.concat(recs, ignore_index=True)
    SP_["city_name"] = [hosts[i] for i in SP_.focal]
    D2 = fl.merge(SP_, on=["pool_type", "focal"], how="outer")

    # binned VR medians against group exposure, focal-city bootstrap
    rngb = np.random.default_rng(20261112)
    cities = np.array(sorted(D2.focal.unique()))
    boot_idx = [rngb.choice(cities, len(cities), replace=True) for _ in range(B)]

    def binned(d, x, y, scheme):
        d = d.copy()
        if scheme == "quintile_rank_first":
            d = d.sort_values([x, "focal", "pool_type"])
            d["q"] = pd.qcut(np.arange(len(d)), 5, labels=False) + 1
        else:
            pos = d[x] > 0
            d["q"] = 0
            d.loc[pos, "q"] = pd.qcut(d.loc[pos, x].rank(method="first"), 5, labels=False) + 1
        qs = sorted(d.q.unique())
        byc = {c: g for c, g in d.groupby("focal")}
        pt = d.groupby("q").agg(n_rows=(y, "size"), S_min=(x, "min"), S_max=(x, "max"), S_median=(x, "median"), VR_median=(y, "median"))
        bm = np.full((B, len(qs)), np.nan)
        for b_, ix in enumerate(boot_idx):
            db = pd.concat([byc[c] for c in ix if c in byc])
            bm[b_] = db.groupby("q")[y].median().reindex(qs).to_numpy()
        pt["VR_lo"] = np.nanquantile(bm, .025, 0)
        pt["VR_hi"] = np.nanquantile(bm, .975, 0)
        pt["type_mix"] = d.groupby("q").pool_type.agg(lambda s: "; ".join(f"{k}:{v}" for k, v in s.value_counts().items()))
        return pt.reset_index().assign(binning=scheme)

    bins_out = []
    for n in (5, 10):
        for sv in ("S_pool", "S_pool_early"):
            x, y = f"{sv}_n{n}", f"VR_n{n}"
            d = D2[["pool_type", "focal", x, y]].dropna().copy()
            for sch in ("quintile_rank_first", "zero_bin_plus_quintiles_of_positive"):
                bins_out.append(binned(d, x, y, sch).assign(n=n, S_version=sv, x=x, y=y))
    BIN = pd.concat(bins_out, ignore_index=True)

    ST = []
    for t_, g in D2.groupby("pool_type"):
        for n in (5, 10):
            s = g[f"S_pool_n{n}"].dropna()
            ST.append(dict(pool_type=t_, n=n, N_focal=len(s), S_median=s.median(), S_p25=s.quantile(.25), S_p75=s.quantile(.75), S_mean=s.mean(),
                           S_early_median=g[f"S_pool_early_n{n}"].median(), S_late_median=g[f"S_pool_late_n{n}"].median(), S_cpc_median=g[f"S_pool_cpc_n{n}"].median(),
                           share_S_zero=(s == 0).mean(), VR_median=g[f"VR_n{n}"].median(), mean_member_dist_km_median=g[f"mean_member_dist_km_n{n}"].median()))
    ST = pd.DataFrame(ST)
    D2out = D2.copy()
    D2out.insert(0, "city_name", D2out.pop("city_name"))
    D2out.to_csv(os.path.join(INTERMEDIATE, "fig5c_pool_S_vs_VR.csv"), index=False, encoding=ENC)
    BIN.to_csv(os.path.join(INTERMEDIATE, "fig5c_binned.csv"), index=False, encoding=ENC)
    ST.to_csv(os.path.join(INTERMEDIATE, "fig5c_S_by_pool_type.csv"), index=False, encoding=ENC)

    # daily loss matrix diagnostics
    npos = (L > 0).sum(0)
    diag = pd.DataFrame([dict(T_days=L.shape[0], k_days_1pct=int(np.ceil(0.01 * L.shape[0])), n_cities=L.shape[1], pos_loss_days_median=float(np.median(npos)),
                              pos_loss_days_p90=float(np.percentile(npos, 90)), pos_loss_days_max=int(npos.max()), share_cities_pos_days_le_k=float((npos <= 11).mean()),
                              share_cities_zero_pos_days=float((npos == 0).mean()), share_exposure_days_negative_L=float(((L < 0) & (Ex > 0)).sum() / (Ex > 0).sum()))])
    diag.to_csv(os.path.join(INTERMEDIATE, "fig5d_tail_diagnostics.csv"), index=False, encoding=ENC)
    print(ST[ST.n == 5][["pool_type", "N_focal", "S_median", "VR_median"]].to_string(index=False))


def east_west_groups():
    """East-West cooperation pairing groups: pairwise co-extreme probability pi of each eastern unit with its western
    partners (and the reverse), group-level shared-rain exposure against distance-matched, national-random and nearest-5
    benchmarks, and pair-level summaries."""
    ENC = "utf-8-sig"
    WEST = {"\u5185\u8499\u53e4\u81ea\u6cbb\u533a", "\u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a", "\u91cd\u5e86\u5e02", "\u56db\u5ddd\u7701",
            "\u8d35\u5dde\u7701", "\u4e91\u5357\u7701", "\u897f\u85cf\u81ea\u6cbb\u533a", "\u9655\u897f\u7701", "\u7518\u8083\u7701",
            "\u9752\u6d77\u7701", "\u5b81\u590f\u56de\u65cf\u81ea\u6cbb\u533a", "\u65b0\u7586\u7ef4\u543e\u5c14\u81ea\u6cbb\u533a"}  # the 12 western provinces

    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    pn = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC, usecols=["host_name", "network_year", "guest_name"])
    pairs = pd.read_csv(os.path.join(EXTERNAL, "east_west_cooperation_pairs.csv"), encoding=ENC)
    units = set(hp.focal_city) | set(hp.partner_city)
    hosts = set(pn.host_name)
    pairs["in_hazard_pairs"] = pairs.west_unit.isin(units) & pairs.east_unit.isin(units)
    pairs["east_in_sample"] = pairs.east_unit.isin(hosts)
    pairs["west_in_sample"] = pairs.west_unit.isin(hosts)
    pairs["recipient_is_western_province"] = pairs.west_province.isin(WEST)
    hpi = hp.set_index(["focal_city", "partner_city"])

    def lk(a, b, col):
        try:
            return float(hpi.at[(a, b), col])
        except KeyError:
            return np.nan

    for c in ["pi", "pi_early", "pi_late", "distance_km"]:
        pairs["east_to_west_" + c] = [lk(a, b, c) for a, b in zip(pairs.east_unit, pairs.west_unit)]
        pairs["west_to_east_" + c] = [lk(b, a, c) for a, b in zip(pairs.east_unit, pairs.west_unit)]

    g_all = hp.groupby("focal_city")
    nat = g_all.pi.mean()
    nat_e = g_all.pi_early.mean()
    near5 = hp.sort_values("distance_km").groupby("focal_city").head(5).groupby("focal_city").agg(near5_S=("pi", "mean"), near5_dist=("distance_km", "median"))

    def distmatched(f, d):
        sub = hp[(hp.focal_city == f) & (hp.distance_km.between(0.9 * d, 1.1 * d))]
        return sub.pi.mean()

    ok = pairs[pairs.in_hazard_pairs].copy()
    ok["dm_pi"] = [distmatched(a, d) for a, d in zip(ok.east_unit, ok.east_to_west_distance_km)]
    ok["dm_pi_west"] = [distmatched(b, d) for b, d in zip(ok.west_unit, ok.east_to_west_distance_km)]
    out = []
    for scope, sub0 in [("western_recipients", ok[ok.recipient_is_western_province]), ("all_recipients", ok)]:
        for per, sub in sub0.groupby("period"):
            for side, key, pcol, ecol, dcol, dmcol in [("east_focal", "east_unit", "east_to_west_pi", "east_to_west_pi_early", "east_to_west_distance_km", "dm_pi"),
                                                       ("west_focal", "west_unit", "west_to_east_pi", "west_to_east_pi_early", "west_to_east_distance_km", "dm_pi_west")]:
                sub2 = sub.drop_duplicates([key, "west_unit" if side == "east_focal" else "east_unit"])
                for u, gg in sub2.groupby(key):
                    out.append(dict(scope=scope, period=per, side=side, focal_unit=u,
                                    province=gg.east_province.iloc[0] if side == "east_focal" else gg.west_province.iloc[0],
                                    focal_in_sample=u in hosts, n_partners=len(gg),
                                    partners="\u3001".join(gg["west_unit" if side == "east_focal" else "east_unit"]),
                                    pool_S=gg[pcol].mean(), pool_S_early=gg[ecol].mean(), pool_S_late=np.nan,
                                    median_distance_km=gg[dcol].median(), max_pi=gg[pcol].max(),
                                    distmatched_random_S=gg[dmcol].mean(),
                                    national_random_S=nat.get(u, np.nan), national_random_S_early=nat_e.get(u, np.nan),
                                    nearest5_S=near5.near5_S.get(u, np.nan), nearest5_median_km=near5.near5_dist.get(u, np.nan)))
    S = pd.DataFrame(out)
    for i, r in S.iterrows():
        sub = ok[(ok.period == r.period) & ((ok.recipient_is_western_province) | (r.scope == "all_recipients"))]
        if r.side == "east_focal":
            gg = sub[sub.east_unit == r.focal_unit].drop_duplicates("west_unit")
            S.at[i, "pool_S_late"] = gg.east_to_west_pi_late.mean()
        else:
            gg = sub[sub.west_unit == r.focal_unit].drop_duplicates("east_unit")
            S.at[i, "pool_S_late"] = gg.west_to_east_pi_late.mean()
    summ = S.groupby(["scope", "period", "side"]).agg(
        n_units=("focal_unit", "size"), n_in_sample=("focal_in_sample", "sum"),
        S_median=("pool_S", "median"), S_min=("pool_S", "min"), S_max=("pool_S", "max"), S_q25=("pool_S", lambda x: x.quantile(.25)), S_q75=("pool_S", lambda x: x.quantile(.75)),
        S_early_median=("pool_S_early", "median"), S_late_median=("pool_S_late", "median"),
        n_partners_median=("n_partners", "median"), dist_median=("median_distance_km", "median"), dist_min=("median_distance_km", "min"), dist_max=("median_distance_km", "max"),
        share_units_dist_above_451=("median_distance_km", lambda x: (x > 451).mean()),
        distmatched_median=("distmatched_random_S", "median"), natrand_median=("national_random_S", "median"), near5_median=("nearest5_S", "median"),
        share_S_below_0p05=("pool_S", lambda x: (x < 0.05).mean()), share_S_zero=("pool_S", lambda x: (x == 0).mean())).reset_index()
    m = S.groupby(["scope", "period", "side"]).agg(S_mean=("pool_S", "mean"), S_early_mean=("pool_S_early", "mean"),
                                                    share_S_positive=("pool_S", lambda x: (x > 0).mean())).reset_index()
    summ.merge(m, on=["scope", "period", "side"]).to_csv(os.path.join(INTERMEDIATE, "fig5_eastwest_pool_S_summary.csv"), index=False, encoding=ENC)

    pr = ok[ok.recipient_is_western_province]
    pairlvl = pr.groupby("period").agg(n_pairs=("east_to_west_pi", "size"), pi_mean=("east_to_west_pi", "mean"), pi_median=("east_to_west_pi", "median"),
                                       share_pi_zero=("east_to_west_pi", lambda x: (x == 0).mean()), distance_median_km=("east_to_west_distance_km", "median"),
                                       share_within_L05_451km=("east_to_west_distance_km", lambda x: (x <= 451).mean()),
                                       share_within_L05_442km=("east_to_west_distance_km", lambda x: (x <= 442).mean())).reset_index()
    pairlvl.to_csv(os.path.join(INTERMEDIATE, "fig5_eastwest_pairlevel_summary.csv"), index=False, encoding=ENC)
    cols = ["period", "east_province", "east_unit", "west_province", "west_unit", "level", "source_url", "in_hazard_pairs", "east_in_sample",
            "west_in_sample", "recipient_is_western_province", "east_to_west_pi", "east_to_west_pi_early", "east_to_west_pi_late", "west_to_east_pi",
            "west_to_east_pi_early", "east_to_west_distance_km", "note"]
    pairs[cols].to_csv(os.path.join(INTERMEDIATE, "fig5_eastwest_pairs.csv"), index=False, encoding=ENC)
    print(pairlvl.to_string(index=False))


def grouping_comparison():
    """Shared-rain exposure S, variance ratio VR and excess peak shortfall X1 of city groups under six groupings (within
    province, within city cluster, existing demand ties, nationwide unselected, beyond the rainfall system, East-West
    cooperation groups), spending-weighted (main) and equal-weighted, with within-grouping slopes of VR on S and the
    peak day of each city cluster."""
    ENC = "utf-8-sig"
    B = 999
    TY = ["province", "cluster", "demand", "random", "crossbelt"]
    NAME = {"province": "Within province", "cluster": "Within city cluster", "demand": "Existing demand ties", "random": "Nationwide, unselected",
            "crossbelt": "Beyond the rainfall system", "eastwest": "East\u2013West cooperation pairs"}
    GROUP = {"province": "Within the region", "cluster": "Within the region", "demand": "Market ties", "random": "Across regions",
             "crossbelt": "Across regions", "eastwest": "Across regions"}
    T4LAB = {"City cluster": "cluster", "Province": "province", "National random": "random", "Demand partners": "demand", "Cross-rain-belt": "crossbelt"}
    NAT5 = ["JJJ", "YRD", "PRD", "MYR", "CY"]
    SEP = "\u3001"
    warnings.filterwarnings("ignore")

    I = pickle.load(open(os.path.join(INTERMEDIATE, "t4_inputs.pkl"), "rb"))
    hosts = I["hosts"]
    N = len(hosts)
    hix = {c: i for i, c in enumerate(hosts)}
    L = I["LOSS"]["main"]
    WSP = np.asarray(I["WSP"], float)
    POOLS = I["POOLS"]
    ps = pd.read_csv(os.path.join(INTERMEDIATE, "T4_pool_statistics_main.csv"), encoding=ENC)
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    for c in ["focal_city", "partner_city"]:
        hp[c] = hp[c].astype(str).str.strip()
    h = hp[hp.focal_city.isin(hix) & hp.partner_city.isin(hix)]

    def mat(col):
        Mx = np.full((N, N), np.nan)
        Mx[h.focal_city.map(hix), h.partner_city.map(hix)] = h[col].to_numpy()
        return Mx

    PIM = {"pi": mat("pi"), "pi_early": mat("pi_early")}
    Lc = L - L.mean(0)
    COV = Lc.T @ Lc / L.shape[0]

    def wmat(Mn, wt):
        return WSP[Mn] if wt == "spending" else np.ones(Mn.shape)

    def vr(Mn, wt):
        w = wmat(Mn, wt)
        num = np.einsum("pa,pab,pb->p", w, COV[Mn[:, :, None], Mn[:, None, :]], w)
        den = (w ** 2 * COV[Mn, Mn]).sum(1)
        with np.errstate(all="ignore"):
            return np.sqrt(np.clip(num, 0, None)) / np.sqrt(den)

    def gS(Mn, wt, pik="pi"):
        w = wmat(Mn, wt)[:, 1:]
        p = PIM[pik][Mn[:, :1], Mn[:, 1:]]
        return (w * p).sum(1) / w.sum(1)

    def pool_vals(P, n, fun):
        M = P["M"]
        size = np.repeat(P["size"], P["draws"])
        ok = size >= n
        out = np.full(len(M), np.nan)
        idx = np.where(ok)[0]
        for c0 in range(0, len(idx), 5000):
            ii = idx[c0:c0 + 5000]
            out[ii] = fun(M[ii, :n])
        return out

    def agg_focal(P, val):
        d = P["draws"]
        R_ = len(P["focal"])
        x = val.reshape(R_, d)
        ok = np.isfinite(x)
        c = ok.sum(1)
        rm = np.where(c > 0, np.where(ok, x, 0).sum(1) / np.maximum(c, 1), np.nan)
        return pd.DataFrame(dict(f=P["focal"], v=rm)).groupby("f").v.mean()

    def focal_from_rows(t, arr):
        P = POOLS[t]
        uf = np.unique(P["focal"])
        okr = np.isfinite(arr)
        RF = (P["focal"][:, None] == uf[None, :]).astype(float)
        num = np.where(okr, arr, 0) @ RF
        dd_ = okr.astype(float) @ RF
        return pd.Series(np.where(dd_ > 0, num / np.maximum(dd_, 1), np.nan), index=uf)

    def ols_cluster(d, y, x, cat, groups):
        """OLS of y on x and category dummies; coefficient on x with standard error clustered by groups
        (small-sample factor G/(G-1) * (N-1)/(N-K))."""
        D = pd.get_dummies(d[cat], drop_first=True, dtype=float).to_numpy()
        X = np.column_stack([np.ones(len(d)), D, d[x].to_numpy(float)])
        yv = d[y].to_numpy(float)
        bread = np.linalg.pinv(X.T @ X)
        beta = np.linalg.pinv(X) @ yv
        u = yv - X @ beta
        g = pd.factorize(d[groups])[0]
        G = g.max() + 1
        scores = np.zeros((G, X.shape[1]))
        np.add.at(scores, g, X * u[:, None])
        N, K = X.shape
        cov = bread @ (scores.T @ scores) @ bread * (G / (G - 1)) * ((N - 1) / (N - K))
        return float(beta[-1]), float(np.sqrt(cov[-1, -1])), N

    def slope(x, y):
        xc = x - x.mean()
        return (xc * (y - y.mean())).sum() / (xc ** 2).sum()

    # focal-level S and VR for the five groupings
    recs = []
    for t in TY:
        P = POOLS[t]
        for n in (2, 5, 10):
            cols = {}
            for wt in ("spending", "equal"):
                cols[f"VR_{wt}"] = agg_focal(P, pool_vals(P, n, lambda Mn: vr(Mn, wt)))
                for pik in ("pi", "pi_early"):
                    cols[f"S_{wt}" + ("" if pik == "pi" else "_early")] = agg_focal(P, pool_vals(P, n, lambda Mn: gS(Mn, wt, pik)))
            d = pd.DataFrame(cols)
            d["type"] = t
            d["n"] = n
            d["focal"] = d.index
            recs.append(d)
    FC = pd.concat(recs, ignore_index=True)
    FC["city"] = [hosts[i] for i in FC.focal]
    FC["arrangement"] = FC.type.map(NAME)
    FC["group"] = FC.type.map(GROUP)

    # East-West cooperation groups (2021-present list, western recipients, sample cities only)
    ew = pd.read_csv(os.path.join(INTERMEDIATE, "fig5_eastwest_pairs.csv"), encoding=ENC)
    for c in ["east_unit", "west_unit"]:
        ew[c] = ew[c].astype(str).str.strip()
    e21 = ew[(ew.period == "2021-present") & (ew.recipient_is_western_province == True)]
    ok = e21[e21.east_unit.isin(hix) & e21.west_unit.isin(hix)]
    grp = {}
    for e, g in ok.groupby("east_unit", sort=True):
        ps_ = [w for w in dict.fromkeys(g.west_unit) if w != e]
        grp[e] = [hix[e]] + [hix[w] for w in ps_]
    EW = []
    for e, m in grp.items():
        Mn = np.array([m])
        row = dict(east_city=e, east_province=ok[ok.east_unit == e].east_province.iloc[0], group_size=len(m), western_partners=SEP.join(hosts[j] for j in m[1:]),
                   n_east_partners_total_2021=int((e21.east_unit == e).sum()))
        for wt in ("spending", "equal"):
            row[f"S_{wt}"] = gS(Mn, wt)[0]
            row[f"S_{wt}_early"] = gS(Mn, wt, "pi_early")[0]
            row[f"VR_{wt}"] = vr(Mn, wt)[0]
        row["mean_distance_km"] = I["DIST"][m[0], m[1:]].mean()
        EW.append(row)
    EW = pd.DataFrame(EW)

    # grouping table
    inv = {v: k for k, v in T4LAB.items()}
    AT = []
    for n in (5, 10):
        for wt in ("spending", "equal"):
            for t in TY:
                d = FC[(FC.type == t) & (FC.n == n)]
                v = d[f"VR_{wt}"].dropna()
                s = d.loc[v.index, f"S_{wt}"]
                r = ps[(ps.weights == wt) & (ps.n == n) & (ps.pool_type == inv[t]) & (ps.statistic == "VR")]
                AT.append(dict(arrangement=NAME[t], group=GROUP[t], weighting=wt, n=n, N_focal=len(v), S_median=s.median(), S_p25=s.quantile(.25), S_p75=s.quantile(.75),
                               VR_median=v.median(), VR_ci_lo=r.median_ci_lo.iloc[0] if len(r) else np.nan, VR_ci_hi=r.median_ci_hi.iloc[0] if len(r) else np.nan,
                               VR_median_T4=r["median"].iloc[0] if len(r) else np.nan, N_focal_T4=r.N_focal.iloc[0] if len(r) else np.nan,
                               share_VR_gt_1=(v > 1 + 1e-12).mean(),
                               CI_source="T4 group-statistics table (999 year-month block draws)" if len(r) else "none: T4 did not run this weighting at this n", note=""))
            AT.append(dict(arrangement=NAME["eastwest"], group=GROUP["eastwest"], weighting=wt, n=n, N_focal=len(EW), S_median=EW[f"S_{wt}"].median(),
                           S_p25=EW[f"S_{wt}"].quantile(.25), S_p75=EW[f"S_{wt}"].quantile(.75), VR_median=EW[f"VR_{wt}"].median(), VR_ci_lo=np.nan, VR_ci_hi=np.nan,
                           VR_median_T4=np.nan, N_focal_T4=np.nan, share_VR_gt_1=(EW[f"VR_{wt}"] > 1 + 1e-12).mean(), CI_source="none (not in T4)",
                           note=f"not an n={n} group: actual group sizes " + str(EW.group_size.value_counts().sort_index().to_dict()) + "; same rows repeated for both n"))
    AT = pd.DataFrame(AT)
    AT["VR_median_diff_vs_T4"] = AT.VR_median - AT.VR_median_T4

    # within-grouping slopes of VR on S
    EWl = EW.assign(type="eastwest", focal=EW.east_city.map(hix), n=0)
    rng = np.random.default_rng(20261112)
    WS = []
    for n in (5, 10):
        for wt in ("spending", "equal"):
            for sv in ("", "_early"):
                xs, ys = f"S_{wt}{sv}", f"VR_{wt}"
                svl = "pi 1981-2016" if sv == "" else "pi_early 1981-1998 (out-of-sample)"
                for t in TY + ["eastwest"]:
                    if t == "eastwest" and n == 10:
                        continue
                    d = (FC[(FC.type == t) & (FC.n == n)] if t != "eastwest" else EWl)[[xs, ys, "focal"]].dropna()
                    x = d[xs].to_numpy()
                    y = d[ys].to_numpy()
                    z0 = float((x == 0).mean())
                    base = dict(arrangement=NAME[t], weighting=wt, n=(n if t != "eastwest" else "own size (39 of 44 groups have 2 members)"), S_version=svl,
                                N_focal=len(d), share_S_zero=z0)
                    if z0 > 0.9 or len(d) < 10:
                        WS.append(dict(base, status="not estimable: S = 0 for %.0f%% of focal cities" % (100 * z0)))
                        continue
                    b = slope(x, y)
                    bs = np.array([slope(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))])
                    q = pd.qcut(pd.Series(x).rank(method="first"), 3, labels=False).values
                    tm = pd.Series(y).groupby(q).median()
                    WS.append(dict(base, status="estimated" if t != "eastwest" else "estimated, descriptive only (mixed group sizes; S = 0 for most groups)",
                                   slope_per_0p1S=b * 0.1, ci_lo=np.nanquantile(bs, .025) * 0.1, ci_hi=np.nanquantile(bs, .975) * 0.1,
                                   spearman=spearmanr(x, y)[0], spearman_p=spearmanr(x, y)[1], VR_median_T1=tm.get(0), VR_median_T2=tm.get(1), VR_median_T3=tm.get(2),
                                   S_T1_range=f"{x[q == 0].min():.3f}-{x[q == 0].max():.3f}", S_T2_range=f"{x[q == 1].min():.3f}-{x[q == 1].max():.3f}",
                                   S_T3_range=f"{x[q == 2].min():.3f}-{x[q == 2].max():.3f}", CI_method="999 focal-city bootstrap draws (seed 20261112)"))
                d = FC[FC.n == n][["type", xs, ys, "focal"]].dropna().rename(columns={xs: "S", ys: "VR"})
                b_s, se_s, n_obs = ols_cluster(d, "VR", "S", "type", "focal")
                z = norm.ppf(0.975)
                WS.append(dict(arrangement="All five T4 arrangements, arrangement fixed effects", weighting=wt, n=n, S_version=svl, N_focal=n_obs,
                               status="estimated", slope_per_0p1S=b_s * 0.1, ci_lo=(b_s - z * se_s) * 0.1, ci_hi=(b_s + z * se_s) * 0.1,
                               spearman_p=2 * norm.sf(abs(b_s / se_s)), CI_method=f"OLS, SE clustered by focal city ({d.focal.nunique()} clusters)"))
    WS = pd.DataFrame(WS)

    # East-West groups against two-member groups of the other groupings
    cmp2 = []
    for wt in ("spending", "equal"):
        for t in TY:
            d = FC[(FC.type == t) & (FC.n == 2)]
            v = d[f"VR_{wt}"].dropna()
            cmp2.append(dict(arrangement=NAME[t], group=GROUP[t], weighting=wt, group_size="2", N=len(v), S_median=d.loc[v.index, f"S_{wt}"].median(),
                             VR_median=v.median(), VR_p25=v.quantile(.25), VR_p75=v.quantile(.75), share_VR_gt_1=(v > 1 + 1e-12).mean()))
        for lab, sub in (("2", EW[EW.group_size == 2]), ("all sizes (2-7)", EW)):
            v = sub[f"VR_{wt}"]
            cmp2.append(dict(arrangement=NAME["eastwest"], group=GROUP["eastwest"], weighting=wt, group_size=lab, N=len(v), S_median=sub[f"S_{wt}"].median(),
                             VR_median=v.median(), VR_p25=v.quantile(.25), VR_p75=v.quantile(.75), share_VR_gt_1=(v > 1 + 1e-12).mean()))
    CMP2 = pd.DataFrame(cmp2)
    CMP2["flag"] = np.where(CMP2.N < 30, "small N", "")
    FCo = FC[["city", "focal", "arrangement", "group", "type", "n", "S_spending", "S_spending_early", "VR_spending", "S_equal", "S_equal_early", "VR_equal"]].rename(
        columns={"type": "T4_design_code"})

    # excess peak shortfall X1 by group and member count
    mx = np.maximum(L.max(0), 0.0)
    ROWX, PEAK = {}, {}
    for t in TY:
        P = POOLS[t]
        M = P["M"]
        d = P["draws"]
        size = np.repeat(P["size"], d)
        Rr = len(P["focal"])
        Pn = len(M)
        M0 = np.where(M >= 0, M, 0)
        for wt in ("spending", "equal"):
            W = np.where(M >= 0, WSP[M0] if wt == "spending" else 1.0, 0.0)
            X = np.full((19, Pn), np.nan)
            PK = np.full((19, Pn), -1, dtype=np.int32)
            for c0 in range(0, Pn, 8000):
                sl = slice(c0, min(c0 + 8000, Pn))
                Mc = M0[sl]
                Wc = W[sl]
                sz = size[sl]
                S_ = L[:, Mc[:, 0]] * Wc[:, 0]
                den = Wc[:, 0] * mx[Mc[:, 0]]
                top = den.copy()
                for n in range(2, 21):
                    a = Wc[:, n - 1] * mx[Mc[:, n - 1]]
                    S_ += L[:, Mc[:, n - 1]] * Wc[:, n - 1]
                    den = den + a
                    top = np.maximum(top, a)
                    okn = (sz >= n) & (den > 0)
                    if not okn.any():
                        continue
                    pmax = np.maximum(S_.max(0), 0)
                    X[n - 2, sl] = np.where(okn, (pmax - top) / np.where(den > 0, den, 1), np.nan)
                    PK[n - 2, sl] = np.where(okn & (pmax > 0), S_.argmax(0), -1)
            xv = X.reshape(19, Rr, d)
            okk = np.isfinite(xv)
            cnt = okk.sum(2)
            ROWX[(t, wt)] = np.where(cnt > 0, np.where(okk, xv, 0).sum(2) / np.maximum(cnt, 1), np.nan)
            PEAK[(t, wt)] = PK

    # peak day of each city cluster
    ev = pd.read_csv(os.path.join(INTERMEDIATE, "T4_event_losses.csv"), encoding=ENC)
    ev["city_event_start"] = pd.to_datetime(ev.city_event_start)
    ev["city_event_end"] = pd.to_datetime(ev.city_event_end)
    dates = I["dates"]
    Pc = POOLS["cluster"]
    lab = np.array(Pc["label"])
    memb = POOLS["members"]
    d = Pc["draws"]
    rows = []
    for cname in sorted(memb, key=lambda c: -len(memb[c])):
        s = len(memb[cname])
        if s < 5:
            continue
        rr = np.where(lab == cname)[0]
        nmax = min(s, 20)
        for wt in ("spending", "equal"):
            for n in range(2, nmax + 1):
                v = ROWX[("cluster", wt)][n - 2][rr]
                v = v[np.isfinite(v)]
                row = dict(cluster=cname, national_level_cluster=cname in NAT5, n_sample_members=s, weighting=wt, n=n, N_focal=len(v),
                           X1_median=np.median(v) if len(v) else np.nan, X1_p25=np.quantile(v, .25) if len(v) else np.nan,
                           X1_p75=np.quantile(v, .75) if len(v) else np.nan, largest_n=(n == nmax))
                if n == nmax:
                    pk = PEAK[("cluster", wt)][n - 2][np.concatenate([np.arange(r * d, (r + 1) * d) for r in rr])]
                    pk = pk[pk >= 0]
                    if len(pk):
                        vc = pd.Series(pk).value_counts()
                        day = int(vc.index[0])
                        dt = dates[day]
                        mem_names = {hosts[m] for m in memb[cname]}
                        e = ev[ev.city_name.isin(mem_names) & (ev.city_event_start <= dt) & (ev.city_event_end >= dt)]
                        sysid = e.meteorological_event_id.mode().iloc[0] if len(e) else ""
                        row.update(peak_date=str(dt.date()), peak_date_share_of_draws=vc.iloc[0] / len(pk), n_groups_at_largest_n=len(pk),
                                   rainfall_system_id=sysid, tropical_cyclone_name="",
                                   n_member_city_events_on_peak_day=len(e), members_with_event_on_peak_day=SEP.join(sorted(e.city_name.unique())))
                rows.append(row)
    PC = pd.DataFrame(rows)
    PC["note"] = np.where(PC.n_sample_members > 20, "groups capped at 20 members as in T4; largest n = 20 < cluster size", "")

    def tcname(cname, dt_str, sysid):
        if not sysid:
            return ""
        mem_names = {hosts[m] for m in memb[cname]}
        dt = pd.Timestamp(dt_str)
        e = ev[ev.city_name.isin(mem_names) & (ev.city_event_start <= dt) & (ev.city_event_end >= dt) & (ev.meteorological_event_id == sysid) & (ev.TC500 == 1)]
        t_ = [x for x in e.TCname.dropna().astype(str) if x and x != "nan"]
        return pd.Series(t_).mode().iloc[0] if t_ else ""

    m_ = PC.largest_n & PC.peak_date.notna()
    PC.loc[m_, "tropical_cyclone_name"] = [tcname(r.cluster, r.peak_date, r.rainfall_system_id) for r in PC[m_].itertuples()]
    PC.loc[m_, "tropical_cyclone_rule"] = "name given only if the system has a cyclone track within 500 km of a member city (T4 TC500)"
    PC.insert(0, "arrangement", NAME["cluster"])

    def x1_group(m, wt):
        w = WSP[m] if wt == "spending" else np.ones(len(m))
        a = w * mx[m]
        den = a.sum()
        if den <= 0:
            return np.nan
        return (max((L[:, m] * w).sum(1).max(), 0) - a.max()) / den

    for wt in ("spending", "equal"):
        EW[f"X1_{wt}"] = [x1_group(np.array(grp[e]), wt) for e in EW.east_city]

    SUMR = []
    for wt in ("spending", "equal"):
        for t in TY:
            for n in range(2, 21):
                f = focal_from_rows(t, ROWX[(t, wt)][n - 2]).dropna()
                if len(f) == 0:
                    continue
                r = dict(arrangement=NAME[t], group=GROUP[t], weighting=wt, n=n, N_focal=len(f), X1_median=f.median(), X1_p25=f.quantile(.25), X1_p75=f.quantile(.75))
                if t == "cluster":
                    cm = []
                    for cname in memb:
                        rr = np.where(lab == cname)[0]
                        v = ROWX[("cluster", wt)][n - 2][rr]
                        v = v[np.isfinite(v)]
                        if len(v):
                            cm.append(np.median(v))
                    r.update(n_clusters=len(cm), cluster_median_min=min(cm), cluster_median_max=max(cm))
                    r["flag"] = "n_clusters < 8" if len(cm) < 8 else ""
                else:
                    r["flag"] = "N_focal < 60" if len(f) < 60 else ""
                SUMR.append(r)
        for n, g in EW.groupby("group_size"):
            v = g[f"X1_{wt}"].dropna()
            SUMR.append(dict(arrangement=NAME["eastwest"], group=GROUP["eastwest"], weighting=wt, n=n, N_focal=len(v), X1_median=v.median() if len(v) else np.nan,
                             X1_p25=v.quantile(.25) if len(v) else np.nan, X1_p75=v.quantile(.75) if len(v) else np.nan, flag="N_focal < 60"))
    SUMR = pd.DataFrame(SUMR)

    bs_ = sorted(set(focal_from_rows("cluster", ROWX[("cluster", "equal")][18]).dropna().index))
    BAL = []
    for wt in ("spending", "equal"):
        for n in range(2, 21):
            f = focal_from_rows("cluster", ROWX[("cluster", wt)][n - 2]).reindex(bs_)
            BAL.append(dict(arrangement=NAME["cluster"], weighting=wt, n=n, N_focal_fixed_set=len(bs_), N_finite=int(f.notna().sum()),
                            X1_median=f.median(), X1_p25=f.quantile(.25), X1_p75=f.quantile(.75)))
    BAL = pd.DataFrame(BAL)

    EWo = EW.copy()
    EWo.insert(0, "period", "2021-present")
    EWo["flag"] = "small N" if len(EW) < 30 else ""
    EWo["note"] = "group = eastern sample city + its 2021 western partner units that are also sample cities; non-sample partners dropped, never substituted"
    outs = {"fig5c_arrangements_table.csv": AT, "fig5c_within_slopes.csv": WS, "fig5c_group_S_VR_by_city.csv": FCo, "fig5_eastwest_groups_VR.csv": EWo,
            "fig5_eastwest_vs_n2.csv": CMP2, "fig5d_peak_by_cluster.csv": PC, "fig5d_peak_summary.csv": SUMR, "fig5d_peak_balanced_cluster.csv": BAL}
    for f, df in outs.items():
        df.to_csv(os.path.join(INTERMEDIATE, f), index=False, encoding=ENC)
    print(AT[AT.n == 5][["arrangement", "weighting", "N_focal", "S_median", "VR_median"]].to_string(index=False))


def excess_peak_shortfall(argv=None):
    """Excess peak shortfall X_k = R_k - B_k of equal-weighted city groups (k = 1 and 3 worst days): the group's share of
    members' combined worst-day shortfall minus its value if no two members ever lost on the same day, for the five group
    designs and group sizes 2-20, with year-month block bootstrap intervals (999 draws)."""
    ENC = "utf-8-sig"
    SEED, NB = 20261110, 999
    NWORK = 28
    _I = None

    def draws(T, N, B=36):
        return np.random.default_rng([SEED, 1, B]).multinomial(B, np.full(B, 1 / B), size=NB)

    def _load():
        nonlocal _I
        if _I is None:
            I = pickle.load(open(os.path.join(INTERMEDIATE, "t4_inputs.pkl"), "rb"))
            L = I["LOSS"]["main"]
            T, N = L.shape
            _I = dict(L=L, T=T, KB=draws(T, N, 36), block=I["block"], POOLS=I["POOLS"])
        return _I

    def top_pos(X, K=None):
        T = X.shape[0]
        K = max(int((X > 0).sum(0).max()), 1) if K is None else K
        order = np.argsort(-X, axis=0, kind="stable")[:K]
        v = np.take_along_axis(X, order, 0)
        order = np.where(v > 0, order, T)
        v = np.where(v > 0, v, 0.0)
        return v.T.copy(), order.T.copy()

    def tail_from(v, di, mult, kb):
        m = mult[di]
        cum = np.cumsum(m, 1)
        take = np.clip(kb - (cum - m), 0, m)
        return (take * v).sum(1) / kb

    def run_job(t, n, chunk=2000):
        G = _load()
        L, T, KB, block = G["L"], G["T"], G["KB"], G["block"]
        P = G["POOLS"][t]
        M = P["M"]
        d = P["draws"]
        size = np.repeat(P["size"], d)
        ok = size >= n
        R = len(P["focal"])
        uf = np.unique(P["focal"])
        if not ok.any():
            return None
        mults = [np.append(np.ones(T), 0.0)] + [np.append(k[block], 0.0).astype(float) for k in KB]
        S = len(mults)
        ks = (1, 3)
        vm, dm = top_pos(L, 11)
        MEMT = [{k: tail_from(vm, dm, m_, k) for k in ks} for m_ in mults]
        acc = {(q, k): np.zeros((S, R)) for q in ("X", "R", "B") for k in ks}
        cnt = {k: np.zeros((S, R)) for k in ks}
        idx = np.where(ok)[0]
        for c in range(0, len(idx), chunk):
            ii = idx[c:c + chunk]
            Mm = M[ii, :n]
            row = ii // d
            v, di = top_pos(L[:, Mm].mean(-1))
            bv = vm[Mm].reshape(len(ii), -1) / n
            bd = dm[Mm].reshape(len(ii), -1)
            o = np.argsort(-bv, axis=1, kind="stable")
            bv = np.take_along_axis(bv, o, 1)
            bd = np.take_along_axis(bd, o, 1)
            for s, m_ in enumerate(mults):
                for k in ks:
                    den = MEMT[s][k][Mm].mean(1)
                    good = den > 0
                    dd = np.where(good, den, 1)
                    r = tail_from(v, di, m_, k) / dd
                    b = tail_from(bv, bd, m_, k) / dd
                    x = r - b
                    np.add.at(acc[("X", k)][s], row, np.where(good, x, 0))
                    np.add.at(acc[("R", k)][s], row, np.where(good, r, 0))
                    np.add.at(acc[("B", k)][s], row, np.where(good, b, 0))
                    np.add.at(cnt[k][s], row, good.astype(float))
        RF = (P["focal"][:, None] == uf[None, :]).astype(float)
        res = {}
        for (q, k), a in acc.items():
            c_ = cnt[k]
            rowm = np.where(c_ > 0, a / np.maximum(c_, 1), np.nan)
            okr = np.isfinite(rowm)
            num = np.where(okr, rowm, 0) @ RF
            dn = okr.astype(float) @ RF
            res[f"{q}_{k}"] = np.where(dn > 0, num / np.maximum(dn, 1), np.nan)
        return res

    def run_worker(work, spec):
        for item in spec.split(","):
            t, n = item.split(":")
            z = run_job(t, int(n))
            if z is not None:
                np.savez(os.path.join(work, f"x_{t}_{n}.npz"), **z)

    def main():
        jobs = [(t, n) for n in range(20, 1, -1) for t in ["random", "province", "cluster"]] + [(t, n) for t in ["demand", "crossbelt"] for n in range(2, 21)]
        work = tempfile.mkdtemp(prefix="excess_peak_")
        me = os.path.abspath(__file__)
        procs = [subprocess.Popen([sys.executable, me, "excess_peak_shortfall", "worker", work, ",".join(f"{t}:{n}" for t, n in jobs[i::NWORK])]) for i in range(NWORK)]
        for p in procs:
            p.wait()
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError("worker failed")
        res = {}
        for t, n in jobs:
            fn = os.path.join(work, f"x_{t}_{n}.npz")
            res[(t, n)] = dict(np.load(fn)) if os.path.exists(fn) else None
        rows = []
        for (t, n), z in res.items():
            if z is None:
                continue
            for k in (1, 3):
                X = z[f"X_{k}"]
                pt = X[0]
                fin = np.isfinite(pt)
                bmed = np.nanmedian(X[1:], 1)
                bmean = np.nanmean(X[1:], 1)
                rows.append(dict(pool_type=t, n=n, metric=f"X{k}", k_days=k, N_focal=int(fin.sum()), flag_N_lt_60=bool(fin.sum() < 60),
                                 X_median=np.nanmedian(pt), ci_lo=np.nanquantile(bmed, .025), ci_hi=np.nanquantile(bmed, .975),
                                 X_mean=np.nanmean(pt), mean_ci_lo=np.nanquantile(bmean, .025), mean_ci_hi=np.nanquantile(bmean, .975),
                                 share_focal_X_gt_0=float(np.mean(pt[fin] > 1e-9)), R_median=np.nanmedian(z[f"R_{k}"][0][fin]), B_median=np.nanmedian(z[f"B_{k}"][0][fin]),
                                 point_outside_CI=bool((np.nanmedian(pt) < np.nanquantile(bmed, .025)) or (np.nanmedian(pt) > np.nanquantile(bmed, .975)))))
        XT = pd.DataFrame(rows).sort_values(["metric", "pool_type", "n"]).reset_index(drop=True)
        XT.insert(0, "definition", np.where(XT.metric == "X1", "X1 = R1 - B1 per pool (B1 = max_m max_t L_m / sum_m max_t L_m), focal mean over pools, median over focal; CIs: T4 999 year-month block draws, R and B recomputed per draw",
                                            "X3 = R3 - B3 per pool (B3 = top-3 mean of members positive losses as if on distinct days, /n, over members mean top-3)"))
        XT.to_csv(os.path.join(INTERMEDIATE, "fig5d_excess_reserve.csv"), index=False, encoding=ENC)
        print(XT[XT.n.isin([5, 10])][["metric", "pool_type", "n", "N_focal", "X_median", "ci_lo", "ci_hi"]].to_string(index=False))

    argv = argv or [sys.argv[0]]
    if len(argv) > 1 and argv[1] == "worker":
        run_worker(argv[2], argv[3])
    else:
        main()


def random_group_curve():
    """Shared-rain exposure S of random five-member city groups against their median member-to-member distance
    (radius-capped random draws near each sample city, binned means with a focal-city bootstrap band), and the
    distance and exposure of the five group designs at five members."""
    ENC = "utf-8-sig"
    DESIGN = ["cluster", "province", "demand", "random", "crossbelt"]
    BINS = [0, 150, 250, 350, 450, 600, 800, 1000, 1300, 1700, 2200, np.inf]
    CAPS = [200, 300, 400, 500, 700, 1000, 1500, 2500, np.inf]
    IU = np.triu_indices(5, 1)

    def random_curve(hosts, PI, D, seed=20261111, draws=50, nboot=999):
        """Random groups on the pi-based group S (S_pool): for each focal host and radius cap, 50 groups of the focal city plus
        4 members drawn uniformly (without replacement) from the other hosts within the cap; caps with < 4 candidates
        are skipped. S_pool = mean pi(focal, member); x = median of the 10 member-to-member distances."""
        rng = np.random.default_rng(seed)
        n = len(hosts)
        F, S, X = [], [], []
        for i in range(n):
            for cap in CAPS:
                cand = np.where((np.arange(n) != i) & (D[i] <= cap))[0]
                if cand.size < 4:
                    continue
                for _ in range(draws):
                    mem = rng.choice(cand, 4, replace=False)
                    pool = np.r_[i, mem]
                    F.append(i)
                    S.append(np.nanmean(PI[i, mem]))
                    X.append(np.median(D[np.ix_(pool, pool)][IU]))
        F, S, X = np.array(F), np.array(S), np.array(X)
        b = np.searchsorted(BINS, X, side="right") - 1
        nb = len(BINS) - 1
        labs = [f"{BINS[k]:.0f}-{BINS[k + 1]:.0f}" if np.isfinite(BINS[k + 1]) else f">{BINS[k]:.0f}" for k in range(nb)]

        def binmeans(w):
            cnt = np.bincount(b, weights=w, minlength=nb)
            return (np.bincount(b, weights=w * S, minlength=nb) / np.where(cnt > 0, cnt, np.nan),
                    np.bincount(b, weights=w * X, minlength=nb) / np.where(cnt > 0, cnt, np.nan))
        est, xm = binmeans(np.ones_like(S))
        boot = np.empty((nboot, nb))
        for r in range(nboot):
            k = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            boot[r] = binmeans(k[F])[0]
        out = pd.DataFrame({"bin_km": labs, "n_pools": np.bincount(b, minlength=nb),
                            "n_focal": [np.unique(F[b == k]).size for k in range(nb)], "mean_dist_km": xm,
                            "S_pool_mean": est, "S_lo": np.nanpercentile(boot, 2.5, axis=0),
                            "S_hi": np.nanpercentile(boot, 97.5, axis=0)})
        out.to_csv(os.path.join(INTERMEDIATE, "fig5b_random_pool_S_curve.csv"), index=False, encoding=ENC)
        return out, len(S)

    def design_points(hosts, PI, D, POOLS):
        """Five-member groups (focal + 4) of each design: S_pool checked against fig5c_S_by_pool_type.csv; x = per-group median
        member-to-member distance, averaged over draws per focal city, then over focal cities (same N as fig5c)."""
        sv = pd.read_csv(os.path.join(INTERMEDIATE, "fig5c_pool_S_vs_VR.csv"), encoding=ENC)
        sv = sv.loc[sv.VR_n5.notna()]
        sp = pd.read_csv(os.path.join(INTERMEDIATE, "fig5c_S_by_pool_type.csv"), encoding=ENC).query("n == 5").set_index("pool_type")
        rows = []
        for pt in DESIGN:
            d = POOLS[pt]
            F = np.asarray(d["focal"])
            M5 = np.asarray(d["M"])[:, :5]
            nd = M5.shape[0] // F.size
            keep = set(sv.loc[sv.pool_type == pt, "city_name"])
            s_, x_ = {}, {}
            for row, f in zip(M5, np.repeat(F, nd)):
                if hosts[f] not in keep:
                    continue
                s_.setdefault(f, []).append(PI[f, row[1:]].mean())
                x_.setdefault(f, []).append(np.median(D[np.ix_(row, row)][IU]))
            S = np.array([np.mean(v) for v in s_.values()])
            X = np.array([np.mean(x_[f]) for f in s_])
            rows.append({"pool_type": pt, "N_focal": len(S), "S_mean": S.mean(), "S_mean_csv": sp.loc[pt, "S_mean"],
                         "med_pair_km_mean": X.mean(), "med_pair_km_median": np.median(X),
                         "mean_focal_member_km_median_csv": sp.loc[pt, "mean_member_dist_km_median"]})
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(INTERMEDIATE, "fig5b_design_points.csv"), index=False, encoding=ENC)
        return out.set_index("pool_type")

    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    t4 = pickle.load(open(os.path.join(INTERMEDIATE, "t4_inputs.pkl"), "rb"))
    hosts = list(t4["hosts"])
    PI = hp.pivot(index="focal_city", columns="partner_city", values="pi").loc[hosts, hosts].to_numpy()
    D = hp.pivot(index="focal_city", columns="partner_city", values="distance_km").loc[hosts, hosts].to_numpy()
    curve, npools = random_curve(hosts, PI, D)
    dp = design_points(hosts, PI, D, t4["POOLS"])
    print(curve.to_string(index=False))
    print(dp.to_string())


def main():
    group_loss_inputs()
    group_loss_tests()
    decoupling_distance_by_city()
    group_exposure()
    east_west_groups()
    grouping_comparison()
    excess_peak_shortfall()
    random_group_curve()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("group_loss_tests", "excess_peak_shortfall"):
        globals()[sys.argv[1]]([sys.argv[0]] + sys.argv[2:])
    else:
        main()
