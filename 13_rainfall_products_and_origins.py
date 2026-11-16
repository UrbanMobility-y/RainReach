"""Rainfall-product checks of the shared-rain share and its consequences (Supplementary results supporting Figs 2
and 3): decay of same-day extreme-rain co-occurrence with distance for ERA5-Land, CPC and CHIRPS; origin-count
analysis by product; validation of S computed from either half of the rainfall record; and the lead-lag profile of
spending around rain at partner cities. Writes SI2_decay_curves_binned.csv, SI2_decay_L10_L05.csv,
SI2_origins_products_cityyear.csv, SI2_origins_products_regressions.csv, SI2_S_halfrecord_validation.csv and
SI2_origin_dynamics.csv.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import pyfixest as pf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
DATASET = ROOT / "Dataset"
INTERMEDIATE = ROOT / "Intermediate"
WEATHER_DIR = DATASET / "city daily weather"
ENC = "utf-8-sig"
HISTORY = (1981, 2016)
N_BOOT, SEED = 999, 20261112
CELLS = [("1981-2016", "all"), ("1981-1998", "all"), ("1999-2016", "all"), ("1981-2016", "AMJ"), ("1981-2016", "JAS")]
PRODUCTS = ("ERA5-EF", "CPC-EF", "CHIRPS-EF")


def need(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file missing: {path}")
    return path


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def era5_matrix():
    frames = []
    for y in range(HISTORY[0], HISTORY[1] + 1):
        b = pd.read_csv(need(WEATHER_DIR / f"city_daily_weather_{y}.csv"), encoding=ENC)
        b["city_name"] = b.city_name.astype(str).str.strip()
        frames.append(b)
    wx = pd.concat(frames, ignore_index=True)
    thr = pd.read_csv(need(INTERMEDIATE / "city_rainfall_thresholds.csv"), encoding=ENC)
    thr["city_name"] = thr.city_name.astype(str).str.strip()
    cut = dict(zip(thr.city_name, thr.return_level_3yr_mm.fillna(np.inf)))
    wx["H"] = ((wx.rain_mm >= 50.0) | (wx.rain_mm >= wx.city_name.map(cut))).astype(np.int8)
    wide = wx.pivot(index="date", columns="city_name", values="H").sort_index()
    assert not wide.isna().any().any()
    return wide.to_numpy(np.float64), list(wide.columns), pd.to_datetime(wide.index)


def ef_matrix(path, cities, dates, n_ext):
    """Equal-frequency extreme days: each unit's n_i largest values (missing days never extreme); NaN column if no data."""
    d = pd.read_parquet(need(path))
    d["city_name"] = d.city_name.astype(str).str.strip()
    d["date"] = pd.to_datetime(d.date)
    d = d.loc[d.date.dt.year.between(*HISTORY)]
    R = d.pivot(index="date", columns="city_name", values="rain_mm").reindex(index=dates, columns=cities).to_numpy()
    H = np.zeros(R.shape)
    for j in range(R.shape[1]):
        x = R[:, j]
        if np.isnan(x).all():
            H[:, j] = np.nan
            continue
        k = int(n_ext[j])
        if k > 0:
            H[np.argsort(-np.where(np.isnan(x), -np.inf, x), kind="stable")[:k], j] = 1.0
    return H


def pava_dec(y, w):
    """Weighted non-increasing isotonic regression (pool-adjacent-violators)."""
    y = -np.asarray(y, float)
    w = np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww if ww > 0 else (v1 + v2) / 2); wts.append(ww); cnt.append(c1 + c2)
    return -np.repeat(vals, cnt)


def decay_by_product():
    t0 = time.time()
    He, cities, dates = era5_matrix()
    n_ext = He.sum(0)
    H = {"ERA5-EF": He,
         "CPC-EF": ef_matrix(INTERMEDIATE / "city_daily_cpc_1981_2019.parquet", cities, dates, n_ext),
         "CHIRPS-EF": ef_matrix(INTERMEDIATE / "city_daily_chirps_1981_2019.parquet", cities, dates, n_ext)}
    missing = {p: [c for j, c in enumerate(cities) if np.isnan(h[:, j]).all()] for p, h in H.items()}
    totals = {p: int(np.nansum(h)) for p, h in H.items()}
    print(f"extreme city-days: {totals}; units without data: {missing}; {time.time() - t0:.0f} s")

    grid = pd.read_parquet(need(INTERMEDIATE / "city_grid_weights.parquet"))
    grid["city_name"] = grid.city_name.astype(str).str.strip()
    c = grid.assign(wlat=grid.latitude * grid.weight, wlon=grid.longitude * grid.weight).groupby("city_name")[
        ["wlat", "wlon", "weight"]].sum()
    la, lo = (c.wlat / c.weight).reindex(cities).to_numpy(), (c.wlon / c.weight).reindex(cities).to_numpy()
    D = haversine(la[:, None], lo[:, None], la[None, :], lo[None, :])
    n = len(cities)
    I, J = np.where(~np.eye(n, dtype=bool))
    LD = np.log(np.maximum(D[I, J], 5.0))
    edges = np.unique(np.quantile(LD, np.linspace(0, 1, int(min(60, len(I) // 150)) + 1)))
    b = np.clip(np.searchsorted(edges, LD, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    ctr = np.bincount(b, weights=LD, minlength=nb) / np.bincount(b, minlength=nb)

    def curve(piv):
        ok = np.isfinite(piv)
        nn = np.bincount(b[ok], minlength=nb).astype(float)
        s = np.bincount(b[ok], weights=piv[ok], minlength=nb)
        mean = np.divide(s, nn, out=np.full(nb, np.nan), where=nn > 0)
        g = nn > 0
        fit = np.full(nb, np.nan)
        fit[g] = pava_dec(mean[g], nn[g])
        return mean, fit, nn

    def cross(fit, c_):
        g = np.isfinite(fit)
        x, f = ctr[g], fit[g]
        k = np.where(f < c_)[0]
        if len(k) == 0:
            return np.inf
        k = k[0]
        if k == 0:
            return 0.0
        return float(np.exp(x[k - 1] + (f[k - 1] - c_) * (x[k] - x[k - 1]) / (f[k - 1] - f[k])))

    years = dates.year.to_numpy()
    month = dates.month.to_numpy()
    season_mask = {"all": np.ones(len(dates), bool), "AMJ": np.isin(month, (4, 5, 6)), "JAS": np.isin(month, (7, 8, 9))}
    yrs = np.arange(HISTORY[0], HISTORY[1] + 1)
    # per-year co-count tensors (pairs I, J) and own counts, by product and season
    CO, OWN, OK = {}, {}, {}
    for p, h in H.items():
        okcol = ~np.isnan(h).all(0)
        hz = np.nan_to_num(h)
        OK[p] = okcol[I] & okcol[J]
        for se, sm in season_mask.items():
            co = np.empty((len(yrs), len(I)))
            own = np.empty((len(yrs), n))
            for k, y in enumerate(yrs):
                blk = hz[sm & (years == y)]
                co[k] = (blk.T @ blk)[I, J]
                own[k] = blk.sum(0)
            CO[(p, se)], OWN[(p, se)] = co, own

    def pi_from(p, se, kvec):
        co = kvec @ CO[(p, se)]
        own = (kvec @ OWN[(p, se)])[I]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = co / own
        out[(own <= 0) | ~OK[p]] = np.nan
        return out

    rng = np.random.default_rng(SEED)
    curves, Ltab = [], []
    for period, season in CELLS:
        lo_y, hi_y = map(int, period.split("-"))
        inside = (yrs >= lo_y) & (yrs <= hi_y)
        base = inside.astype(float)
        pool = np.where(inside)[0]
        draws = [np.bincount(rng.choice(pool, size=len(pool), replace=True), minlength=len(yrs)).astype(float)
                 for _ in range(N_BOOT)]
        for p in PRODUCTS:
            piv = pi_from(p, season, base)
            mean, fit, nn = curve(piv)
            for kb in range(nb):
                curves.append({"product": p, "period": period, "season": season, "subset": "national", "bin": kb,
                               "distance_km": float(np.exp(ctr[kb])), "pi_mean": mean[kb], "pi_iso": fit[kb],
                               "n_pairs": int(nn[kb])})
            L10, L05 = cross(fit, 0.10), cross(fit, 0.05)
            bt = np.array([[cross(curve(pi_from(p, season, kv))[1], c_) for c_ in (0.10, 0.05)] for kv in draws])
            row = {"product": p, "period": period, "season": season, "subset": "national", "L10": L10, "L05": L05,
                   "n_pairs": int(np.isfinite(piv).sum()), "pi_first_bin": fit[0]}
            for k, nm in enumerate(("L10", "L05")):
                v = bt[:, k]
                fin = np.isfinite(v) & (v > 0)
                row[f"{nm}_lo"], row[f"{nm}_hi"] = np.percentile(v[fin], [2.5, 97.5])
                row[f"{nm}_boot_out_of_range"] = int((~fin).sum())
            Ltab.append(row)
            print(f"  {p:9s} {period} {season}: L10 {L10:.0f} [{row['L10_lo']:.0f}, {row['L10_hi']:.0f}], "
                  f"L05 {L05:.0f} [{row['L05_lo']:.0f}, {row['L05_hi']:.0f}] km ({time.time() - t0:.0f} s)")
    curves, Ltab = pd.DataFrame(curves), pd.DataFrame(Ltab)
    curves.to_csv(INTERMEDIATE / "SI2_decay_curves_binned.csv", index=False, encoding=ENC)
    Ltab.to_csv(INTERMEDIATE / "SI2_decay_L10_L05.csv", index=False, encoding=ENC)

    saved_L = pd.read_csv(need(INTERMEDIATE / "T1_decorrelation_L10_L05.csv"))
    saved_curves = pd.read_csv(need(INTERMEDIATE / "T1_decay_curves_binned.csv"))
    worst_pt, worst_curve, worst_ci = 0.0, 0.0, 0.0
    for p in ("ERA5-EF", "CPC-EF"):
        for period, season in CELLS:
            r = saved_L.loc[(saved_L["product"] == p) & (saved_L.period == period) & (saved_L.season == season) & (saved_L.subset == "national")]
            m = Ltab.loc[(Ltab["product"] == p) & (Ltab.period == period) & (Ltab.season == season)]
            assert len(r) == 1 and len(m) == 1
            r, m = r.iloc[0], m.iloc[0]
            dpt = max(abs(r.L10 - m.L10), abs(r.L05 - m.L05))
            dci = max(abs(r[c] - m[c]) for c in ("L10_lo", "L10_hi", "L05_lo", "L05_hi"))
            cx = saved_curves.loc[(saved_curves["product"] == p) & (saved_curves.period == period) & (saved_curves.season == season) & (saved_curves.subset == "national")].sort_values("bin")
            cm = curves.loc[(curves["product"] == p) & (curves.period == period) & (curves.season == season)].sort_values("bin")
            dcu = float(np.nanmax(np.abs(cx.pi_iso.to_numpy() - cm.pi_iso.to_numpy())))
            worst_pt, worst_curve, worst_ci = max(worst_pt, dpt), max(worst_curve, dcu), max(worst_ci, dci)
    assert worst_pt < 0.05 and worst_curve < 1e-6, (worst_pt, worst_curve)
    print(f"wrote SI2_decay_curves_binned.csv, SI2_decay_L10_L05.csv "
          f"(ERA5-Land and CPC vs T1_decorrelation_L10_L05.csv and T1_decay_curves_binned.csv: points {worst_pt:.2e} km, curves {worst_curve:.1e}, CIs {worst_ci:.1f} km; {time.time() - t0:.0f} s)")


NETWORK_YEARS = (2017, 2018)
PHI_DAYS = 72


def weather():
    frames = []
    for y in range(HISTORY[0], HISTORY[1] + 1):
        b = pd.read_csv(need(WEATHER_DIR / f"city_daily_weather_{y}.csv"), encoding=ENC)
        b["city_name"] = b.city_name.astype(str).str.strip()
        frames.append(b)
    wx = pd.concat(frames, ignore_index=True)
    thr = pd.read_csv(need(INTERMEDIATE / "city_rainfall_thresholds.csv"), encoding=ENC)
    thr["city_name"] = thr.city_name.astype(str).str.strip()
    cut = dict(zip(thr.city_name, thr.return_level_3yr_mm.fillna(np.inf)))
    wx["H"] = ((wx.rain_mm >= 50.0) | (wx.rain_mm >= wx.city_name.map(cut))).astype(np.int8)
    H = wx.pivot(index="date", columns="city_name", values="H").sort_index()
    R = wx.pivot(index="date", columns="city_name", values="rain_mm").sort_index()
    assert list(H.columns) == list(R.columns) and not H.isna().any().any()
    return H.to_numpy(float), R.to_numpy(float), list(H.columns), pd.to_datetime(H.index)


def top_days(R, counts):
    """Indicator of each unit's counts[j] largest days (missing days never selected); NaN column if no data."""
    H = np.zeros(R.shape)
    for j in range(R.shape[1]):
        x = R[:, j]
        if np.isnan(x).all():
            H[:, j] = np.nan
            continue
        k = int(counts[j])
        if k > 0:
            H[np.argsort(-np.where(np.isnan(x), -np.inf, x), kind="stable")[:k], j] = 1.0
    return H


def product_rain(path, cities, dates):
    d = pd.read_parquet(need(path))
    d["city_name"] = d.city_name.astype(str).str.strip()
    d["date"] = pd.to_datetime(d.date)
    d = d.loc[d.date.dt.year.between(*HISTORY)]
    return d.pivot(index="date", columns="city_name", values="rain_mm").reindex(index=dates, columns=cities).to_numpy()


def ols_crv(d, y, xs, cl="city_id"):
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    Y = d[y].to_numpy(float)
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ Y
    u = Y - X @ b
    g = pd.factorize(d[cl])[0]
    G = g.max() + 1
    sc = np.zeros((G, X.shape[1]))
    np.add.at(sc, g, X * u[:, None])
    N, K = X.shape
    V = XtX @ (sc.T @ sc) @ XtX * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(V))
    tc = stats.t.ppf(0.975, G - 1)
    p = 2 * stats.t.sf(np.abs(b / se), G - 1)
    r2 = 1 - (u ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    return pd.DataFrame({"term": ["Intercept"] + xs, "est": b, "se": se, "ci_lo": b - tc * se, "ci_hi": b + tc * se,
                         "p": p, "N": N, "G": G, "R2": r2})


def build(net, H, H72, cities, product):
    cix = {c: k for k, c in enumerate(cities)}
    ok = ~np.isnan(H).all(0)
    Hz = np.nan_to_num(H)
    rows = []
    for (host, year), g in net.groupby(["host_name", "network_year"]):
        if host not in cix:
            continue
        g = g.loc[g.guest_name.map(lambda x: x in cix and ok[cix[x]])]
        i = cix[host]
        js = np.array([cix[x] for x in g.guest_name])
        w = g.w.to_numpy() / g.w.sum()
        days = Hz[:, i] == 1
        A = Hz[days][:, js] @ w
        X = np.nan_to_num(H72[:, js])
        Phi = np.corrcoef(X.T) if len(js) > 1 else np.ones((1, 1))
        Phi = np.nan_to_num(np.atleast_2d(Phi))
        np.fill_diagonal(Phi, 1.0)
        dist = g.distance_km.to_numpy()
        rows.append({"product": product, "city_id": g.host_id.iloc[0], "city_name": host, "network_year": year,
                     "n_focal_days": int(days.sum()), "k": len(js), "S": A.mean(), "sdA": A.std(ddof=0),
                     "tailA": (A >= 0.5).mean(), "N_eff_u": 1 / (w ** 2).sum(), "N_ind": 1 / (w @ Phi @ w),
                     "logd": np.log((w * dist).sum()), "Near300": w[dist <= 300].sum()})
    d = pd.DataFrame(rows)
    d["ratio_ind"] = d.N_ind / d.N_eff_u
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    d["zlnN"], d["zlnNind"], d["zS"] = z(np.log(d.N_eff_u)), z(np.log(d.N_ind)), z(d.S)
    d["zS2"] = d.zS ** 2
    for c in ("S", "sdA", "tailA"):
        d[f"{'meanA' if c == 'S' else c}_pp"] = 100 * d[c]
    return d


FORMULAS = {"a_mean": "meanA_pp ~ zlnN + logd + Near300", "a_mean_noGeo": "meanA_pp ~ zlnN",
            "b_sd": "sdA_pp ~ zlnN + zS + zS2", "b_tail": "tailA_pp ~ zlnN + zS + zS2",
            "b_sd_geo": "sdA_pp ~ zlnN + zS + zS2 + logd + Near300",
            "b_tail_geo": "tailA_pp ~ zlnN + zS + zS2 + logd + Near300",
            "desc_sd_Nind": "sdA_pp ~ zlnNind + zS + zS2", "desc_tail_Nind": "tailA_pp ~ zlnNind + zS + zS2"}


def origins_by_product():
    He, Re, cities, dates = weather()
    counts = He.sum(0)
    Rc = product_rain(INTERMEDIATE / "city_daily_cpc_1981_2019.parquet", cities, dates)
    Rh = product_rain(INTERMEDIATE / "city_daily_chirps_1981_2019.parquet", cities, dates)
    MAT = {"ERA5-Land": (He, top_days(Re, np.full(len(cities), PHI_DAYS))),
           "CPC": (top_days(Rc, counts), top_days(Rc, np.full(len(cities), PHI_DAYS))),
           "CHIRPS": (top_days(Rh, counts), top_days(Rh, np.full(len(cities), PHI_DAYS)))}
    net = pd.read_csv(need(INTERMEDIATE / "partner_network.csv"), encoding=ENC,
                      dtype={"host_id": str, "guest_id": str}, low_memory=False)
    net = net.loc[(net.usable == 1) & net.network_year.isin(NETWORK_YEARS)].copy()
    cy = pd.concat([build(net, H, H72, cities, p) for p, (H, H72) in MAT.items()], ignore_index=True)
    reg = []
    for p, d in cy.groupby("product", sort=False):
        for test, f in FORMULAS.items():
            y, rhs = f.split(" ~ ")
            r = ols_crv(d, y, rhs.split(" + "))
            reg.append(r.assign(product=p, test=test, formula=f))
    reg = pd.concat(reg, ignore_index=True)[["product", "test", "formula", "term", "est", "se", "ci_lo", "ci_hi", "p",
                                             "N", "G", "R2"]]

    saved_cy = pd.read_csv(need(INTERMEDIATE / "B3_portfolio_cityyear.csv"))
    saved_reg = pd.read_csv(need(INTERMEDIATE / "B3_historical_portfolio_regressions.csv"))
    e = cy.loc[cy["product"] == "ERA5-Land"].merge(saved_cy, on=["city_name", "network_year"], suffixes=("", "_saved"))
    assert len(e) == len(saved_cy) == 448
    dev_cy = max(float(np.abs(e[c] - e[c + "_saved"]).max()) for c in
                 ("n_focal_days", "k", "sdA", "tailA", "N_eff_u", "N_ind", "logd", "Near300", "zlnN", "zS"))
    dev_cy = max(dev_cy, float(np.abs(e.S - e.meanA).max()))
    re_ = reg.loc[reg["product"] == "ERA5-Land"].merge(saved_reg, on=["test", "term"], suffixes=("", "_saved"))
    assert set(re_.test) == set(FORMULAS) and len(re_) == len(saved_reg.loc[saved_reg.test.isin(list(FORMULAS))])
    dev_reg = float(np.abs(re_[["est", "se", "p"]].to_numpy() - re_[["est_saved", "se_saved", "p_saved"]].to_numpy()).max())
    assert dev_cy < 1e-6 and dev_reg < 1e-6, (dev_cy, dev_reg)
    # zS differs from the saved ERA5-Land column by < 3e-8 (S there was computed from float32 pi)
    cy.to_csv(INTERMEDIATE / "SI2_origins_products_cityyear.csv", index=False, encoding=ENC)
    reg.to_csv(INTERMEDIATE / "SI2_origins_products_regressions.csv", index=False, encoding=ENC)

    def line(p):
        q = reg.loc[(reg["product"] == p) & (reg.term == "zlnN")].set_index("test")
        m = cy.loc[cy["product"] == p]
        f = lambda t: f"{q.loc[t, 'est']:+.2f} [{q.loc[t, 'ci_lo']:.2f}, {q.loc[t, 'ci_hi']:.2f}]"
        return (f"| {p} | {len(m)} | {m.S.median():.3f} | {m.N_eff_u.median():.2f} | {m.N_ind.median():.2f} | "
                f"{m.ratio_ind.median():.2f} | {f('a_mean')} | {f('b_sd')} | {f('b_tail')} |")
    print(f"ERA5-Land vs B3_portfolio_cityyear.csv / B3_historical_portfolio_regressions.csv: city-years {dev_cy:.1e}, regressions {dev_reg:.1e}")
    for p in MAT:
        print(line(p))


FE = "meteorological_event_id"
VCOV = {"CRV1": f"city_name + {FE}"}
ID_COLUMNS = {"city_id": str, "host_id": str, "guest_id": str}


def fit(data, formula, vcov=None):
    import pyfixest as pf
    return pf.feols(formula, data=data, vcov=vcov or VCOV)


def coefficient(model, term):
    row = model.tidy().loc[term]
    G = int(np.atleast_1d(np.asarray(model._G, float)).min())
    return {"estimate": float(row["Estimate"]), "se": float(row["Std. Error"]), "ci_low": float(row["2.5%"]),
            "ci_high": float(row["97.5%"]), "p_value": float(row["Pr(>|t|)"]), "N": int(model._N), "G_min": G}


def halfrecord_validation():
    ev = pd.read_csv(need(INTERMEDIATE / "events.csv"), encoding=ENC, dtype=ID_COLUMNS, low_memory=False)
    ns = ev.loc[ev.network_sample == 1].copy()
    ns["S"], ns["Y"] = 1.0 - ns.HDC, 100.0 * ns.AffectedWeight
    est = ns.loc[ns.in_estimation_sample == 1].copy()
    assert len(ns) == 440 and len(est) == 364
    # partner demand hit by the same rainfall system on S computed from the full record and from each half
    half = est.dropna(subset=["HDC_early", "HDC_late"]).copy()
    hv = []
    for col, lab, d in (("HDC", "1981–2016", est), ("HDC", "1981–2016, events with both halves", half),
                        ("HDC_early", "1981–1998", half), ("HDC_late", "1999–2016", half)):
        d = d.assign(Sx=1.0 - d[col])
        r = coefficient(fit(d, f"Y ~ Sx + z_sev | {FE}"), "Sx")
        hv.append({"S_from_record": lab, "N": r["N"], "G_systems": r["G_min"], "b_per0p1": 0.1 * r["estimate"],
                   "ci_lo": 0.1 * r["ci_low"], "ci_hi": 0.1 * r["ci_high"], "p_2way": r["p_value"]})
    hv = pd.DataFrame(hv)
    hv.to_csv(INTERMEDIATE / "SI2_S_halfrecord_validation.csv", index=False, encoding=ENC)
    print(hv.round(3).to_string(index=False))


FE_MAIN = "city_name^ym + city_name^dow + datef + city_name^is_public_holiday + city_name^is_adjusted_workday"
CTRL = "C(rain_t_bin) + C(rain_l1_bin) + X_lag1 + X_lead1"
VC = {"CRV1": "city_name+datef"}
LAGS = ["A_F3", "A_F2", "A_F1", "A_L0", "A_L1", "A_L2", "A_L3"]
DAY = {"A_F3": -3, "A_F2": -2, "A_F1": -1, "A_L0": 0, "A_L1": 1, "A_L2": 2, "A_L3": 3}
FLOWS = {"inbound": "y_in", "local": "y_loc", "outbound": "y_out"}
CHECK_TOLERANCE = 1e-6


def dof(m):
    return int(np.atleast_1d(np.asarray(m._G, float)).min()) - 1


def coef(m, term):
    t = m.tidy().loc[term]
    return dict(est=float(t["Estimate"]), se=float(t["Std. Error"]), ci_lo=float(t["2.5%"]), ci_hi=float(t["97.5%"]),
                p=float(t["Pr(>|t|)"]), N=int(m._N), G_min=dof(m) + 1, dof=dof(m))


def lincomb(m, wmap):
    names = list(m._coefnames)
    w = np.zeros(len(names))
    for k, v in wmap.items():
        w[names.index(k)] += v
    e = float(w @ np.asarray(m._beta_hat).ravel())
    se = float(np.sqrt(w @ m._vcov @ w))
    df = dof(m)
    c = stats.t.ppf(0.975, df)
    return dict(est=e, se=se, ci_lo=e - c * se, ci_hi=e + c * se, p=float(2 * stats.t.sf(abs(e / se), df)),
                N=int(m._N), G_min=df + 1, dof=df)


def origin_dynamics():
    d = pd.read_parquet(INTERMEDIATE / "panel.parquet")
    # dry host-days outside the host's own event windows
    s = d[~d.in_event_window & (d.X_t == 0)]
    s = s[s["A_main"].notna() & s["Ageo_main"].notna()].copy()
    sd_ = s.dropna(subset=LAGS).copy()
    for c in LAGS:
        sd_[c + "_10"] = sd_[c] / 0.1
    rhs = " + ".join(c + "_10" for c in LAGS)
    rows = []
    for flow, y in FLOWS.items():
        m = pf.feols(f"{y} ~ {rhs} + {CTRL} | {FE_MAIN}", data=sd_, vcov=VC)
        for c in LAGS:
            rows.append(dict(flow=flow, term=c, day_after_partner_rain=DAY[c], **coef(m, c + "_10")))
        rows.append(dict(flow=flow, term="cumulative_t_to_t-3",
                         **lincomb(m, {c + "_10": 1 for c in ("A_L0", "A_L1", "A_L2", "A_L3")})))
        rows.append(dict(flow=flow, term="sum_leads_t+1_to_t+3", **lincomb(m, {c + "_10": 1 for c in ("A_F1", "A_F2", "A_F3")})))
        nm = list(m._coefnames)
        ix = [nm.index(c + "_10") for c in ("A_F1", "A_F2", "A_F3")]
        bL = np.asarray(m._beta_hat).ravel()[ix]
        Wd = float(bL @ np.linalg.solve(m._vcov[np.ix_(ix, ix)], bL))
        rows.append(dict(flow=flow, term="joint_leads_F_test", est=Wd / 3, p=float(stats.f.sf(Wd / 3, 3, dof(m))),
                         N=int(m._N), dof=dof(m)))
        print(f"  {flow}: N {m._N:,}; day 0 {coef(m, 'A_L0_10')['est']:+.3f}")
    out = pd.DataFrame(rows)

    # consistency with the inbound dynamics written by the transmission estimation
    ref = pd.read_csv(INTERMEDIATE / "results_dynamics.csv", encoding=ENC)
    mine = out.loc[out.flow == "inbound"].set_index("term")
    dev = 0.0
    for r in ref.itertuples():
        for c in ("est", "se", "p"):
            v = getattr(r, c)
            if pd.notna(v) and pd.notna(mine.loc[r.term, c]):
                dev = max(dev, abs(float(v) - float(mine.loc[r.term, c])) / max(1.0, abs(float(v))))
    print(f"inbound vs results_dynamics.csv: max relative deviation {dev:.1e}")
    assert dev < CHECK_TOLERANCE, f"inbound dynamics differ from results_dynamics.csv (max relative deviation {dev})"
    out.to_csv(INTERMEDIATE / "SI2_origin_dynamics.csv", index=False, encoding=ENC)


def main():
    t0 = time.time()
    print("== decay_by_product", flush=True)
    decay_by_product()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== origins_by_product", flush=True)
    origins_by_product()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== halfrecord_validation", flush=True)
    halfrecord_validation()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== origin_dynamics", flush=True)
    origin_dynamics()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
