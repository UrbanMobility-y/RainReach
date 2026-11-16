"""Whether the ex-ante shared-rain share predicts the realised partner shock and the inbound decline: historical portfolio regressions and random reassignment of partners, the transmission test on city-days without extreme rain and its variants, the summaries behind Figure 3 and the inbound-local gap by tercile of the shared-rain share."""
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
import pyfixest as pf
import scipy.stats as st

ROOT = r"E:\Extreme Rainfall and UnionPay Spending"
DATASET = os.path.join(ROOT, "Dataset")
EXTERNAL = os.path.join(ROOT, "External data")
INTERMEDIATE = os.path.join(ROOT, "Intermediate")


def portfolio_diversity():
    """Partner diversification versus shared rain: historical (1981-2016) distribution of the partner-weighted share hit
    on each host's extreme days and its relation to the effective number of partners (B3), realised 2018-2019 event shares
    regressed on S and on the effective number of partners (B3c), and a reassignment benchmark that separates the role of
    geography (distance bands) from partner choice in S (B4)."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
    FE = "meteorological_event_id"
    VC = {"CRV1": f"city_name + {FE}"}
    WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    warnings.filterwarnings("ignore")

    def z(s):
        return (s - s.mean()) / s.std()

    def _dm(A, s, n):
        S = np.zeros((n.size, A.shape[1]))
        np.add.at(S, s, A)
        return A - (S / n[:, None])[s]

    def _t(Xd, yd, s, G, k):
        XtX = np.linalg.inv(Xd.T @ Xd)
        b = XtX @ Xd.T @ yd
        u = yd - Xd @ b
        sc = np.zeros((G, Xd.shape[1]))
        np.add.at(sc, s, Xd * u[:, None])
        V = XtX @ (sc.T @ sc) @ XtX
        return b[k] / np.sqrt(V[k, k])

    def wcr(d, y, xs, term, B=9999, seed=1):
        """Wild cluster restricted bootstrap p-value (Webb weights by rainfall system, system fixed effects)."""
        sid = pd.factorize(d[FE])[0]
        cnt = np.bincount(sid)
        keep = cnt[sid] > 1
        s = np.unique(sid[keep], return_inverse=True)[1]
        n = np.bincount(s)
        G = n.size
        Xd = _dm(d.loc[keep, xs].to_numpy(float), s, n)
        yd = _dm(d.loc[keep, [y]].to_numpy(float), s, n)[:, 0]
        k = xs.index(term)
        t0 = _t(Xd, yd, s, G, k)
        Xr = np.delete(Xd, k, 1)
        br = np.linalg.lstsq(Xr, yd, rcond=None)[0]
        fr_ = Xr @ br
        ur = yd - fr_
        rg = np.random.default_rng(seed)
        V = WEBB[rg.integers(0, 6, (B, G))]
        XtX = np.linalg.inv(Xd.T @ Xd)
        P = XtX @ Xd.T
        ts = np.empty(B)
        for bb in range(B):
            ys = fr_ + V[bb][s] * ur
            b = P @ ys
            u = ys - Xd @ b
            sc = np.zeros((G, Xd.shape[1]))
            np.add.at(sc, s, Xd * u[:, None])
            Vc = XtX @ (sc.T @ sc) @ XtX
            ts[bb] = b[k] / np.sqrt(Vc[k, k])
        return float((np.abs(ts) >= abs(t0)).mean())

    def coef(d, y, xs, terms, scale, wild=True, B=9999):
        m = pf.feols(f"{y} ~ {' + '.join(xs)} | {FE}", data=d, vcov=VC)
        tb = m.tidy()
        out = []
        G = int(np.min(np.atleast_1d(m._G)))
        for tm in terms:
            r = tb.loc[tm]
            sc = scale[tm]
            out.append(dict(outcome=y, model="+".join(xs), term=tm, unit=("pp per +0.10" if sc == 10 else "pp per +1 s.d."),
                            est=r["Estimate"] * sc, se=r["Std. Error"] * sc, ci_low=r["2.5%"] * sc, ci_high=r["97.5%"] * sc, p_twoway=r["Pr(>|t|)"],
                            dof=G - 1, N=m._N, p_wcr=(wcr(d, y, xs, tm, B=B) if wild else np.nan)))
        return out

    ev = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, dtype={"city_id": str})
    ns = ev.loc[ev.network_sample == 1].reset_index(drop=True)

    # Historical extreme days (ERA5-Land rule, 1981-2016), co-exposure pi and the phi correlation of 2-day-per-year extremes
    wx = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC) for y in range(1981, 2017)], ignore_index=True)
    wx["city_name"] = wx.city_name.astype(str).str.strip()
    rain = wx.pivot(index="date", columns="city_name", values="rain_mm").sort_index()
    thr = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC)
    thr["city_name"] = thr.city_name.str.strip()
    cut = dict(zip(thr.city_name, thr.return_level_3yr_mm.fillna(np.inf)))
    cities = list(rain.columns)
    cutv = np.array([cut.get(c, np.inf) for c in cities])
    Rv = rain.to_numpy()
    H = ((Rv >= 50) | (Rv >= cutv[None, :])).astype(np.int8)
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    Hf = H.astype(np.float32)
    co = Hf.T @ Hf
    own = Hf.sum(0)
    PI = np.divide(co, own[:, None], out=np.full_like(co, np.nan), where=own[:, None] > 0)
    np.fill_diagonal(PI, np.nan)
    ix = {c: i for i, c in enumerate(cities)}
    freq = H.mean(0)
    thr2 = np.nanquantile(Rv, 1 - 2 / 365.25, axis=0)

    # CPC extremes at the ERA5-Land extreme-day frequency
    cpc = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_cpc_1981_2019.parquet"))
    cpc["city_name"] = cpc.city_name.str.strip()
    cw = cpc.pivot(index="date", columns="city_name", values="rain_mm").sort_index()
    cw.index = pd.to_datetime(cw.index)
    ch = cw.loc["1981-01-01":"2016-12-31"]
    cthr = {}
    for c in cities:
        if c in ch.columns and ch[c].notna().sum() > 1000:
            cthr[c] = float(np.nanquantile(ch[c].to_numpy(), 1 - freq[ix[c]]))
    x = pd.read_csv(os.path.join(INTERMEDIATE, "extreme_rainfall_city_days_2017_2019.csv"), encoding=ENC)
    x["city_name"] = x.city_name.str.strip()
    x = x.loc[x.is_extreme_rainfall == 1]
    x["date"] = pd.to_datetime(x.date)
    ext_sets = {k: set(v.date) for k, v in x.groupby("city_name")}
    cw2 = cw.loc["2017-01-01":"2019-12-31"]

    # Realised event shares of partner weight hit: catalogue (same-day) and CPC rules
    pn = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC, dtype={"host_id": str, "guest_id": str})
    pn["guest_name"] = pn.guest_name.astype(str).str.strip().fillna("nan")  # guests outside the city list are labelled "nan"
    pn["host_name"] = pn.host_name.astype(str).str.strip().fillna("nan")
    lk = {k: v for k, v in pn.loc[pn.usable == 1].groupby(["host_id", "network_year"])}
    cpcE = (cw2 >= pd.Series(cthr)).astype(float).where(cw2.notna())
    rows = []
    for r in ns.itertuples():
        b = lk.get((r.city_id, r.baseline_year))
        s, e = pd.Timestamp(r.city_event_start), pd.Timestamp(r.city_event_end)
        days = pd.date_range(s, e)
        w = b.w.to_numpy()
        g = b.guest_name.to_numpy()
        hsd = np.array([any(d in ext_sets.get(j, ()) for d in days) for j in g], float)
        hc = cpcE.loc[days, g].max(0).to_numpy()
        ok = ~np.isnan(hc)
        rows.append(dict(city_event_id=r.city_event_id, Ysd=(w * hsd).sum() / w.sum(), Ycpc=(w[ok] * hc[ok]).sum() / w[ok].sum()))
    ns = ns.merge(pd.DataFrame(rows), on="city_event_id")
    ns["S"] = 1 - ns.HDC

    cy = pd.read_csv(os.path.join(INTERMEDIATE, "city_year_network.csv"), encoding=ENC, dtype={"city_id": str})
    div = pn.groupby(["host_id", "network_year"]).apply(lambda d: pd.Series(dict(HHI=(d.w ** 2).sum(), npart=len(d)))).reset_index()
    cyx = cy.merge(div.rename(columns={"host_id": "city_id"}), on=["city_id", "network_year"])
    cyx["S"] = 1 - cyx.HDC
    ns["year"] = pd.to_datetime(ns.city_event_start).dt.year
    ns["N_eff"] = ns.EffectiveOrigins
    for y in ["Ysd", "Ycpc"]:
        ns["T" + y[1:]] = (ns[y] >= 0.5).astype(float)
    ns["zN_eff"] = z(ns.N_eff)

    # B3: distribution of the partner-weighted hit share on the host's historical extreme days
    E2 = (Rv >= thr2[None, :]).astype(np.float32)
    PHI = np.corrcoef(E2.T)
    rng = np.random.default_rng(11)
    cyu = cyx.loc[cyx.network_year.isin([2017, 2018])]
    pu = pn.loc[pn.usable == 1]
    lkn = {k: v for k, v in pu.groupby(["host_id", "network_year"])}
    recs = []
    for r in cyu.itertuples():
        b = lkn.get((r.city_id, r.network_year))
        if b is None or r.city_name not in ix:
            continue
        i = ix[r.city_name]
        jj = b.guest_name.map(ix).to_numpy().astype(int)
        w = b.w.to_numpy()
        w = w / w.sum()
        days = np.where(H[:, i] == 1)[0]
        if len(days) < 5:
            recs.append(dict(city_id=r.city_id, city_name=r.city_name, network_year=r.network_year, n_focal_days=len(days)))
            continue
        A = H[np.ix_(days, jj)] @ w
        pi = PI[i, jj]
        sd_ind = np.sqrt((w ** 2 * pi * (1 - pi)).sum())
        sims = (rng.random((20000, len(w))) < pi[None, :]).astype(float) @ w  # independent partners with the same pi
        Ph = PHI[np.ix_(jj, jj)]
        Ph = np.nan_to_num(Ph, nan=0.0)
        np.fill_diagonal(Ph, 1.0)
        recs.append(dict(city_id=r.city_id, city_name=r.city_name, network_year=r.network_year, n_focal_days=len(days), k=len(w),
                         meanA=A.mean(), S=r.S, sdA=A.std(ddof=0), tailA=(A >= .5).mean(), sd_ind=sd_ind, tail_ind=(sims >= .5).mean(),
                         N_eff_u=1 / (w ** 2).sum(), N_ind=1 / (w @ Ph @ w), N_eff=r.EffectiveOrigins, logd=np.log(r.MeanPartnerDistance), Near300=r.Near300))
    b3 = pd.DataFrame(recs)
    b3v = b3.dropna(subset=["meanA"]).copy()
    b3v["overdisp"] = b3v.sdA / b3v.sd_ind
    b3v["ratio_ind"] = b3v.N_ind / b3v.N_eff_u
    b3v["zlnN"] = z(np.log(b3v.N_eff_u))
    b3v["zlnNind"] = z(np.log(b3v.N_ind))
    b3v["zS"] = z(b3v.S)
    b3v["zS2"] = b3v.zS ** 2
    b3v["meanA_pp"] = 100 * b3v.meanA
    b3v["sdA_pp"] = 100 * b3v.sdA
    b3v["tailA_pp"] = 100 * b3v.tailA
    rows = []
    for nm, f in [("a_mean", "meanA_pp ~ zlnN + logd + Near300"), ("a_mean_noGeo", "meanA_pp ~ zlnN"),
                  ("b_sd", "sdA_pp ~ zlnN + zS + zS2"), ("b_tail", "tailA_pp ~ zlnN + zS + zS2"),
                  ("b_sd_geo", "sdA_pp ~ zlnN + zS + zS2 + logd + Near300"), ("b_tail_geo", "tailA_pp ~ zlnN + zS + zS2 + logd + Near300"),
                  ("desc_sd_Nind", "sdA_pp ~ zlnNind + zS + zS2"), ("desc_tail_Nind", "tailA_pp ~ zlnNind + zS + zS2"),
                  ("desc_overdisp", "overdisp ~ zlnN + zS + zS2")]:
        m = pf.feols(f, data=b3v, vcov={"CRV1": "city_name"})
        tb = m.tidy()
        for tm in tb.index:
            if tm == "Intercept":
                continue
            rows.append(dict(test=nm, formula=f, term=tm, est=tb.loc[tm, "Estimate"], se=tb.loc[tm, "Std. Error"], p=tb.loc[tm, "Pr(>|t|)"], N=m._N, R2=m._r2))
    pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "B3_historical_portfolio_regressions.csv"), index=False, encoding=ENC)
    b3v.to_csv(os.path.join(INTERMEDIATE, "B3_portfolio_cityyear.csv"), index=False, encoding=ENC)

    # B3c: realised event shares on S and the effective number of partners
    ns["zS"] = z(ns.S)
    ns["absErr_sd"] = (ns.Ysd - ns.S).abs()
    ns["absErr_cpc"] = (ns.Ycpc - ns.S).abs()
    SC3 = {"zS": 100, "zN_eff": 100}
    rows = []
    for samp in ["pooled", 2018, 2019]:
        d = (ns if samp == "pooled" else ns.loc[ns.year == samp]).reset_index(drop=True)
        for y in ["Ysd", "Tsd", "absErr_sd", "Ycpc", "Tcpc", "absErr_cpc"]:
            for r in coef(d, y, ["z_sev", "zS", "zN_eff"], ["zS", "zN_eff"], SC3, B=9999 if samp == "pooled" else 1999):
                r["sample"] = samp
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "B3c_realized_event_tests.csv"), index=False, encoding=ENC)

    # B4: expected S if partners were drawn at random nationally or within each partner's distance band
    U225 = sorted(set(pn.host_name) | set(pn.guest_name))
    DM = np.full((len(cities), len(cities)), np.nan)
    DM[hp.focal_city.map(ix), hp.partner_city.map(ix)] = hp.distance_km
    ld_all = np.log(hp.distance_km.clip(lower=5))
    edges = np.unique(np.quantile(ld_all, np.linspace(0, 1, 61)))

    def band(dkm):
        return np.clip(np.searchsorted(edges, np.log(np.clip(dkm, 5, None)), side="right") - 1, 0, len(edges) - 2)

    U = [c for c in U225 if c in ix]
    rng = np.random.default_rng(44)
    NB = 999
    recs = []
    for r in cyu.itertuples():
        b = lkn.get((r.city_id, r.network_year))
        i = ix[r.city_name]
        jj = b.guest_name.map(ix).to_numpy().astype(int)
        w = b.w.to_numpy()
        w = w / w.sum()
        k = len(w)
        cand = np.array([ix[c] for c in U if c != r.city_name and np.isfinite(PI[i, ix[c]])])
        pc = PI[i, cand]
        bc = band(DM[i, cand])
        bj = band(DM[i, jj])
        S = (w * PI[i, jj]).sum()
        draws_nat = np.array([(w * pc[rng.choice(len(cand), k, replace=False)]).sum() for _ in range(NB)])
        pools = []
        for bb in bj:
            sel = np.where(bc == bb)[0]
            if len(sel) == 0:  # empty band: use the nearest non-empty band
                avail = np.unique(bc)
                nb = avail[np.argmin(np.abs(avail - bb))]
                sel = np.where(bc == nb)[0]
            pools.append(sel)
        Eband = sum(w[t] * pc[pools[t]].mean() for t in range(k))
        draws_band = np.array([sum(w[t] * pc[rng.choice(pools[t])] for t in range(k)) for _ in range(NB)])
        recs.append(dict(city_id=r.city_id, city_name=r.city_name, network_year=r.network_year, S=S, E_nat=pc.mean(), E_nat_draw=draws_nat.mean(),
                         E_band=Eband, E_band_draw=draws_band.mean(), band_p025=np.percentile(draws_band, 2.5), band_p975=np.percentile(draws_band, 97.5),
                         pct_rank_band=(draws_band < S).mean(), pool_med=np.median([len(p) for p in pools])))
    b4 = pd.DataFrame(recs)
    b4["geo_share"] = (b4.E_band - b4.E_nat) / (b4.S - b4.E_nat)
    b4["choice_share"] = (b4.S - b4.E_band) / (b4.S - b4.E_nat)
    b4.to_csv(os.path.join(INTERMEDIATE, "B4_reassignment_cityyear.csv"), index=False, encoding=ENC)
    print("B3 city-years", len(b3v), "B4 city-years", len(b4))


def transmission_panel():
    """Build the host-day panel for the test for transmission. For each host city and day, A is the network-weighted share of
    partner cities with extreme rainfall that day and A_geo the same share for distance-matched non-partner cities; the
    panel also carries spending outcomes (100 ln spending), own rainfall controls, calendar fields and sample flags."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"

    # Spending (already inverted from the index scale)
    up = pd.read_csv(os.path.join(INTERMEDIATE, "unionpay_city_daily_2017_2019.csv"),
                     encoding=ENC, dtype={"city_id": str})
    up["city_name"] = up.city_name.astype(str).str.strip()
    up["date"] = pd.to_datetime(up.date)
    for c, n in [("inbound_spending", "y_in"), ("local_spending", "y_loc"), ("outbound_spending", "y_out")]:
        up[n] = 100 * np.log(up[c])

    # Extreme city-days (ERA5-Land catalogue)
    wx = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC) for y in (2016, 2017, 2018, 2019)],
                   ignore_index=True)
    wx["city_name"] = wx.city_name.astype(str).str.strip()
    wx["date"] = pd.to_datetime(wx.date)
    units = sorted(wx.city_name.unique())
    uix = {u: i for i, u in enumerate(units)}
    dates = pd.date_range("2016-12-25", "2020-01-05")
    dix = {d: i for i, d in enumerate(dates)}
    xd = pd.read_csv(os.path.join(INTERMEDIATE, "extreme_rainfall_city_days_2017_2019.csv"), encoding=ENC)
    xd["city_name"] = xd.city_name.astype(str).str.strip()
    xd["date"] = pd.to_datetime(xd.date)
    X = np.zeros((len(units), len(dates)), np.float64)
    X[xd.city_name.map(uix).values, xd.date.map(dix).values] = 1.0
    known = (dates >= "2017-01-01") & (dates <= "2019-12-31")

    # CPC extreme days: >= 50 mm or the CPC threshold with the same 1981-2016 frequency as the ERA5-Land rule
    thr = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC)
    thr["city_name"] = thr.city_name.astype(str).str.strip()
    cut = dict(zip(thr.city_name, thr.return_level_3yr_mm.fillna(np.inf)))
    hist = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC) for y in range(1981, 2017)],
                     ignore_index=True)
    hist["city_name"] = hist.city_name.astype(str).str.strip()
    hist["H"] = (hist.rain_mm >= 50) | (hist.rain_mm >= hist.city_name.map(cut))
    n_era = hist.groupby("city_name").H.sum()
    cpc = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_cpc_1981_2019.parquet"))
    cpc["city_name"] = cpc.city_name.astype(str).str.strip()
    cpc["date"] = pd.to_datetime(cpc.date)
    ch = cpc[cpc.date.dt.year <= 2016]
    u_ef = {}
    for c, g in ch.groupby("city_name"):
        n = int(n_era.get(c, 0))
        v = np.sort(g.rain_mm.dropna().values)[::-1]
        u_ef[c] = v[n - 1] if n > 0 and len(v) >= n else np.inf
    c3 = cpc[(cpc.date >= "2017-01-01") & (cpc.date <= "2019-12-31")].copy()
    c3["Xc"] = ((c3.rain_mm >= 50) | (c3.rain_mm >= c3.city_name.map(u_ef))).astype(float)
    Xc = np.zeros_like(X)
    cc = c3[c3.Xc == 1]
    Xc[cc.city_name.map(uix).values, cc.date.map(dix).values] = 1.0

    # Partner weights and distance-matched comparison units
    net = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC,
                      dtype={"city_id": str, "host_id": str, "guest_id": str}, low_memory=False)
    net["guest_name"] = net.guest_name.astype(str).str.strip()
    net["host_name"] = net.host_name.astype(str).str.strip()
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    byf = {f: (g.partner_city.values, g.distance_km.values) for f, g in hp.groupby("focal_city")}
    hosts = sorted(up.city_name.unique())

    def weights(host, year, min_km=None):
        """Weight vectors over units such that A = wA @ X[:, t] and A_geo = wG @ X[:, t]; None without a usable partner.
        Each partner j is matched to non-partner units k with |d_hk - d_hj| <= 0.10 d_hj, widened in 10-point steps."""
        blk = net[(net.host_name == host) & (net.network_year == year)]
        if host not in byf:
            return None
        allp = set(blk.guest_name)
        u = blk[blk.usable == 1]
        if min_km is not None:
            u = u[u.distance_km > min_km]
        if len(u) == 0:
            return None
        wn = u.w.values / u.w.values.sum()
        wA = np.zeros(len(units))
        wG = np.zeros(len(units))
        pc, dd = byf[host]
        ok = ~np.isin(pc, list(allp)) & (pc != host)
        for g, d, w_ in zip(u.guest_name.values, u.distance_km.values, wn):
            wA[uix[g]] += w_
            win = 0.10
            while True:
                sel = ok & (np.abs(dd - d) <= win * d + 1e-12)
                if sel.any():
                    break
                win += 0.10
            k = [uix[x] for x in pc[sel]]
            wG[k] += w_ / len(k)
        return wA, wG

    # Day t in year Y uses the network of year Y-1 (R9: fixed 2018 network for all years)
    rows = []
    for spec, yr_map, mk, XX in [("main", {2018: 2017, 2019: 2018}, None, X), ("R6", {2018: 2017, 2019: 2018}, 150.0, X),
                                 ("R7", {2018: 2017, 2019: 2018}, None, Xc), ("R9", {2017: 2018, 2018: 2018, 2019: 2018}, None, X)]:
        for h in hosts:
            for Y, ny in yr_map.items():
                ww = weights(h, ny, mk)
                dts = pd.date_range(f"{Y}-01-01", f"{Y}-12-31")
                ii = np.array([dix[d] for d in dts])
                if ww is None:
                    continue
                wA, wG = ww
                rec = {"city_name": h, "date": dts, f"A_{spec}": wA @ XX[:, ii], f"Ageo_{spec}": wG @ XX[:, ii]}
                if spec == "main":
                    for k in range(-3, 4):  # A^(k): partner extremes on day t-k, network of day t
                        v = wA @ XX[:, ii - k]
                        v[~known[ii - k]] = np.nan
                        rec[f"A_L{k}" if k >= 0 else f"A_F{-k}"] = v
                rows.append(pd.DataFrame(rec).assign(spec=spec))
    P = None
    for spec in ["main", "R6", "R7", "R9"]:
        blk = pd.concat([r for r in rows if r.spec.iat[0] == spec]).drop(columns="spec")
        P = blk if P is None else P.merge(blk, on=["city_name", "date"], how="outer")

    # Host-day frame
    d = up[["city_id", "city_name", "date", "y_in", "y_loc", "y_out"]].merge(P, on=["city_name", "date"], how="left")
    rain = wx.set_index(["city_name", "date"]).rain_mm
    d["rain_t"] = rain.reindex(pd.MultiIndex.from_frame(d[["city_name", "date"]])).values
    d["rain_l1"] = rain.reindex(pd.MultiIndex.from_arrays([d.city_name, d.date - pd.Timedelta(days=1)])).values
    bins = [-np.inf, 0, 5, 10, 25, 50, np.inf]
    lab = ["b0", "b0_5", "b5_10", "b10_25", "b25_50", "b50p"]
    for c in ("rain_t", "rain_l1"):
        d[c + "_bin"] = pd.cut(d[c], bins=bins, labels=lab, right=True).astype(object).fillna("nan").astype(str)  # missing rain is level "nan"
    hu = np.array([uix.get(c, -1) for c in d.city_name])
    di = np.array([dix[x] for x in d.date])
    has = hu >= 0

    def own_extreme(M, off):
        v = np.full(len(d), np.nan)
        v[has] = M[hu[has], di[has] + off]
        return v

    d["X_t"] = own_extreme(X, 0)
    d["X_lag1"] = own_extreme(X, -1)
    d["X_lead1"] = own_extreme(X, 1)
    d["Xc_t"] = own_extreme(Xc, 0)
    d["Xc_lag1"] = own_extreme(Xc, -1)
    d["Xc_lead1"] = own_extreme(Xc, 1)

    # Own event windows: city event start - 3 to end + 7 days
    cat = pd.read_csv(os.path.join(INTERMEDIATE, "extreme_rainfall_events_2017_2019.csv"), encoding=ENC)
    cat["city_name"] = cat.city_name.astype(str).str.strip()
    win = set()
    for r in cat.itertuples(index=False):
        for x in pd.date_range(pd.Timestamp(r.city_event_start) - pd.Timedelta(days=3), pd.Timestamp(r.city_event_end) + pd.Timedelta(days=7)):
            win.add((r.city_name, x))
    d["in_event_window"] = [((c, x) in win) for c, x in zip(d.city_name, d.date)]

    cal = pd.read_csv(os.path.join(DATASET, "China_calendar_2017_2019", "China_calendar_2017_2019.csv"), encoding=ENC)
    cal["date"] = pd.to_datetime(cal.date)
    d = d.merge(cal[["date", "weekday", "is_public_holiday", "is_adjusted_workday", "holiday_name"]], on="date", how="left")
    sf = cal[cal.holiday_name == "Spring Festival"].groupby(cal.date.dt.year).date.agg(["min", "max"])
    d["sf_pm7"] = False
    for _, r in sf.iterrows():
        d.loc[d.date.between(r["min"] - pd.Timedelta(days=7), r["max"] + pd.Timedelta(days=7)), "sf_pm7"] = True

    # Province (data dictionary, city list sheet), macro-region and Pearl River Delta flags
    xlsx = [f for f in os.listdir(DATASET) if f.endswith(".xlsx")][0]
    dic = pd.read_excel(os.path.join(DATASET, xlsx), sheet_name=5).dropna()
    dic.columns = ["province", "city_name", "city_id"]
    prov = dict(zip(dic.city_name.astype(str).str.strip(), dic.province.astype(str).str.strip()))
    d["province"] = d.city_name.map(prov)
    regions = pd.read_csv(os.path.join(EXTERNAL, "province_regions.csv"), encoding=ENC)
    d["region4"] = d.province.map(dict(zip(regions.province, regions.region4))).fillna("West")
    prd = pd.read_csv(os.path.join(EXTERNAL, "pearl_river_delta_cities.csv"), encoding=ENC)
    d["PRD"] = d.city_name.isin(set(prd.city_name))

    # Remote-sector-heavy hosts: mean 2017-2019 share of inbound spending in remote sectors above 0.5
    REMOTE_SECTORS = {"D", "G56", "I63", "I64", "I65", "J66", "J67", "J68", "J69", "K70", "Y"}
    id2name = dict(zip(up.city_id.str.strip(), up.city_name))
    an = pd.read_parquet(os.path.join(INTERMEDIATE, "sector_annual_sums.parquet"))
    an = an[an.year <= 2019].copy()
    an["city_name"] = an.mchnt_city.str.strip().map(id2name)
    an["remote"] = an.sic.isin(REMOTE_SECTORS)
    cy = an.groupby(["city_name", "year"]).apply(lambda g: g.loc[g.remote, "I_sum"].sum() / g.I_sum.sum())
    remote_share = cy.groupby(level=0).mean()
    d["remote25"] = d.city_name.isin(set(remote_share[remote_share > 0.50].index))

    d["year"] = d.date.dt.year
    d["ym"] = d.date.dt.strftime("%Y-%m")
    d["dow"] = d.date.dt.day_of_week
    d["datef"] = d.date.dt.strftime("%Y-%m-%d")
    d.to_parquet(os.path.join(INTERMEDIATE, "panel.parquet"), index=False)
    print("panel rows", len(d), "hosts with A", d.dropna(subset=["A_main"]).city_name.nunique(), "remote hosts", int((remote_share > 0.5).sum()))


def transmission_tests(argv=None):
    """Test for transmission: regress host-city spending on the network-weighted share of partner cities hit by extreme
    rainfall (A, per +10 pp), with the distance-matched geographic shock (A_geo) as a control, a stacked inbound-local
    contrast, day-0 event-time benchmarks, a host bootstrap of the implied event-scale effect, distributed lags and
    robustness specifications R1-R10."""
    ENC = "utf-8-sig"
    warnings.filterwarnings("ignore")

    FE_MAIN = "city_name^ym + city_name^dow + datef + city_name^is_public_holiday + city_name^is_adjusted_workday"
    FE_R2 = FE_MAIN.replace("datef", "province^datef")
    FE_R3 = FE_MAIN.replace("datef", "region4^datef")
    CTRL = "C(rain_t_bin) + C(rain_l1_bin) + X_lag1 + X_lead1"
    CTRL_R7 = "C(rain_t_bin) + C(rain_l1_bin) + Xc_lag1 + Xc_lead1"
    VC = {"CRV1": "city_name+datef"}
    Z_MDE = st.norm.ppf(0.975) + st.norm.ppf(0.80)  # minimum detectable effect: 80% power, two-sided 5% test
    SPECS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
    LABEL = {"R1": "Strict dry host (rain < 1 mm on t and t-1)", "R2": "Province x date FE instead of date FE",
             "R3": "Four macro-region (NBS East/Central/West/Northeast) x date FE", "R4": "Exclude 9 PRD hosts",
             "R5": "Exclude 25 remote-sector-heavy hosts (remote_inbound_share > 0.50)", "R6": "Partners > 150 km only (A and A_geo)",
             "R7": "X from CPC (>= 50 mm or CPC equal-frequency threshold)", "R8": "Exclude public holidays and Spring Festival +-7 d",
             "R9": "Fixed 2018 network, 2017-2019 host-days", "R10": "Winsorise y at P1/P99 within host"}
    N_WORKERS = 12

    def load():
        return pd.read_parquet(os.path.join(INTERMEDIATE, "panel.parquet"))

    def spec_frame(d, spec):
        """Sample, fixed effects and controls for a specification (host-days outside own event windows, no own extreme)."""
        A, G, fe, ctrl = "A_main", "Ageo_main", FE_MAIN, CTRL
        s = d[~d.in_event_window & (d.X_t == 0)]
        if spec == "R6":
            A, G = "A_R6", "Ageo_R6"
        if spec == "R7":
            A, G, ctrl = "A_R7", "Ageo_R7", CTRL_R7
            s = s[s.Xc_t == 0]
        if spec == "R9":
            A, G = "A_R9", "Ageo_R9"
        s = s[s[A].notna() & s[G].notna()].copy()
        if spec == "R1":
            s = s[(s.rain_t < 1) & (s.rain_l1 < 1)]
        if spec == "R2":
            fe = FE_R2
        if spec == "R3":
            fe = FE_R3
        if spec == "R4":
            s = s[~s.PRD]
        if spec == "R5":
            s = s[~s.remote25]
        if spec == "R8":
            s = s[(s.is_public_holiday == 0) & (~s.sf_pm7)]
        s = s.copy()
        s["A10"] = s[A] / 0.1
        s["G10"] = s[G] / 0.1
        if spec == "R10":
            for y in ("y_in", "y_loc", "y_out"):
                lo = s.groupby("city_name")[y].transform(lambda v: v.quantile(0.01))
                hi = s.groupby("city_name")[y].transform(lambda v: v.quantile(0.99))
                s[y] = s[y].clip(lo, hi)
        return s, fe, ctrl

    def dof(m):
        return int(np.atleast_1d(np.asarray(m._G, float)).min()) - 1

    def coef(m, term):
        t = m.tidy().loc[term]
        se = float(t["Std. Error"])
        df = dof(m)
        return dict(est=float(t["Estimate"]), se=se, ci_lo=float(t["2.5%"]), ci_hi=float(t["97.5%"]), p=float(t["Pr(>|t|)"]),
                    N=int(m._N), G_min=df + 1, dof=df, MDE=float(Z_MDE * se))

    def lincomb(m, wmap):
        names = list(m._coefnames)
        w = np.zeros(len(names))
        for k, v in wmap.items():
            w[names.index(k)] += v
        e = float(w @ np.asarray(m._beta_hat).ravel())
        se = float(np.sqrt(w @ m._vcov @ w))
        df = dof(m)
        c = st.t.ppf(0.975, df)
        tt = e / se
        return dict(est=e, se=se, ci_lo=e - c * se, ci_hi=e + c * se, p=float(2 * st.t.sf(abs(tt), df)),
                    p_one_sided_lt0=float(st.t.cdf(tt, df)), N=int(m._N), G_min=df + 1, dof=df, MDE=float(Z_MDE * se))

    def fit(s, y, rhs, fe, ctrl):
        import pyfixest as pf
        return pf.feols(f"{y} ~ {rhs} + {ctrl} | {fe}", data=s, vcov=VC)

    def run_task(spec, model):
        d = load()
        s, fe, ctrl = spec_frame(d, spec)
        if model == "H3":
            m = fit(s, "y_in", "A10 + G10", fe, ctrl)
            out = [dict(spec=spec, quantity=lab, **coef(m, term)) for term, lab in (("A10", "H3_beta"), ("G10", "H3_gamma"))]
            out.append(dict(spec=spec, quantity="H3_beta_minus_gamma", **lincomb(m, {"A10": 1, "G10": -1})))
            return out
        y = {"in": "y_in", "loc": "y_loc", "out": "y_out"}[model]
        m = fit(s, y, "A10", fe, ctrl)
        return [dict(spec=spec, quantity=f"beta_{model}", **coef(m, "A10"), n_hosts=int(s.city_name.nunique()))]

    # Host bootstrap: point estimates by alternating projections on the five fixed-effect dimensions
    def demean(M, fes, tol=1e-10, maxit=10000):
        M = M.copy()
        cnt = [np.bincount(f) for f in fes]
        for _ in range(maxit):
            mx = 0.0
            for f, c in zip(fes, cnt):
                mu = np.vstack([np.bincount(f, weights=M[:, j], minlength=len(c)) for j in range(M.shape[1])]).T / c[:, None]
                M -= mu[f]
                mx = max(mx, float(np.abs(mu).max()))
            if mx < tol:
                break
        return M

    def boot_prepare():
        d = load()
        s, fe, ctrl = spec_frame(d, "main")
        s = s.reset_index(drop=True)
        Xb = pd.get_dummies(s[["rain_t_bin", "rain_l1_bin"]], drop_first=True).astype(float).values
        M = np.column_stack([s.y_in.values, s.A10.values, Xb])
        host = pd.factorize(s.city_name)[0]
        keys = dict(ym=pd.factorize(s.ym)[0], dow=s.dow.values, date=pd.factorize(s.datef)[0],
                    hol=s.is_public_holiday.values.astype(int), adj=s.is_adjusted_workday.values.astype(int))
        names = pd.factorize(s.city_name)[1]
        return M, host, keys, np.asarray(names)

    def beta_point(M, host, keys, rows=None, copy_id=None):
        if rows is not None:
            M = M[rows]
            hid = copy_id
            k = {a: b[rows] for a, b in keys.items()}
        else:
            hid = host
            k = keys
        fes = [pd.factorize(hid * 100000 + k["ym"])[0], pd.factorize(hid * 10 + k["dow"])[0], pd.factorize(k["date"])[0],
               pd.factorize(hid * 2 + k["hol"])[0], pd.factorize(hid * 2 + k["adj"])[0]]
        Md = demean(M, fes)
        return float(np.linalg.lstsq(Md[:, 1:], Md[:, 0], rcond=None)[0][0])

    def boot_chunk(draws):
        M, host, keys, names = boot_prepare()
        idx = [np.where(host == h)[0] for h in range(host.max() + 1)]
        out = []
        for hs in draws:
            rows = np.concatenate([idx[h] for h in hs])
            cid = np.concatenate([np.full(len(idx[h]), c) for c, h in enumerate(hs)])
            out.append(beta_point(M, host, keys, rows, cid))
        return out

    def launch(argsets, workdir, max_par=N_WORKERS):
        """Run this script as worker processes with the given argument lists."""
        me = os.path.abspath(__file__)
        running, queue, failed = [], list(argsets), []
        while queue or running:
            while queue and len(running) < max_par:
                a = queue.pop(0)
                log = open(os.path.join(workdir, "worker_" + "_".join(a) + ".log"), "w")
                running.append((subprocess.Popen([sys.executable, me, "transmission_tests", *a, workdir], stdout=log, stderr=subprocess.STDOUT), log, a))
            for t in running[:]:
                if t[0].poll() is not None:
                    t[1].close()
                    if t[0].returncode != 0:
                        failed.append(t[2])
                    running.remove(t)
            time.sleep(0.5)
        if failed:
            raise RuntimeError(f"workers failed: {failed}")

    def main():
        import pyfixest as pf
        work = tempfile.mkdtemp(prefix="transmission_")
        rows = []
        d = load()
        s, fe, ctrl = spec_frame(d, "main")

        # Inbound, local and outbound responses
        mods = {}
        for y, lab in (("y_in", "beta_in"), ("y_loc", "beta_loc"), ("y_out", "beta_out")):
            mods[lab] = fit(s, y, "A10", fe, ctrl)
            rows.append(dict(quantity=lab, **coef(mods[lab], "A10")))

        # Stacked inbound and local margins, all fixed effects and controls interacted with the margin
        L = []
        for mg, y in (("in", "y_in"), ("loc", "y_loc")):
            t = s[["city_name", "datef", "ym", "dow", "is_public_holiday", "is_adjusted_workday", "rain_t_bin", "rain_l1_bin",
                   "X_lag1", "X_lead1", "A10", y]].rename(columns={y: "y"}).copy()
            t["margin"] = mg
            L.append(t)
        L = pd.concat(L, ignore_index=True)
        for mg in ("in", "loc"):
            ind = (L.margin == mg).astype(float)
            L[f"A10_{mg}"] = L.A10 * ind
            L[f"Xl1_{mg}"] = L.X_lag1 * ind
            L[f"Xf1_{mg}"] = L.X_lead1 * ind
        L["rtb_m"] = L.rain_t_bin + "_" + L.margin
        L["rl1_m"] = L.rain_l1_bin + "_" + L.margin
        L["host_m"] = L.city_name + "|" + L.margin
        L["date_m"] = L.datef + "|" + L.margin
        feS = "host_m^ym + host_m^dow + date_m + host_m^is_public_holiday + host_m^is_adjusted_workday"
        mS = pf.feols(f"y ~ A10_in + A10_loc + C(rtb_m) + C(rl1_m) + Xl1_in + Xl1_loc + Xf1_in + Xf1_loc | {feS}", data=L, vcov=VC)
        rows.append(dict(quantity="H2_stacked_beta_in", **coef(mS, "A10_in")))
        rows.append(dict(quantity="H2_stacked_beta_loc", **coef(mS, "A10_loc")))
        rows.append(dict(quantity="H2_beta_in_minus_beta_loc", **lincomb(mS, {"A10_in": 1, "A10_loc": -1})))
        del L

        # Joint model with the geography-predicted shock
        m3 = fit(s, "y_in", "A10 + G10", fe, ctrl)
        rows.append(dict(quantity="H3_beta", **coef(m3, "A10")))
        rows.append(dict(quantity="H3_gamma", **coef(m3, "G10")))
        rows.append(dict(quantity="H3_beta_minus_gamma", **lincomb(m3, {"A10": 1, "G10": -1})))
        mg_ = fit(s, "y_in", "G10", fe, ctrl)
        rows.append(dict(quantity="Ageo_only_inbound_info", **coef(mg_, "G10")))

        # Day-0 event-time benchmarks: day-0 deviation minus the clean pre-event mean (k = -14..-1, >= 7 clean days)
        cfp = pd.read_csv(os.path.join(INTERMEDIATE, "counterfactual_panel.csv"), encoding=ENC, dtype={"city_id": str})
        fr = cfp.loc[cfp.cf_complete_end14 == 1].copy()
        fr["city_name"] = fr.city_name.astype(str).str.strip()
        fr["k"] = fr.event_time_from_start.astype(int)
        MG = {"local": ("local_spending", "normal_local_spending"), "inbound": ("inbound_spending", "normal_inbound_spending"),
              "outbound": ("outbound_spending", "normal_outbound_spending")}
        for lab, (o, n) in MG.items():
            fr[f"A_{lab}"] = 100.0 * (fr[o] / fr[n] - 1.0)
        pre = fr.loc[fr.k.between(-14, -1) & (fr.is_other_city_event_exposure == 0)]
        nd = pre.groupby("city_event_id").size()
        keep = set(nd[nd >= 7].index)
        basem = pre.loc[pre.city_event_id.isin(keep)].groupby("city_event_id")[[f"A_{m}" for m in MG]].mean()
        c0 = fr.loc[fr.city_event_id.isin(keep) & (fr.k == 0)].copy()
        day0 = []
        for lab in MG:
            c0[f"C_{lab}"] = c0[f"A_{lab}"] - c0.city_event_id.map(basem[f"A_{lab}"])
            mm = pf.feols(f"C_{lab} ~ 1", data=c0, vcov={"CRV1": "city_name + meteorological_event_id"})
            day0.append(dict(margin=lab, **coef(mm, "Intercept"),
                             source="counterfactual_panel.csv; day-0 event-time deviation minus clean pre-event mean, k=0, CRV1 city + system"))
        day0 = pd.DataFrame(day0)
        day0.to_csv(os.path.join(INTERMEDIATE, "results_day0_benchmark.csv"), index=False, encoding=ENC)
        d0 = day0.set_index("margin")
        bi = [r for r in rows if r["quantity"] == "beta_in"][0]

        # Implied event-scale share D = beta_in x (mean AffectedWeight / 0.1) / day-0 inbound effect; 999-draw host bootstrap
        ev = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, dtype={"city_id": str}, low_memory=False)
        ev["city_name"] = ev.city_name.astype(str).str.strip()
        aw = ev.loc[ev.AffectedWeight.notna(), ["city_name", "AffectedWeight"]]
        M, host, keys, names = boot_prepare()
        assert abs(beta_point(M, host, keys) - bi["est"]) < 1e-8
        rng = np.random.default_rng(20261109)
        H = host.max() + 1
        NB = 999
        draws = [rng.integers(0, H, H) for _ in range(NB)]
        np.save(os.path.join(work, "boot_draws.npy"), np.vstack(draws))
        launch([("boot", str(i), "16") for i in range(16)], work)
        betas = np.empty(NB)
        for i in range(16):
            betas[i::16] = np.load(os.path.join(work, f"boot_chunk_{i}.npy"))
        awg = aw.groupby("city_name").AffectedWeight.agg(["sum", "count"])
        c0g = c0.groupby("city_name").C_inbound.agg(["sum", "count"])
        awS = awg.reindex(names).fillna(0)
        c0S = c0g.reindex(names).fillna(0)
        B_ = []
        for b, hs in enumerate(draws):
            mA = awS["sum"].values[hs].sum() / awS["count"].values[hs].sum()
            m0 = c0S["sum"].values[hs].sum() / c0S["count"].values[hs].sum()
            B_.append(dict(draw=b, beta_in=betas[b], mean_AffectedWeight=mA, day0_inbound=m0, D=betas[b] * (mA / 0.1) / m0))
        pd.DataFrame(B_).to_csv(os.path.join(INTERMEDIATE, "results_D_bootstrap_draws.csv"), index=False, encoding=ENC)

        # Distributed lags t-3..t+3
        lagc = ["A_F3", "A_F2", "A_F1", "A_L0", "A_L1", "A_L2", "A_L3"]
        sd_ = s.dropna(subset=lagc).copy()
        for c in lagc:
            sd_[c + "_10"] = sd_[c] / 0.1
        md = fit(sd_, "y_in", " + ".join(c + "_10" for c in lagc), fe, ctrl)
        dyn = [dict(term=c, **coef(md, c + "_10")) for c in lagc]
        dyn.append(dict(term="cumulative_t_to_t-3", **lincomb(md, {c + "_10": 1 for c in ["A_L0", "A_L1", "A_L2", "A_L3"]})))
        dyn.append(dict(term="sum_leads_t+1_to_t+3", **lincomb(md, {c + "_10": 1 for c in ["A_F1", "A_F2", "A_F3"]})))
        nm = list(md._coefnames)
        ix = [nm.index(c + "_10") for c in ["A_F1", "A_F2", "A_F3"]]
        bL = np.asarray(md._beta_hat).ravel()[ix]
        VL = md._vcov[np.ix_(ix, ix)]
        Wd = float(bL @ np.linalg.solve(VL, bL))
        dfd = dof(md)
        dyn.append(dict(term="joint_leads_F_test", est=Wd / 3, p=float(st.f.sf(Wd / 3, 3, dfd)), N=int(md._N), dof=dfd))
        pd.DataFrame(dyn).to_csv(os.path.join(INTERMEDIATE, "results_dynamics.csv"), index=False, encoding=ENC)
        pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "results_main.csv"), index=False, encoding=ENC)
        print(pd.DataFrame(rows)[["quantity", "est", "se", "ci_lo", "ci_hi", "p", "N"]].to_string())

        # Robustness specifications
        tasks = [(sp, m) for sp in SPECS for m in ("in", "loc", "out", "H3")]
        launch([("task", sp, m) for sp, m in tasks], work)
        res = [json.load(open(os.path.join(work, f"task_{sp}_{m}.json"))) for sp, m in tasks]
        df = pd.DataFrame([r for rr in res for r in rr])
        df["description"] = df.spec.map(LABEL)
        df.to_csv(os.path.join(INTERMEDIATE, "results_robustness.csv"), index=False, encoding=ENC)
        print(df[["spec", "quantity", "est", "se", "p", "N"]].to_string())

    argv = argv or [sys.argv[0]]
    if len(argv) > 1 and argv[1] == "boot":
        i, n, work = int(argv[2]), int(argv[3]), argv[4]
        draws = np.load(os.path.join(work, "boot_draws.npy"))
        np.save(os.path.join(work, f"boot_chunk_{i}.npy"), np.array(boot_chunk([draws[b] for b in range(i, len(draws), n)])))
    elif len(argv) > 1 and argv[1] == "task":
        spec, model, work = argv[2], argv[3], argv[4]
        json.dump(run_task(spec, model), open(os.path.join(work, f"task_{spec}_{model}.json"), "w"))
    else:
        main()


def figure3_summaries():
    """Figure 3 inputs: quintile means of partner demand hit by the same rainfall system across ex-ante shared-rain exposure S
    (two-way clustered CIs, with CPC same-day quintile means), and the transmission decomposition D linking the
    partner-shock coefficient to the day-0 inbound decline."""
    ENC = "utf-8-sig"
    FE = "meteorological_event_id"
    VCOV = {"CRV1": f"city_name + {FE}"}

    def fit(data, formula, vcov=None):
        return pf.feols(formula, data=data, vcov=vcov or VCOV)

    def dof_of(model):
        groups = np.atleast_1d(np.asarray(getattr(model, "_G", []) or [], dtype=float))
        return int(groups.min()) - 1 if groups.size else int(model._N - len(model._coefnames))

    def coefficient(model, term):
        row = model.tidy().loc[term]
        return {"estimate": float(row["Estimate"]), "se": float(row["Std. Error"]), "ci_low": float(row["2.5%"]),
                "ci_high": float(row["97.5%"]), "p_value": float(row["Pr(>|t|)"]), "dof": dof_of(model), "N": int(model._N)}

    ev = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC,
                     dtype={"city_id": str, "host_id": str, "guest_id": str}, low_memory=False)
    ns = ev.loc[ev.network_sample == 1].copy()
    ns["S"] = 1.0 - ns.HDC
    ns["Y"] = 100.0 * ns.AffectedWeight

    # quintile means of partner demand hit
    nq = ns.assign(q=pd.qcut(ns.S, 5, labels=False) + 1)
    rows = []
    for k, g in nq.groupby("q"):
        r = coefficient(fit(g, "Y ~ 1"), "Intercept")
        rows.append({"quintile": int(k), "n_events": len(g), "n_cities": g.city_name.nunique(),
                     "n_events_multi": g[FE].nunique(), "S_min": g.S.min(), "S_max": g.S.max(),
                     "mean_pct": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "dof": r["dof"]})
    qa = pd.DataFrame(rows).set_index("quintile")
    d = nq.loc[nq.q.isin([1, 5])].assign(top=lambda x: (x.q == 5).astype(int))
    diff = coefficient(fit(d, "Y ~ top"), "top")
    dec = pd.read_csv(os.path.join(INTERMEDIATE, "H6_reliability_deciles.csv"))
    c = dec.loc[(dec["product"] == "CPC") & (dec.outcome == "Y_CPCsd_EF")].copy()
    c["quin"] = (c.dec + 1) // 2
    cpc = c.groupby("quin").apply(lambda x: 100 * np.average(x.mean_realised, weights=x.n))
    qa["cpc_mean_pct"] = cpc.values
    out_a = os.path.join(INTERMEDIATE, "fig3a_quintile_means_ci.csv")
    qa.to_csv(out_a, encoding=ENC)
    pd.DataFrame([{"contrast": "Q5 - Q1", **diff}]).to_csv(out_a, mode="a", index=False, encoding="utf-8")

    # transmission decomposition D
    rm = pd.read_csv(os.path.join(INTERMEDIATE, "results_main.csv")).set_index("quantity")
    bm = pd.read_csv(os.path.join(INTERMEDIATE, "results_day0_benchmark.csv")).set_index("margin")
    dr = pd.read_csv(os.path.join(INTERMEDIATE, "results_D_bootstrap_draws.csv"))
    mean_aw = float(ns.AffectedWeight.mean())
    b_in, day0 = float(rm.loc["beta_in", "est"]), float(bm.loc["inbound", "est"])
    implied = b_in * mean_aw / 0.1
    D = implied / day0
    d_lo, d_hi = np.percentile(dr.D, [2.5, 97.5])
    imp = dr.beta_in * dr.mean_AffectedWeight / 0.1
    i_lo, i_hi = np.percentile(imp, [2.5, 97.5])
    out = {"beta_in_per10pp": b_in, "mean_AffectedWeight": mean_aw, "n_events_AffectedWeight": int(ns.AffectedWeight.notna().sum()),
           "day0_inbound_pp": day0, "implied_partner_pp": implied, "implied_lo": i_lo, "implied_hi": i_hi,
           "D": D, "D_lo": d_lo, "D_hi": d_hi, "n_draws": len(dr), "boot_sd_beta_in": float(dr.beta_in.std(ddof=1)),
           "twoway_se_beta_in": float(rm.loc["beta_in", "se"])}
    pd.DataFrame([out]).to_csv(os.path.join(INTERMEDIATE, "fig3d_origin_shock.csv"), index=False, encoding=ENC)
    print(qa.to_string(float_format="%.3f"))
    print(f"D = {D:.4f} [{d_lo:.4f}, {d_hi:.4f}]")


def gap_by_exposure_tercile():
    """Inbound minus local spending change by tercile of the ex-ante shared-rain share S (440 network-sample events of
    2018-2019), with 95% CIs clustered by rainfall system."""
    warnings.filterwarnings("ignore")

    e = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding="utf-8-sig")
    d = e.dropna(subset=["HDC"]).copy()
    d["S"] = 1 - d.HDC
    d["St"] = pd.qcut(d.S, 3, labels=["low", "mid", "high"])
    rows = []
    for k, g in d.groupby("St", observed=True):
        r = {"S_tercile": k, "n_events": len(g), "n_systems": g.meteorological_event_id.nunique(), "mean_S": g.S.mean(), "mean_peak_rain_mm": g.peak_rain_mm.mean(),
             "mean_partner_demand_hit": g.AffectedWeight.mean(), "ratio_inbound_to_local": g.CurrentShock_inbound.mean() / g.CurrentShock_local.mean()}
        for y, lab in [("CurrentShock_local", "local_pp"), ("CurrentShock_inbound", "inbound_pp"), ("G_in", "inbound_minus_local_pp")]:
            m = pf.feols(f"{y} ~ 1", data=g, vcov={"CRV1": "meteorological_event_id"}).tidy().iloc[0]
            r[lab] = m["Estimate"]
            r[lab + "_lo"] = m["2.5%"]
            r[lab + "_hi"] = m["97.5%"]
        rows.append(r)
    t = pd.DataFrame(rows).round(3)
    note = ("# Event-level spending change by tercile of ex-ante shared-rain share S (events.csv, 440 events with S, 2018-2019); "
            "CIs clustered by rainfall system\n")
    p = os.path.join(INTERMEDIATE, "check_gap_by_S_tercile.csv")
    open(p, "w", encoding="utf-8").write(note)
    t.to_csv(p, mode="a", index=False)
    print(t.T.to_string())


def main():
    portfolio_diversity()
    transmission_panel()
    transmission_tests()
    figure3_summaries()
    gap_by_exposure_tercile()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("transmission_tests",):
        globals()[sys.argv[1]]([sys.argv[0]] + sys.argv[2:])
    else:
        main()
