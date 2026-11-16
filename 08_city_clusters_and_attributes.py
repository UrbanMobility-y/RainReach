"""City clusters of the national urbanization plan and their validation, and differences in spending declines and exposure by city attributes (income, size, location, cluster membership), with the pre-specified incidence tests."""
import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = r"E:\Extreme Rainfall and UnionPay Spending"
DATASET = os.path.join(ROOT, "Dataset")
EXTERNAL = os.path.join(ROOT, "External data")
INTERMEDIATE = os.path.join(ROOT, "Intermediate")


def city_clusters():
    """City-cluster membership (19 national-plan city clusters) joined to the city identifiers, the 369 grid units and the
    225 UnionPay hosts (city_clusters_19.csv), and the within- versus outside-cluster shared-rain index of each host's
    inbound partners by cluster and year with bootstrap intervals (cluster19_inside_outside.csv)."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
    warnings.filterwarnings("ignore")

    members = pd.read_csv(os.path.join(EXTERNAL, "city_cluster_membership.csv"), encoding=ENC)
    xlsx = [f for f in os.listdir(DATASET) if f.endswith(".xlsx")][0]
    dic = pd.read_excel(os.path.join(DATASET, xlsx), sheet_name=5).dropna()
    dic.columns = ["province", "city_name", "city_id"]
    dic["city_name"] = dic.city_name.astype(str).str.strip()
    cid = dict(zip(dic.city_name, dic.city_id.astype(str).str.replace(r"\.0$", "", regex=True)))
    units = set(pd.read_csv(os.path.join(WEATHER, "city_daily_weather_2017.csv"), encoding=ENC, usecols=["city_name"]).city_name.astype(str).str.strip())
    hosts = set(pd.read_csv(os.path.join(INTERMEDIATE, "city_year_network.csv"), encoding=ENC).city_name)
    tab = members[["city_name"]].copy()
    tab["city_id"] = members.city_name.map(cid).fillna("")
    for c in ["cluster_en", "cluster_zh", "fyp14_tier", "source", "approximate_flag", "note"]:
        tab[c] = members[c]
    tab["in_369_grid_units"] = members.city_name.isin(units).astype(int)
    tab["unionpay_host"] = members.city_name.isin(hosts).astype(int)
    tab.to_csv(os.path.join(INTERMEDIATE, "city_clusters_19.csv"), index=False, encoding=ENC)

    # Within- vs outside-cluster S of hosts' usable inbound partners (bootstrap over hosts, 2,000 draws, seed 5)
    CL = {k: list(g.city_name) for k, g in members.groupby("cluster_en", sort=False)}
    pn = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC)
    pn = pn[pn.usable == 1].copy()
    pn["wpi"] = pn.w * pn.pi

    def inside_outside(CLd, seed=5):
        rng = np.random.default_rng(seed)
        out = []
        for yr in [2018, 2019]:
            q_ = pn[pn.network_year == yr]
            for c, mem in CLd.items():
                ms = set(mem)
                hq = q_[q_.host_name.isin(ms)].copy()
                if hq.empty:
                    continue
                hq["inc"] = hq.guest_name.isin(ms)
                g = hq.groupby("host_name")
                t = pd.DataFrame({"w_in": g.apply(lambda d: d.w[d.inc].sum()), "w_out": g.apply(lambda d: d.w[~d.inc].sum()),
                                  "m_in": g.apply(lambda d: d.wpi[d.inc].sum()), "m_out": g.apply(lambda d: d.wpi[~d.inc].sum())})
                t["S_in"] = t.m_in / t.w_in
                t["S_out"] = t.m_out / t.w_out
                t["wshare_in"] = t.w_in / (t.w_in + t.w_out)
                t["mshare_in"] = t.m_in / (t.m_in + t.m_out)
                both = t[(t.w_in > 0) & (t.w_out > 0)]
                dd = (both.S_in - both.S_out).to_numpy()
                if len(dd) >= 2:
                    bs = np.array([rng.choice(dd, len(dd)).mean() for _ in range(2000)])
                    lo, hi = np.quantile(bs, [.025, .975])
                else:
                    lo = hi = np.nan
                out.append(dict(year=yr, cluster=c, n_hosts=len(t), n_both=len(both), S_in=both.S_in.mean(), S_out=both.S_out.mean(),
                                d=dd.mean() if len(dd) else np.nan, ci_lo=lo, ci_hi=hi, weight_share_in=t.wshare_in.mean(),
                                mass_share_in=t.mshare_in.mean()))
        return pd.DataFrame(out)

    res = inside_outside(CL)
    res.to_csv(os.path.join(INTERMEDIATE, "cluster19_inside_outside.csv"), index=False, encoding=ENC)
    print("membership rows", len(tab), "host members", tab.loc[tab.unionpay_host == 1, "city_name"].nunique(), "cluster-year rows", len(res))


def incidence_by_city_attributes():
    """Who carries shared-rain exposure: differences in the shared-rain index S and in K = E x S (inbound share times S)
    between the top and bottom terciles (or groups) of host attributes in 2018 and 2019, raw and after removing province
    and smooth geographic trends, with host bootstrap intervals and Holm-adjusted p-values."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    SECTOR_MONTHLY = os.path.join(DATASET, "\u57ce\u5e02\u5206\u884c\u4e1a\u6d3b\u529b\u6307\u6570_\u6708\u5ea6.csv")  # licensed city-sector monthly file
    CITY_MONTHLY = os.path.join(DATASET, "\u57ce\u5e02\u7efc\u5408\u6d3b\u529b\u6307\u6570_\u6708\u5ea6.csv")  # licensed city monthly file
    ENC = "utf-8-sig"
    warnings.filterwarnings("ignore")
    q = lambda x: np.expm1(x / 10.0)

    # Tourism-and-food share of inbound spending and small-ticket share of spending, by city and year
    sec = pd.read_csv(SECTOR_MONTHLY, usecols=["mchnt_city", "sic", "year", "month", "cval_in"], encoding=ENC)
    sec["qin"] = q(sec.cval_in)
    TF = ["H", "H61", "H62", "N78", "R86", "R87", "R88", "R89"]
    g = sec.groupby(["mchnt_city", "year"])
    tf = (sec[sec.sic.isin(TF)].groupby(["mchnt_city", "year"]).qin.sum() / g.qin.sum()).rename("TF_in").reset_index()
    comp = pd.read_csv(CITY_MONTHLY, usecols=["mchnt_city", "year", "month", "cval", "cval_small"], encoding=ENC)
    comp["qs"] = q(comp.cval_small)
    comp["qc"] = q(comp.cval)
    sm = comp.groupby(["mchnt_city", "year"]).apply(lambda d: d.qs.sum() / d.qc.sum()).rename("small_share").reset_index()
    u = tf.merge(sm, on=["mchnt_city", "year"]).rename(columns={"mchnt_city": "city_id", "year": "network_year"})

    # Historical extreme-event frequency (events per year, 1981-2016; extreme days at most one day apart form one event)
    thr = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC)
    thr["city_name"] = thr.city_name.astype(str).str.strip()
    cut = dict(zip(thr.city_name, thr.return_level_3yr_mm.fillna(np.inf)))
    wx = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC) for y in range(1981, 2017)], ignore_index=True)
    wx["city_name"] = wx.city_name.astype(str).str.strip()
    ext = wx.loc[(wx.rain_mm >= 50) | (wx.rain_mm >= wx.city_name.map(cut)), ["city_name", "date"]].copy()
    ext["date"] = pd.to_datetime(ext.date)
    ext = ext.sort_values(["city_name", "date"]).reset_index(drop=True)
    gap = ext.groupby("city_name").date.diff().dt.days
    ext["event_index"] = (gap.isna() | (gap > 1)).astype(int).groupby(ext.city_name).cumsum()
    hef = (ext.groupby("city_name").event_index.max() / 36).rename("HistoricalExtremeFrequency").reset_index()

    # Host attributes by city-year
    cyn = pd.read_csv(os.path.join(INTERMEDIATE, "city_year_network.csv"), encoding=ENC)
    attrs = pd.read_csv(os.path.join(EXTERNAL, "city_attributes.csv"), encoding=ENC)
    members = pd.read_csv(os.path.join(EXTERNAL, "city_cluster_membership.csv"), encoding=ENC)
    H = cyn.merge(u, on=["city_id", "network_year"], how="left").merge(
        attrs[["city_name", "network_year", "gdppc", "logpop", "coast_km", "imperv_share", "high_tier", "pilot", "province"]],
        on=["city_name", "network_year"], how="left").merge(hef, on="city_name", how="left")
    H["HistoricalExtremeFrequency"] = H.HistoricalExtremeFrequency.fillna(0.0)
    H["S"] = 1 - H.HDC
    H["K"] = H.E * H.S
    H["cluster19"] = H.city_name.isin(set(members.city_name)).astype(int)

    def demean(X, fes, tol=1e-10, maxit=2000):
        X = X.astype(float).copy()
        if not fes:
            return X
        codes = [pd.factorize(f)[0] for f in fes]
        for it in range(maxit):
            Xo = X.copy()
            for c in codes:
                cnt = np.bincount(c)
                for j in range(X.shape[1]):
                    X[:, j] -= np.bincount(c, X[:, j])[c] / cnt[c]
            if np.max(np.abs(X - Xo)) < tol:
                break
        return X

    def ols_resid(df, y, xs, fes=()):
        d = df.dropna(subset=[y] + list(xs) + list(fes)).reset_index(drop=True)
        Z = demean(d[[y] + list(xs)].to_numpy(), [d[f].to_numpy() for f in fes])
        Y, X = Z[:, 0], Z[:, 1:]
        b = np.linalg.pinv(X.T @ X) @ X.T @ Y
        return Y - X @ b

    def geo_resid(d, y):
        """y net of province fixed effects and a quadratic surface in latitude and longitude, plus the mean."""
        X = pd.DataFrame({"lat": d.lat, "lon": d.lon, "lat2": d.lat ** 2, "lon2": d.lon ** 2, "latlon": d.lat * d.lon})
        X = (X - X.mean()) / X.std()
        dd = pd.concat([d[[y, "province"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        return ols_resid(dd, y, list(X.columns), fes=["province"]) + d[y].mean()

    def contrast(x, y, kind):
        if kind == "binary":
            return np.median(y[x == 1]) - np.median(y[x == 0]), np.median(y[x == 0]), np.nan, np.median(y[x == 1])
        q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
        lo, hi, mid = y[x <= q1], y[x > q2], y[(x > q1) & (x <= q2)]
        return np.median(hi) - np.median(lo), np.median(lo), np.median(mid), np.median(hi)

    VARS = [("gdppc", "tercile"), ("logpop", "tercile"), ("coast_km", "tercile"), ("TF_in", "tercile"), ("small_share", "tercile"),
            ("imperv_share", "tercile"), ("high_tier", "binary"), ("cluster19", "binary"), ("pilot", "binary"), ("HistoricalExtremeFrequency", "tercile")]
    rng = np.random.default_rng(20261023)
    B = 2000
    rows = []
    for yr in [2018, 2019]:
        d = H[H.network_year == yr].reset_index(drop=True)
        outs = {("S", "raw"): d.S.to_numpy(), ("K", "raw"): d.K.to_numpy(), ("S", "geo"): geo_resid(d, "S"), ("K", "geo"): geo_resid(d, "K")}
        idx = rng.integers(0, len(d), (B, len(d)))
        for (yv, ver), yy in outs.items():
            for v, kind in VARS:
                x = d[v].to_numpy()
                est, lo, mid, hi = contrast(x, yy, kind)
                bs = np.array([contrast(x[ii], yy[ii], kind)[0] for ii in idx])
                p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
                p = max(p, 1 / B)
                rows.append(dict(year=yr, outcome=yv, version=ver, variable=v, kind=kind, diff=est, ci_lo=np.quantile(bs, .025),
                                 ci_hi=np.quantile(bs, .975), p_boot=p, med_low=lo, med_mid=mid, med_high=hi, n=len(d)))
    E2 = pd.DataFrame(rows)

    def holm(p):
        o = np.argsort(p)
        m = len(p)
        adj = np.empty(m)
        run = 0
        for r, i in enumerate(o):
            run = max(run, (m - r) * p[i])
            adj[i] = min(1, run)
        return adj

    E2["p_holm"] = E2.groupby(["year", "version"]).p_boot.transform(lambda s: holm(s.to_numpy()))
    E2["ci_excl0"] = (E2.ci_lo > 0) | (E2.ci_hi < 0)
    w = E2.pivot_table(index=["outcome", "version", "variable"], columns="year", values=["diff", "ci_lo", "ci_hi", "p_holm", "ci_excl0"], aggfunc="first")
    res = []
    for (o, ver, v), r in w.iterrows():
        same = np.sign(r[("diff", 2018)]) == np.sign(r[("diff", 2019)])
        ok = same and r[("ci_excl0", 2018)] and r[("ci_excl0", 2019)] and r[("p_holm", 2018)] < 0.05 and r[("p_holm", 2019)] < 0.05
        res.append(dict(outcome=o, version=ver, variable=v, d2018=r[("diff", 2018)], ci2018=f"[{r[('ci_lo', 2018)]:.3f}, {r[('ci_hi', 2018)]:.3f}]",
                        holm2018=r[("p_holm", 2018)], d2019=r[("diff", 2019)], ci2019=f"[{r[('ci_lo', 2019)]:.3f}, {r[('ci_hi', 2019)]:.3f}]",
                        holm2019=r[("p_holm", 2019)], PASS=ok))
    E2.to_csv(os.path.join(INTERMEDIATE, "E2_incidence_long.csv"), index=False, encoding=ENC)
    pd.DataFrame(res).to_csv(os.path.join(INTERMEDIATE, "E2_incidence_passrule.csv"), index=False, encoding=ENC)
    print("rows", len(E2))


def incidence_tests():
    """Incidence tests: event-level spending losses by transaction size (small vs not-large tickets) and by sector type
    (in-person vs remote), their dependence on network exposure (with a distance-matched placebo), differences across host
    income terciles, and robustness checks."""
    CITY_DAILY = os.path.join(DATASET, "\u57ce\u5e02\u7efc\u5408\u6d3b\u529b\u6307\u6570_\u65e5\u5ea6.csv")  # licensed city-daily file
    ENC = "utf-8-sig"
    FE = "meteorological_event_id"
    VCOV = {"CRV1": f"city_name + {FE}"}
    DIMS = ["city_moy", "datef"]
    PRE_DAYS, MIN_CLEAN_PRE = 14, 7
    IN_PERSON = ["F52", "H", "H61", "H62", "R86", "R87", "R88", "R89", "O79", "O80"]
    REMOTE = ["F51", "K70", "C29", "C36", "Y"]
    q = lambda x: np.expm1(np.asarray(x, dtype="float64") / 10.0)
    HS = (0, 3, 7, 14)
    warnings.filterwarnings("ignore")

    def drop_singletons(frame, dims):
        keep = np.ones(len(frame), bool)
        columns = {d: frame[d].to_numpy() for d in dims}
        while True:
            changed = False
            for d in dims:
                counts = pd.Series(columns[d][keep]).value_counts()
                singles = set(counts[counts == 1].index)
                if singles:
                    mask = keep & pd.Series(columns[d]).isin(singles).to_numpy()
                    if mask.any():
                        keep &= ~mask
                        changed = True
            if not changed:
                return keep

    def fit_ipf(y, codes, nlev, tol=1e-12, maxit=5000):
        """Poisson pseudo-maximum likelihood with categorical fixed effects only, by iterative proportional fitting."""
        y = np.asarray(y, dtype="float64")
        log_mu = np.zeros(len(y))
        log_fe = [np.zeros(k) for k in nlev]
        y_sum = [np.bincount(c, weights=y, minlength=k) for c, k in zip(codes, nlev)]
        for iteration in range(maxit):
            delta = 0.0
            for j, (c, k) in enumerate(zip(codes, nlev)):
                mu_sum = np.bincount(c, weights=np.exp(log_mu), minlength=k)
                with np.errstate(divide="ignore", invalid="ignore"):
                    adjust = np.where(mu_sum > 0, np.log(y_sum[j] / mu_sum), 0.0)
                adjust[~np.isfinite(adjust)] = 0.0
                log_fe[j] += adjust
                log_mu += adjust[c]
                delta = max(delta, float(np.abs(adjust).max()))
            if delta < tol:
                break
        return log_fe, log_mu, iteration + 1, delta

    def load_city_daily():
        """City-daily totals from the licensed file, inverted, 2017-2019."""
        raw = pd.read_csv(CITY_DAILY, dtype={"mchnt_city": str},
                          usecols=["mchnt_city", "date", "cval", "cval_in", "cval_small", "cval_notlarge", "cvol",
                                   "cvol_in", "cvol_small", "cvol_notlarge"])
        raw["date"] = pd.to_datetime(raw.date, format="%d%b%Y")
        raw = raw[(raw.date >= "2017-01-01") & (raw.date <= "2019-12-31")].copy()
        raw["city_id"] = raw.mchnt_city.str.strip()
        up = pd.read_csv(os.path.join(INTERMEDIATE, "unionpay_city_daily_2017_2019.csv"),
                         dtype={"city_id": str}, encoding=ENC)
        up["date"] = pd.to_datetime(up.date)
        up["city_name"] = up.city_name.astype(str).str.strip()
        names = up.drop_duplicates("city_id").set_index("city_id").city_name
        raw["city_name"] = raw.city_id.map(names)
        out = pd.DataFrame({"city_id": raw.city_id, "city_name": raw.city_name, "date": raw.date,
                            "TOT": q(raw.cval), "INB": q(raw.cval_in), "SMALL": q(raw.cval_small),
                            "NOTLARGE": q(raw.cval_notlarge), "TOTv": q(raw.cvol), "INBv": q(raw.cvol_in),
                            "SMALLv": q(raw.cvol_small), "NOTLARGEv": q(raw.cvol_notlarge)})
        out["LOC"] = out.TOT - out.INB
        return out.reset_index(drop=True), up

    def counterfactual(panel, cols, sk):
        """Normal (no-event) spending for every city-day in the event-time panel: PPML with city x month-of-year and date
        fixed effects, trained on all city-days outside the event windows. Returns N_<col> and support flag sup_<col>."""
        p = panel[["city_name", "date", *cols]].copy()
        p["datef"] = p.date.dt.strftime("%Y-%m-%d")
        p["city_moy"] = p.city_name + "|" + p.date.dt.month.astype(str)
        wk = sk[["city_name", "date"]].drop_duplicates()
        hold = set(zip(wk.city_name, wk.date))
        p["hold"] = np.fromiter(((c, d) in hold for c, d in zip(p.city_name, p.date)), bool, len(p))
        tr = p[~p.hold]
        days = wk.copy().reset_index(drop=True)
        days["datef"] = days.date.dt.strftime("%Y-%m-%d")
        days["city_moy"] = days.city_name + "|" + days.date.dt.month.astype(str)
        for o in cols:
            t = tr[tr[o].notna()]
            sub = t.loc[drop_singletons(t, DIMS)]
            cats, codes, nlev = {}, [], []
            for d in DIMS:
                cat = pd.Categorical(sub[d])
                cats[d] = cat.categories
                codes.append(cat.codes.astype(np.int64))
                nlev.append(len(cat.categories))
            log_fe, log_mu, it, dl = fit_ipf(sub[o].to_numpy(), codes, nlev)
            tot = np.zeros(len(days))
            sup = np.ones(len(days), bool)
            for j, d in enumerate(DIMS):
                ys = np.bincount(codes[j], weights=sub[o].to_numpy(), minlength=nlev[j])
                s_ = pd.Series(np.where(ys > 0, log_fe[j], np.nan), index=cats[d])
                v = days[d].map(s_)
                sup &= v.notna().to_numpy()
                tot += v.fillna(0).to_numpy()
            pred = np.where(sup, np.exp(tot), np.nan)
            days["N_" + o] = pred
            days["sup_" + o] = (sup & np.isfinite(pred) & (pred > 0)).astype(int)
        return days.drop(columns=DIMS)

    def load_cf_skeleton():
        cols = ["city_event_id", FE, "city_name", "date", "event_time_from_start", "event_time_from_end",
                "is_focal_exposure_day", "is_other_city_event_exposure", "city_event_duration", "cf_complete_end14",
                "local_spending", "normal_local_spending", "merchant_place_spending", "normal_merchant_place_spending"]
        sk = pd.read_csv(os.path.join(INTERMEDIATE, "counterfactual_panel.csv"), encoding=ENC, usecols=cols)
        sk["date"] = pd.to_datetime(sk.date)
        sk["city_name"] = sk.city_name.astype(str).str.strip()
        return sk

    def event_metrics(sk, days, panel, comp, events, label):
        """One row per event: day-0 change, current shock and cumulative deficit CD_h (onset to h days after the end),
        all relative to the clean pre-event baseline; events need full counterfactual support from end-14 to end+14."""
        fr = sk.merge(days, on=["city_name", "date"], how="left").merge(
            panel[["city_name", "date", *comp]], on=["city_name", "date"], how="left")
        fr["Y"] = fr[comp].sum(axis=1, min_count=len(comp))
        fr["Y0"] = fr[["N_" + c for c in comp]].sum(axis=1, min_count=len(comp))
        fr["sup"] = np.prod([fr["sup_" + c].fillna(0).to_numpy() for c in comp], axis=0)
        fr["k"] = fr.event_time_from_start.astype(int)
        fr["ke"] = fr.event_time_from_end.astype(int)
        fr = fr[fr.cf_complete_end14 == 1]
        blk = fr[fr.ke.between(-14, 14)].groupby("city_event_id").sup.agg(["sum", "size"])
        ok = (blk["sum"] == blk["size"]) & (blk["size"] == 29)
        pre_all = fr[fr.k.between(-PRE_DAYS, -1)]
        okp = pre_all.groupby("city_event_id").sup.min() == 1
        good = set(ok[ok].index) & set(okp[okp].index)
        fr = fr[fr.city_event_id.isin(good)].copy()
        fr["A"] = 100.0 * (fr.Y / fr.Y0 - 1.0)
        pre = fr[fr.k.between(-PRE_DAYS, -1) & (fr.is_other_city_event_exposure == 0)]
        e = fr.groupby("city_event_id").agg(city_name=("city_name", "first"), meteorological_event_id=(FE, "first"))
        e["clean_pre_days"] = pre.groupby("city_event_id").size()
        e["CleanPreMean"] = pre.groupby("city_event_id").A.mean()
        e["day0"] = fr[fr.k == 0].groupby("city_event_id").A.mean() - e.CleanPreMean
        e["CurrentShock"] = fr[fr.is_focal_exposure_day == 1].groupby("city_event_id").A.mean() - e.CleanPreMean
        c = (pre_all.Y / pre_all.Y0).groupby(pre_all.city_event_id).mean()
        fr["Y0t"] = fr.Y0 * fr.city_event_id.map(c)
        for h in HS:
            w = fr[(fr.k >= 0) & (fr.ke <= h)].groupby("city_event_id")
            s0, sy, nd = w.Y0t.sum(), w.Y.sum(), w.size()
            e[f"CD_{h}"] = 100.0 * (s0 - sy) / s0
            e[f"NDE_{h}"] = (s0 - sy) / (s0 / nd)
        e = e.reset_index()
        e = e[e.city_event_id.isin(set(events))]
        e = e[e.clean_pre_days >= MIN_CLEAN_PRE]
        e.insert(0, "margin", label)
        return e.reset_index(drop=True)

    def mean_test(df, col, vcov=VCOV):
        """Mean with two-way clustered (city, rainfall system) standard error; t(min G - 1); MDE at 80% power."""
        import pyfixest as pf
        d = df[[col, "city_name", FE]].dropna().rename(columns={col: "y"})
        m = pf.feols("y ~ 1", data=d, vcov=vcov)
        r = m.tidy().loc["Intercept"]
        G = np.atleast_1d(np.asarray(m._G, float))
        dof = int(G.min()) - 1
        se = float(r["Std. Error"])
        mde = (st.t.ppf(0.975, dof) + st.t.ppf(0.80, dof)) * se
        return dict(est=float(r["Estimate"]), se=se, ci_low=float(r["2.5%"]), ci_high=float(r["97.5%"]),
                    p=float(r["Pr(>|t|)"]), N=int(m._N), G_city=int(d.city_name.nunique()), G_sys=int(d[FE].nunique()),
                    dof=dof, MDE80=mde)

    def lincomb(m, wmap):
        order = list(m._coefnames)
        w = np.zeros(len(order))
        for k, v in wmap.items():
            names = [n for n in order if n == k or set(n.split(":")) == set(k.split(":"))]
            if not names:
                raise KeyError(k)
            w[order.index(names[0])] += v
        est = float(w @ m._beta_hat.flatten())
        se = float(np.sqrt(w @ m._vcov @ w))
        G = np.atleast_1d(np.asarray(getattr(m, "_G", []) or [], dtype=float))
        df = int(G.min()) - 1 if G.size else int(m._N - len(order))
        tcrit = st.t.ppf(.975, df)
        return dict(est=est, se=se, ci_low=est - tcrit * se, ci_high=est + tcrit * se, p=float(2 * st.t.sf(abs(est / se), df)),
                    dof=df, N=int(m._N), MDE80=(tcrit + st.t.ppf(0.80, df)) * se)

    def tropical_cyclone_flag(events, radius_km=500):
        """1 if an IBTrACS Western Pacific fix (Beijing time) lies within radius_km of the focal city on any day from the
        day before the city event starts to its last day."""
        ib = pd.read_csv(os.path.join(EXTERNAL, "IBTrACS", "ibtracs.WP.list.v04r01.csv"), skiprows=[1],
                         usecols=["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON"], low_memory=False)
        ib = ib[(ib.SEASON >= 1981) & (ib.SEASON <= 2019)].copy()
        ib["date"] = (pd.to_datetime(ib.ISO_TIME) + pd.Timedelta(hours=8)).dt.normalize()
        ib["LAT"] = pd.to_numeric(ib.LAT, errors="coerce")
        ib["LON"] = pd.to_numeric(ib.LON, errors="coerce")
        ib = ib[ib.LAT.between(10, 45) & ib.LON.between(100, 135)]

        def hav(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
            return 6371 * 2 * np.arcsin(np.sqrt(a))

        out = []
        for r in events.itertuples():
            s = pd.to_datetime(r.city_event_start) - pd.Timedelta(days=1)
            e = pd.to_datetime(r.city_event_end)
            sub = ib[(ib.date >= s) & (ib.date <= e)]
            out.append(0 if len(sub) == 0 else int(hav(r.focal_lat, r.focal_lon, sub.LAT.values, sub.LON.values).min() <= radius_km))
        return np.array(out)

    def remote_heavy_cities(id2name, threshold=0.50):
        """Cities whose mean 2017-2019 share of inbound spending in remote sectors exceeds the threshold."""
        remote_sectors = {"D", "G56", "I63", "I64", "I65", "J66", "J67", "J68", "J69", "K70", "Y"}
        an = pd.read_parquet(os.path.join(INTERMEDIATE, "sector_annual_sums.parquet"))
        an = an[an.year <= 2019].copy()
        an["city_name"] = an.mchnt_city.str.strip().map(id2name)
        an["remote"] = an.sic.isin(remote_sectors)
        cy = an.groupby(["city_name", "year"]).apply(lambda g: g.loc[g.remote, "I_sum"].sum() / g.I_sum.sum())
        share = cy.groupby(level=0).mean()
        return set(share[share > threshold].index)

    # Event-level outcomes for every margin (value and volume)
    city, up = load_city_daily()
    city["LOCv"] = city.TOTv - city.INBv
    A = pd.read_parquet(os.path.join(INTERMEDIATE, "sector_sic_daily.parquet"))
    names = city.drop_duplicates("city_id").set_index("city_id").city_name
    A["city_name"] = A.mchnt_city.str.strip().map(names)
    A["grp"] = np.where(A.sic.isin(IN_PERSON), "INP", "REM")
    G = A.groupby(["city_name", "date", "grp"])[["V", "N"]].sum().unstack("grp")
    G.columns = [f"{a}_{b}" for a, b in G.columns]
    grid = city[["city_name", "date"]].merge(G.reset_index(), on=["city_name", "date"], how="left").fillna(0.0)
    grid = grid.rename(columns={"V_INP": "INP", "V_REM": "REM", "N_INP": "INPv", "N_REM": "REMv"})
    city = city.merge(grid, on=["city_name", "date"], how="left")
    sk = load_cf_skeleton()
    ev_ids = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, usecols=["city_event_id"]).city_event_id
    series = ["LOC", "INB", "SMALL", "NOTLARGE", "INP", "REM", "LOCv", "INBv", "SMALLv", "NOTLARGEv", "INPv", "REMv"]
    days = counterfactual(city, series, sk)
    MARG = {"total": ["LOC", "INB"], "small": ["SMALL"], "notlarge": ["NOTLARGE"], "inperson": ["INP"], "remote": ["REM"],
            "local": ["LOC"], "total_v": ["LOCv", "INBv"], "small_v": ["SMALLv"], "notlarge_v": ["NOTLARGEv"],
            "inperson_v": ["INPv"], "remote_v": ["REMv"]}
    O = pd.concat([event_metrics(sk, days, city, c, ev_ids, m) for m, c in MARG.items()], ignore_index=True)

    evf = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, dtype={"city_id": str}, low_memory=False)
    evf["year"] = pd.to_datetime(evf.city_event_start).dt.year
    evf["TC500"] = tropical_cyclone_flag(evf)
    REMOTE_CITIES = remote_heavy_cities(dict(zip(up.city_id.str.strip(), up.city_name)))
    PRD = set(pd.read_csv(os.path.join(EXTERNAL, "pearl_river_delta_cities.csv"), encoding=ENC).city_name)
    meta = evf[["city_event_id", "year", "TC500"]]
    STATS = ["day0", "CD_0", "CD_3", "CD_7", "CD_14", "CurrentShock"]
    MAIN = ["total", "small", "notlarge", "inperson", "remote"]

    def wide(O, suffix=""):
        W = None
        for m in MAIN:
            x = O[O.margin == m + suffix][["city_event_id", "city_name", FE, *STATS]].rename(columns={s: f"{s}_{m}" for s in STATS})
            W = x if W is None else W.merge(x, on=["city_event_id", "city_name", FE], how="inner")
        for s in STATS:
            W[f"{s}_small_minus_notlarge"] = W[f"{s}_small"] - W[f"{s}_notlarge"]
            W[f"{s}_inperson_minus_remote"] = W[f"{s}_inperson"] - W[f"{s}_remote"]
        return W.merge(meta, on="city_event_id", how="left")

    W = wide(O)
    rows = []
    for col in [c for c in W.columns if c.split("_")[0] in ("day0", "CD", "CurrentShock")]:
        stat = "_".join(col.split("_")[:2]) if col.startswith("CD_") else col.split("_")[0]
        margin = col[len(stat) + 1:]
        rows.append(dict(statistic=stat, margin=margin, **mean_test(W, col)))
    pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "T2_main_estimates.csv"), index=False, encoding=ENC)

    # Exposure gradient: canonical response specification on each outcome; effect per +0.10 S at the mean E
    import pyfixest as pf
    CAN = "{y} ~ E + E:h + Unknown + z_sev + z_sev:E + z_sev:E:h + z_sev:Unknown | " + FE
    EV = evf.merge(W[["city_event_id", *[f"CurrentShock_{m}" for m in MAIN], *[f"day0_{m}" for m in MAIN],
                      "CurrentShock_small_minus_notlarge", "day0_small_minus_notlarge"]], on="city_event_id", how="left")

    def s_slope(df, y):
        d = df[df.network_sample == 1]
        m = pf.feols(CAN.format(y=y), data=d, vcov=VCOV)
        used = set(m._data.city_event_id)
        Eb = d[d.city_event_id.isin(used)].E.mean()
        return dict(**lincomb(m, {"z_sev:E:h": -0.1 * Eb}), Ebar=float(Eb), n_sys=int(d[d.city_event_id.isin(used)][FE].nunique()))

    K3rows = []
    for y in ["CurrentShock_inbound"] + [f"CurrentShock_{m}" for m in MAIN] + ["CurrentShock_small_minus_notlarge"] + \
             [f"day0_{m}" for m in ["small", "notlarge", "total"]] + ["day0_small_minus_notlarge"]:
        K3rows.append(dict(outcome=y, **s_slope(EV, y)))
    K3T = pd.DataFrame(K3rows)

    # Distance-matched placebo: each partner's co-exposure replaced by that of a random unit at a similar distance
    # (within +-10%, widened in 10-point steps); 2017-2018 networks; 1,999 draws
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    net = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC,
                      dtype={"city_id": str, "host_id": str, "guest_id": str}, low_memory=False)
    d = EV[EV.in_estimation_sample == 1].reset_index(drop=True)
    Eb = d.E.mean()
    gcode = pd.factorize(d[FE])[0]
    ng = np.bincount(gcode)

    def dm(A_):
        A_ = np.asarray(A_, float)
        A_ = A_[:, None] if A_.ndim == 1 else A_
        s = np.zeros((gcode.max() + 1, A_.shape[1]))
        np.add.at(s, gcode, A_)
        return A_ - (s / ng[:, None])[gcode]

    E_, z_, Xu = d.E.values, d.z_sev.values, d.Unknown.values

    def effect(yv, h):
        X = dm(np.column_stack([E_, E_ * h, Xu, z_, z_ * E_, z_ * E_ * h, z_ * Xu]))
        return -0.1 * Eb * np.linalg.lstsq(X, dm(yv)[:, 0], rcond=None)[0][5]

    YS = {y: d[y].values for y in ["CurrentShock_inbound", "CurrentShock_small", "CurrentShock_notlarge", "CurrentShock_total",
                                    "CurrentShock_small_minus_notlarge", "CurrentShock_inperson", "day0_small"]}
    U = net[(net.usable == 1) & net.network_year.isin([2017, 2018])].copy()
    for c in ["host_name", "guest_name"]:
        U[c] = U[c].astype(str).str.strip()
    U = U.reset_index(drop=True)
    key = pd.factorize(pd.Series(list(zip(U.host_name, U.network_year))))[0]
    wsum = np.bincount(key, weights=U.w.values)
    kmap = {k: i for k, i in zip(zip(U.host_name, U.network_year), key)}
    evk = np.array([kmap[(c, y)] for c, y in zip(d.city_name, d.baseline_year)])
    byf = {f: (g_.partner_city.values, g_.distance_km.values, g_.pi.values) for f, g_ in hp[hp.pi.notna()].groupby("focal_city")}
    cand = []
    for r in U.itertuples(index=False):
        pc, dd, pp = byf[r.host_name]
        ns = pc != r.guest_name
        idx = np.where((np.abs(dd / r.distance_km - 1) <= 0.10) & ns)[0]
        w_ = 0.10
        if len(idx) < 1:
            while True:
                w_ += 0.10
                idx = np.where((np.abs(dd / r.distance_km - 1) <= w_ + 1e-12) & ns)[0]
                if len(idx):
                    break
        cand.append(pp[idx])
    NP, SEED = 1999, 20261023
    rng = np.random.default_rng(SEED)
    P = np.empty((len(U), NP))
    for k, c in enumerate(cand):
        P[k] = c[rng.integers(0, len(c), NP)]
    hP = 1 - (np.vstack([np.bincount(key, weights=U.w.values * P[:, p]) for p in range(NP)]).T / wsum[:, None])[evk]
    PL = pd.DataFrame({y: [effect(v, hP[:, p]) for p in range(NP)] for y, v in YS.items()})
    plsum = []
    for y in YS:
        obs = float(K3T[K3T.outcome == y].est.iloc[0])
        x = PL[y].values
        plsum.append(dict(outcome=y, observed=obs, placebo_mean=x.mean(), placebo_q025=np.quantile(x, .025),
                          placebo_q975=np.quantile(x, .975), share_placebo_at_least_as_negative=float((x <= obs).mean()),
                          observed_minus_placebo_mean=obs - x.mean(), draws=NP))
    K3T = K3T.merge(pd.DataFrame(plsum).drop(columns="observed"), on="outcome", how="left")
    K3T.to_csv(os.path.join(INTERMEDIATE, "T2_K3_exposure_gradient.csv"), index=False, encoding=ENC)

    # Host income terciles (mean GDP per capita over 2017-2019)
    cov = pd.read_csv(os.path.join(EXTERNAL, "city_attributes.csv"), encoding=ENC)
    cm = cov.groupby("city_name").gdppc.mean().dropna()
    terc = pd.qcut(cm, 3, labels=["T1", "T2", "T3"])
    W["inc_terc"] = W.city_name.map(terc)
    K4rows = []
    for m in ["small", "inperson", "total"]:
        dd = W[W.inc_terc.notna()][["city_name", FE, f"CD_14_{m}", "inc_terc"]].rename(columns={f"CD_14_{m}": "y"})
        dd["T2d"] = (dd.inc_terc == "T2").astype(float)
        dd["T3d"] = (dd.inc_terc == "T3").astype(float)
        mm = pf.feols("y ~ T2d + T3d", data=dd, vcov=VCOV)
        K4rows.append(dict(margin=m, contrast="CD_14 top - bottom GDP-pc tercile", **lincomb(mm, {"T3d": 1.0}),
                           mean_T1=float(dd[dd.inc_terc == "T1"].y.mean()), mean_T2=float(dd[dd.inc_terc == "T2"].y.mean()),
                           mean_T3=float(dd[dd.inc_terc == "T3"].y.mean()),
                           n_T1=int((dd.inc_terc == "T1").sum()), n_T2=int((dd.inc_terc == "T2").sum()), n_T3=int((dd.inc_terc == "T3").sum())))
    K4T = pd.DataFrame(K4rows)
    ps = K4T.p.values
    order = np.argsort(ps)
    adj = np.empty(3)
    running = 0
    for j, i in enumerate(order):  # Holm adjustment
        running = max(running, min(1.0, (3 - j) * ps[i]))
        adj[i] = running
    K4T["p_holm"] = adj
    K4T.to_csv(os.path.join(INTERMEDIATE, "T2_K4_income_terciles.csv"), index=False, encoding=ENC)

    # Robustness
    def summarise(W_, label, ev_=None):
        out = []
        for s, m in [("day0", "small_minus_notlarge"), ("CD_14", "small"), ("CD_14", "notlarge"), ("CD_14", "small_minus_notlarge"),
                     ("CD_14", "inperson"), ("CD_14", "remote"), ("CD_14", "inperson_minus_remote"), ("day0", "small"), ("day0", "notlarge")]:
            out.append(dict(spec=label, quantity=f"{s}|{m}", **mean_test(W_, f"{s}_{m}")))
        if ev_ is not None:
            try:
                out.append(dict(spec=label, quantity="K3 S-slope|CurrentShock_small", **s_slope(ev_, "CurrentShock_small")))
            except Exception as e_:
                out.append(dict(spec=label, quantity="K3 S-slope|CurrentShock_small", est=np.nan, note=str(e_)[:80]))
        return out

    def winsorise(W_):
        W2 = W_.copy()
        for m in MAIN:
            for s in STATS:
                c = f"{s}_{m}"
                lo, hi = W2[c].quantile([.01, .99])
                W2[c] = W2[c].clip(lo, hi)
        for s in STATS:
            W2[f"{s}_small_minus_notlarge"] = W2[f"{s}_small"] - W2[f"{s}_notlarge"]
            W2[f"{s}_inperson_minus_remote"] = W2[f"{s}_inperson"] - W2[f"{s}_remote"]
        return W2

    Wv = wide(O, "_v").merge(meta, on="city_event_id", how="left", suffixes=("", "_dup"))
    EVv = evf.merge(Wv[["city_event_id", "CurrentShock_small"]], on="city_event_id", how="left")
    sub = lambda mask: EV[EV.city_event_id.isin(set(W.loc[mask, "city_event_id"]))]
    ROB = []
    ROB += summarise(W, "Main (value, 643 events)", EV)
    ROB += summarise(Wv, "Volume (cvol_*)", EVv)
    ROB += summarise(W[~W.city_name.isin(PRD)], "Excluding PRD (9 cities)", sub(~W.city_name.isin(PRD)))
    ROB += summarise(W[W.TC500 != 1], "Excluding tropical-cyclone events (IBTrACS <=500 km)", sub(W.TC500 != 1))
    ROB += summarise(W[~W.city_name.isin(REMOTE_CITIES)], f"Excluding {len(REMOTE_CITIES)} remote-heavy cities", sub(~W.city_name.isin(REMOTE_CITIES)))
    Ww = winsorise(W)
    EVw = evf.merge(Ww[["city_event_id", "CurrentShock_small"]], on="city_event_id", how="left")
    ROB += summarise(Ww, "Winsorised 1/99 (event-level outcomes)", EVw)
    ROB += summarise(W[W.year == 2018], "2018 events only", sub(W.year == 2018))
    ROB += summarise(W[W.year == 2019], "2019 events only", sub(W.year == 2019))
    pd.DataFrame(ROB).to_csv(os.path.join(INTERMEDIATE, "T2_robustness.csv"), index=False, encoding=ENC)
    print("events", len(W), "remote cities", len(REMOTE_CITIES), "TC events", int((W.TC500 == 1).sum()))


def main():
    city_clusters()
    incidence_by_city_attributes()
    incidence_tests()


if __name__ == "__main__":
    main()
