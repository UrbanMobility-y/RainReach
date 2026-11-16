"""Cumulative spending deficits up to 28 days after the last day of extreme rain for city totals, local and inbound spending and in-person and remote or large-ticket sectors, with the daily path of the in-person minus remote difference and its robustness checks."""
import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = r"E:\Extreme Rainfall and UnionPay Spending"
DATASET = os.path.join(ROOT, "Dataset")
EXTERNAL = os.path.join(ROOT, "External data")
INTERMEDIATE = os.path.join(ROOT, "Intermediate")


def recovery_tests():
    """Recovery after extreme rainfall: event-level cumulative deficits up to 28 days after the end of rainfall for city
    totals, local, inbound, in-person and remote sectors (event_outcomes.parquet); recovery ratios CD_h / CD_0 with a
    rainfall-system bootstrap, levels on events with a clean 28-day window, in-person vs remote sector deficits, and
    depth and recovery by event class."""
    CITY_DAILY = os.path.join(DATASET, "\u57ce\u5e02\u7efc\u5408\u6d3b\u529b\u6307\u6570_\u65e5\u5ea6.csv")  # licensed city-daily file
    ENC = "utf-8-sig"
    FE = "meteorological_event_id"
    VCOV = {"CRV1": f"city_name + {FE}"}
    DIMS = ["city_moy", "datef"]
    PRE_DAYS, MIN_CLEAN_PRE = 14, 7
    IN_PERSON = ["F52", "H", "H61", "H62", "R86", "R87", "R88", "R89", "O79", "O80"]
    REMOTE = ["F51", "K70", "C29", "C36", "Y"]
    q = lambda x: np.expm1(np.asarray(x, dtype="float64") / 10.0)
    HS = (0, 3, 7, 14, 21, 28)
    SEED, NBOOT = 20261110, 999
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

    def load_skeleton():
        cols = ["city_event_id", FE, "city_name", "date", "event_time_from_start", "event_time_from_end",
                "is_focal_exposure_day", "is_other_city_event_exposure", "city_event_duration", "cf_complete_end14",
                "full_post_28d", "affected_city_count", "days_until_next_city_event", "peak_rain_mm",
                "local_spending", "normal_local_spending", "inbound_spending", "normal_inbound_spending",
                "merchant_place_spending", "normal_merchant_place_spending"]
        sk = pd.read_csv(os.path.join(INTERMEDIATE, "counterfactual_panel.csv"), encoding=ENC, usecols=cols)
        sk["date"] = pd.to_datetime(sk.date)
        sk["city_name"] = sk.city_name.astype(str).str.strip()
        return sk

    def event_metrics(sk, days, panel, comp, events, label):
        """Event-level day-0 change, current shock and deficits at horizons 0-28 days after the end of rainfall.
        CD_h: percent of normal spending in the window; ND_h: in normal-day units; FD_h: fixed event-day denominator.
        Horizons beyond 14 days require counterfactual support on every day from end-14 to end+28."""
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
        b28 = fr[fr.ke.between(-14, 28)].groupby("city_event_id").agg(s=("sup", "sum"), n=("sup", "size"), y=("Y", "count"))
        e["sup28"] = ((b28.s == b28.n) & (b28.n == 43) & (b28.y == 43)).astype(int)
        e["full_post_28d"] = fr.groupby("city_event_id").full_post_28d.first()
        e["other_post28"] = fr[fr.ke.between(1, 28)].groupby("city_event_id").is_other_city_event_exposure.max()
        e["other_post14"] = fr[fr.ke.between(1, 14)].groupby("city_event_id").is_other_city_event_exposure.max()
        w0 = fr[(fr.k >= 0) & (fr.ke <= 0)].groupby("city_event_id")
        den0 = w0.Y0t.sum() / w0.size()
        for h in HS:
            w = fr[(fr.k >= 0) & (fr.ke <= h)].groupby("city_event_id")
            s0, sy, nd = w.Y0t.sum(), w.Y.sum(), w.size()
            cd, nde, fd = 100.0 * (s0 - sy) / s0, (s0 - sy) / (s0 / nd), 100.0 * (s0 - sy) / den0
            if h > 14:
                bad = e.sup28 != 1
                cd, nde, fd = cd.where(~bad), nde.where(~bad), fd.where(~bad)
            e[f"CD_{h}"] = cd
            e[f"NDE_{h}"] = nde
            e[f"ND_{h}"] = 100.0 * nde
            e[f"FD_{h}"] = fd
            e[f"ndays_{h}"] = nd
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

    def ratio_test(df, num, den, thr=0.25):
        """Ratio of means mean(num) / mean(den): rainfall-system cluster bootstrap (999 multinomial draws, fixed seed)
        and a delta-method two-way clustered bound."""
        import pyfixest as pf
        d = df[[num, den, "city_name", FE]].dropna().reset_index(drop=True)
        a, b = d[num].mean(), d[den].mean()
        R = a / b
        g, ug = pd.factorize(d[FE])
        G = len(ug)
        Sn = np.bincount(g, weights=d[num].to_numpy(), minlength=G)
        Sd = np.bincount(g, weights=d[den].to_numpy(), minlength=G)
        rng = np.random.default_rng(SEED)
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=NBOOT).astype(float)
        bn, bd = W @ Sn, W @ Sd
        r = bn / bd
        d["u"] = (d[num] - R * d[den]) / b
        m = pf.feols("u ~ 1", data=d, vcov=VCOV)
        se = float(m.se()["Intercept"])
        dof = int(np.atleast_1d(np.asarray(m._G, float)).min()) - 1
        return dict(num_mean=a, den_mean=b, ratio=R, RF=1 - R,
                    boot_ub95_one_sided=float(np.quantile(r, 0.95)), boot_ci_low=float(np.quantile(r, 0.025)),
                    boot_ci_high=float(np.quantile(r, 0.975)), RF_ci_low=float(1 - np.quantile(r, 0.975)),
                    RF_ci_high=float(1 - np.quantile(r, 0.025)), boot_share_den_le0=float((bd <= 0).mean()),
                    boot_p_ratio_ge_thr=float((r >= thr).mean()),
                    delta_se=se, delta_ub95_one_sided=float(R + st.t.ppf(0.95, dof) * se), delta_dof=dof,
                    N=int(len(d)), G_sys=int(G), G_city=int(d.city_name.nunique()), thr=thr,
                    pass_=bool(np.quantile(r, 0.95) < thr), pass_delta=bool(R + st.t.ppf(0.95, dof) * se < thr))

    # Event-level outcomes
    city, up = load_city_daily()
    city["LOCv"] = city.TOTv - city.INBv
    A = pd.read_parquet(os.path.join(INTERMEDIATE, "sector_sic_daily.parquet"))
    names = city.drop_duplicates("city_id").set_index("city_id").city_name
    A["city_name"] = A.mchnt_city.str.strip().map(names)
    A = A[A.sic.isin(IN_PERSON + REMOTE)]
    A["grp"] = np.where(A.sic.isin(IN_PERSON), "INP", "REM")
    G = A.groupby(["city_name", "date", "grp"])[["V", "N"]].sum().unstack("grp")
    G.columns = [f"{a}_{b}" for a, b in G.columns]
    grid = city[["city_name", "date"]].merge(G.reset_index(), on=["city_name", "date"], how="left").fillna(0.0)
    grid = grid.rename(columns={"V_INP": "INP", "V_REM": "REM", "N_INP": "INPv", "N_REM": "REMv"})
    city = city.merge(grid, on=["city_name", "date"], how="left")
    sk = load_skeleton()
    ev_ids = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, usecols=["city_event_id"]).city_event_id
    series = ["LOC", "INB", "INP", "REM", "LOCv", "INBv", "INPv", "REMv"]
    days = counterfactual(city, series, sk)
    MARG = {"total": ["LOC", "INB"], "local": ["LOC"], "inbound": ["INB"], "inperson": ["INP"], "remote": ["REM"],
            "total_v": ["LOCv", "INBv"], "local_v": ["LOCv"], "inbound_v": ["INBv"], "inperson_v": ["INPv"], "remote_v": ["REMv"]}
    O = pd.concat([event_metrics(sk, days, city, c, ev_ids, m) for m, c in MARG.items()], ignore_index=True)
    O.to_parquet(os.path.join(INTERMEDIATE, "event_outcomes.parquet"), index=False)

    # Event classes
    evf = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, low_memory=False)[
        ["city_event_id", "city_event_start", "city_event_duration", "peak_rain_mm", "z_sev"]]
    evf["year"] = pd.to_datetime(evf.city_event_start).dt.year
    evf["ym"] = pd.to_datetime(evf.city_event_start).dt.strftime("%Y-%m")
    fl = pd.read_csv(os.path.join(INTERMEDIATE, "event_flags.csv"), encoding=ENC)[["city_event_id", "TC500", "lat"]]
    pe = sk.groupby("city_event_id").agg(extent=("affected_city_count", "first"), nxt=("days_until_next_city_event", "first")).reset_index()
    meta = evf.merge(fl, on="city_event_id", how="left").merge(pe, on="city_event_id", how="left")
    CLS = {"all": np.ones(len(meta), bool), "severe (peak >= 100 mm)": meta.peak_rain_mm >= 100, "not severe": meta.peak_rain_mm < 100,
           "wide (> 15 cities)": meta.extent > 15, "not wide": meta.extent <= 15, "isolated (1 city)": meta.extent == 1,
           "tropical cyclone (TC500)": meta.TC500 == 1, "not tropical cyclone": meta.TC500 != 1,
           "multi-day (>= 2 d)": meta.city_event_duration >= 2, "single-day": meta.city_event_duration < 2,
           "[exploratory] north of 33N": meta.lat > 33, "[exploratory] south of 33N": meta.lat <= 33,
           "[exploratory] next event within 14 d": meta.nxt <= 14, "[exploratory] no next event within 14 d": ~(meta.nxt <= 14)}
    CLS = {k: set(meta.city_event_id[np.asarray(v, bool)]) for k, v in CLS.items()}
    KEY = ["all", "severe (peak >= 100 mm)", "wide (> 15 cities)", "tropical cyclone (TC500)"]
    UNIT = {"ND": "normal-day % (primary for ratios)", "CD": "canonical % (diluted by window length)", "FD": "normal-day %, fixed denominator"}
    MARGINS = ["total", "local", "inbound", "inperson", "remote"]

    def frame(margin, suffix=""):
        d = O[O.margin == margin + suffix].merge(meta, on="city_event_id", how="left")
        d["q4"] = ((d.sup28 == 1) & (d.full_post_28d == 1) & (d.other_post28 == 0) & ~(d.nxt <= 28)).astype(int)
        return d

    F = {m: frame(m) for m in MARGINS}
    sub = lambda d, c: d[d.city_event_id.isin(CLS[c])]

    # Recovery ratios CD_h / CD_0
    rows = []
    for m in MARGINS:
        for c in CLS:
            d = sub(F[m], c)
            for u in UNIT:
                for h in (3, 7, 14):
                    if m != "total" and (u != "ND" or h != 14) and not (m in ("local", "inbound") and h == 14):
                        continue
                    rows.append(dict(margin=m, cls=c, unit=u, unit_label=UNIT[u], horizon=h, **ratio_test(d, f"{u}_{h}", f"{u}_0")))
    pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "T5_Q1Q2_recovery_ratio.csv"), index=False, encoding=ENC)

    # Depth and recovery by class
    rows = []
    for m in MARGINS:
        for c in CLS:
            d = sub(F[m], c).copy()
            d["rebound_ND"] = d.ND_0 - d.ND_14
            d["rebound_FD"] = d.FD_0 - d.FD_14
            for s in ["day0", "CurrentShock", "CD_0", "CD_14", "ND_0", "ND_14", "FD_14", "rebound_ND", "rebound_FD"]:
                try:
                    rows.append(dict(margin=m, cls=c, statistic=s, **mean_test(d, s)))
                except Exception as ex:
                    rows.append(dict(margin=m, cls=c, statistic=s, est=float(d[s].mean()), N=int(d[s].notna().sum()), note=str(ex)[:80]))
    pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "T5_Q5_depth_recovery_by_class.csv"), index=False, encoding=ENC)

    # In-person vs remote sectors on events with a clean 28-day window
    def sector_frame(Fd, q4=True):
        a, b = Fd["inperson"], Fd["remote"]
        cols = ["city_event_id", "city_name", FE, "q4", "year"] + [f"{u}_{h}" for u in ("CD", "ND") for h in HS] + ["day0"]
        w = a[cols].merge(b[["city_event_id"] + cols[5:]], on="city_event_id", suffixes=("_inperson", "_remote"))
        for u in ("CD", "ND"):
            for h in HS:
                w[f"{u}_{h}_diff"] = w[f"{u}_{h}_inperson"] - w[f"{u}_{h}_remote"]
        if q4:
            w = w[w.q4 == 1]
        return w

    def q4b_rows(w, spec, classes=KEY):
        out = []
        for c in classes:
            d = sub(w, c)
            for u in ("CD", "ND"):
                for h in (0, 14, 21, 28):
                    for mg in ("inperson", "remote", "diff"):
                        try:
                            out.append(dict(spec=spec, cls=c, unit=u, horizon=h, margin=mg, **mean_test(d, f"{u}_{h}_{mg}")))
                        except Exception as ex:
                            out.append(dict(spec=spec, cls=c, unit=u, horizon=h, margin=mg, note=str(ex)[:80]))
        return out

    Q4B = pd.DataFrame(q4b_rows(sector_frame(F), "clean-28 sample (verdict)") +
                       q4b_rows(sector_frame(F, q4=False), "all 643 events (horizons <= 14 only meaningful)", classes=["all"]))
    Q4B.to_csv(os.path.join(INTERMEDIATE, "T5_Q4b_sectors.csv"), index=False, encoding=ENC)

    # Levels on the clean-28 sample
    rows = []
    for m in ["total", "local", "inbound"]:
        d4 = F[m][F[m].q4 == 1]
        for c in KEY:
            for u in ("CD", "ND"):
                for h in (0, 14, 21, 28):
                    rows.append(dict(margin=m, cls=c, unit=u, horizon=h, **mean_test(sub(d4, c), f"{u}_{h}")))
    pd.DataFrame(rows).to_csv(os.path.join(INTERMEDIATE, "T5_Q4_levels_clean28.csv"), index=False, encoding=ENC)
    print("events per margin", O.groupby("margin").size().to_dict())


def sector_deficit_daily():
    """Cumulative spending deficit of in-person and remote sectors at daily horizons 0-28 days after the end of extreme
    rain, in percent of a normal day, with the in-person minus remote difference; two-way clustered CIs and a
    rainfall-system bootstrap (999 draws)."""
    CITY_DAILY = os.path.join(DATASET, "\u57ce\u5e02\u7efc\u5408\u6d3b\u529b\u6307\u6570_\u65e5\u5ea6.csv")  # licensed city-daily file
    ENC = "utf-8-sig"
    FE = "meteorological_event_id"
    VCOV = {"CRV1": f"city_name + {FE}"}
    DIMS = ["city_moy", "datef"]
    PRE_DAYS, MIN_CLEAN_PRE = 14, 7
    IN_PERSON = ["F52", "H", "H61", "H62", "R86", "R87", "R88", "R89", "O79", "O80"]
    REMOTE = ["F51", "K70", "C29", "C36", "Y"]
    q = lambda x: np.expm1(np.asarray(x, dtype="float64") / 10.0)
    HS = tuple(range(29))
    SEED, NBOOT = 20261110, 999
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

    def load_skeleton():
        cols = ["city_event_id", FE, "city_name", "date", "event_time_from_start", "event_time_from_end",
                "is_focal_exposure_day", "is_other_city_event_exposure", "city_event_duration", "cf_complete_end14",
                "full_post_28d", "affected_city_count", "days_until_next_city_event", "peak_rain_mm",
                "local_spending", "normal_local_spending", "inbound_spending", "normal_inbound_spending",
                "merchant_place_spending", "normal_merchant_place_spending"]
        sk = pd.read_csv(os.path.join(INTERMEDIATE, "counterfactual_panel.csv"), encoding=ENC, usecols=cols)
        sk["date"] = pd.to_datetime(sk.date)
        sk["city_name"] = sk.city_name.astype(str).str.strip()
        return sk

    def event_metrics(sk, days, panel, comp, events, label):
        """Event-level day-0 change, current shock and deficits at horizons 0-28 days after the end of rainfall.
        CD_h: percent of normal spending in the window; ND_h: in normal-day units; FD_h: fixed event-day denominator.
        Horizons beyond 14 days require counterfactual support on every day from end-14 to end+28."""
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
        b28 = fr[fr.ke.between(-14, 28)].groupby("city_event_id").agg(s=("sup", "sum"), n=("sup", "size"), y=("Y", "count"))
        e["sup28"] = ((b28.s == b28.n) & (b28.n == 43) & (b28.y == 43)).astype(int)
        e["full_post_28d"] = fr.groupby("city_event_id").full_post_28d.first()
        e["other_post28"] = fr[fr.ke.between(1, 28)].groupby("city_event_id").is_other_city_event_exposure.max()
        e["other_post14"] = fr[fr.ke.between(1, 14)].groupby("city_event_id").is_other_city_event_exposure.max()
        w0 = fr[(fr.k >= 0) & (fr.ke <= 0)].groupby("city_event_id")
        den0 = w0.Y0t.sum() / w0.size()
        for h in HS:
            w = fr[(fr.k >= 0) & (fr.ke <= h)].groupby("city_event_id")
            s0, sy, nd = w.Y0t.sum(), w.Y.sum(), w.size()
            cd, nde, fd = 100.0 * (s0 - sy) / s0, (s0 - sy) / (s0 / nd), 100.0 * (s0 - sy) / den0
            if h > 14:
                bad = e.sup28 != 1
                cd, nde, fd = cd.where(~bad), nde.where(~bad), fd.where(~bad)
            e[f"CD_{h}"] = cd
            e[f"NDE_{h}"] = nde
            e[f"ND_{h}"] = 100.0 * nde
            e[f"FD_{h}"] = fd
            e[f"ndays_{h}"] = nd
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

    def boot_weights(codes, G, n=NBOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        return rng.multinomial(G, np.full(G, 1.0 / G), size=n).astype(float)

    city, _ = load_city_daily()
    A = pd.read_parquet(os.path.join(INTERMEDIATE, "sector_sic_daily.parquet"))
    names = city.drop_duplicates("city_id").set_index("city_id").city_name
    A["city_name"] = A.mchnt_city.str.strip().map(names)
    A = A[A.sic.isin(IN_PERSON + REMOTE)]
    A["grp"] = np.where(A.sic.isin(IN_PERSON), "INP", "REM")
    G = A.groupby(["city_name", "date", "grp"])[["V"]].sum().unstack("grp")
    G.columns = [b for _, b in G.columns]
    city = city[["city_id", "city_name", "date"]].merge(G.reset_index(), on=["city_name", "date"], how="left")
    city[["INP", "REM"]] = city[["INP", "REM"]].fillna(0.0)
    del A, G
    sk = load_skeleton()
    ev = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, usecols=["city_event_id"]).city_event_id
    days = counterfactual(city, ["INP", "REM"], sk)
    out = {m: event_metrics(sk, days, city, [c], ev, m) for m, c in (("inperson", "INP"), ("remote", "REM"))}

    a, b = out["inperson"], out["remote"]
    cols = ["city_event_id", "city_name", FE] + [f"ND_{h}" for h in HS]
    w = a[cols].merge(b[["city_event_id"] + cols[3:]], on="city_event_id", suffixes=("_inperson", "_remote"))
    for h in HS:
        w[f"ND_{h}_diff"] = w[f"ND_{h}_inperson"] - w[f"ND_{h}_remote"]

    rows = []
    for h in HS:
        for mg in ("inperson", "remote", "diff"):
            col = f"ND_{h}_{mg}"
            t = mean_test(w, col)
            d = w[[col, FE]].dropna()
            g, ug = pd.factorize(d[FE])
            Gs = len(ug)
            S = np.bincount(g, weights=d[col].to_numpy(), minlength=Gs)
            C = np.bincount(g, minlength=Gs).astype(float)
            W = boot_weights(g, Gs)
            bm = (W @ S) / (W @ C)
            rows.append(dict(horizon=h, margin=mg, est=t["est"], crv_ci_low=t["ci_low"], crv_ci_high=t["ci_high"], crv_p=t["p"],
                             boot_ci_low=float(np.quantile(bm, 0.025)), boot_ci_high=float(np.quantile(bm, 0.975)),
                             boot_se=float(bm.std(ddof=1)), N=t["N"], G_sys=t["G_sys"], G_city=t["G_city"]))
    res = pd.DataFrame(rows)
    res.insert(0, "unit", "% of a normal day (ND_h, T5 code)")
    res.to_csv(os.path.join(INTERMEDIATE, "fig4d_sector_cumulative_deficit_daily.csv"), index=False, encoding=ENC)
    print(res.loc[res.margin == "diff", ["horizon", "est", "crv_ci_low", "crv_ci_high", "N"]].to_string(index=False))


def main():
    recovery_tests()
    sector_deficit_daily()


if __name__ == "__main__":
    main()
