"""Shared-rain share of partners inside and outside a host's city cluster (19 clusters, 2017-2019 networks) and
cumulative spending deficits by sector group and sector code, 0-28 days after the event (Supplementary results
supporting Fig. 4). Writes SI3_cluster19_inside_outside.csv and SI3_sector_cumdeficit_daily.csv.
"""
from pathlib import Path
import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
DATASET = ROOT / "Dataset"
INTERMEDIATE = ROOT / "Intermediate"
ENC = "utf-8-sig"
NUM = ["n_hosts", "n_both", "S_in", "S_out", "d", "ci_lo", "ci_hi", "weight_share_in", "mass_share_in"]


def inside_outside(pn, CLd, years, seed=5):
    """Per cluster and network year: mean S of partners inside and outside the cluster, host-bootstrap CI."""
    rng = np.random.default_rng(seed)
    out = []
    for yr in years:
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
            out.append(dict(year=yr, cluster=c, n_hosts=len(t), n_both=len(both), S_in=both.S_in.mean(),
                            S_out=both.S_out.mean(), d=dd.mean() if len(dd) else np.nan, ci_lo=lo, ci_hi=hi,
                            weight_share_in=t.wshare_in.mean(), mass_share_in=t.mshare_in.mean()))
    return pd.DataFrame(out)


def cluster_inside_outside():
    pn = pd.read_csv(INTERMEDIATE / "partner_network.csv", encoding=ENC)
    pn = pn[pn.usable == 1].copy()
    pn["wpi"] = pn.w * pn.pi
    assert sorted(pn.network_year.unique()) == [2017, 2018, 2019]
    c19 = pd.read_csv(INTERMEDIATE / "city_clusters_19.csv", encoding=ENC)
    order = list(pd.unique(c19.cluster_en))
    CLd = {k: list(c19.loc[c19.cluster_en == k, "city_name"]) for k in order}
    assert len(CLd) == 19
    # 2018 and 2019 share one random stream; 2017 has its own (same seed)
    new = pd.concat([inside_outside(pn, CLd, [2018, 2019]), inside_outside(pn, CLd, [2017])], ignore_index=True)

    val = pd.read_csv(INTERMEDIATE / "cluster19_inside_outside.csv", encoding=ENC)
    x = new.merge(val, on=["year", "cluster"], suffixes=("", "_saved"))
    assert len(x) == 38
    d_saved = max(float((x[c] - x[c + "_saved"]).abs().max()) for c in NUM)
    assert d_saved < 1e-9, f"2018/2019 differ from cluster19_inside_outside.csv by {d_saved}"
    hosts = c19.groupby("cluster_en").unionpay_host.sum()
    for yr in (2017, 2018, 2019):
        z = new[new.year == yr].set_index("cluster")
        assert (z.n_hosts == hosts.reindex(z.index)).all(), f"{yr}: host counts differ from city_clusters_19.csv"
    new["source"] = np.where(new.year == 2017, "new (2017 network; own generator, seed 5)",
                             "2018-2019 networks; same values as cluster19_inside_outside.csv")
    new["fyp14_tier"] = new.cluster.map(c19.drop_duplicates("cluster_en").set_index("cluster_en").fyp14_tier)
    new.to_csv(INTERMEDIATE / "SI3_cluster19_inside_outside.csv", index=False, encoding=ENC)
    print(f"  wrote SI3_cluster19_inside_outside.csv ({len(new)} rows); 2018-2019 max diff vs cluster19_inside_outside.csv "
          f"{d_saved:.1e}")


FE = "meteorological_event_id"
VCOV = {"CRV1": f"city_name + {FE}"}
DIMS = ["city_moy", "datef"]
PRE_DAYS, MIN_CLEAN_PRE = 14, 7
HS = tuple(range(29))
SEED, NBOOT = 20261110, 999
INPERSON = ["F52", "H", "H61", "H62", "R86", "R87", "R88", "R89", "O79", "O80"]
REMOTE = ["F51", "K70", "C29", "C36", "Y"]
SECTORS = {"F52": "Retail", "H61": "Accommodation", "H62": "Catering", "R89": "Entertainment",
           "O79": "Resident services", "O80": "Repair services",
           "F51": "Wholesale", "K70": "Real estate", "C36": "Automobiles", "Y": "Administrative fees"}
SPARSE = ["H", "R86", "R87", "R88", "C29"]
# vendor file: daily city activity index
CITY_DAILY_FILE = DATASET / "\u57ce\u5e02\u7efc\u5408\u6d3b\u529b\u6307\u6570_\u65e5\u5ea6.csv"
q = lambda x: np.expm1(np.asarray(x, dtype="float64") / 10.0)


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
    """City-day total and inbound spending from the vendor index (LOC = TOT - INB)."""
    raw = pd.read_csv(CITY_DAILY_FILE, dtype={"mchnt_city": str},
                      usecols=["mchnt_city", "date", "cval", "cval_in"])
    raw["date"] = pd.to_datetime(raw.date, format="%d%b%Y")
    raw = raw[(raw.date >= "2017-01-01") & (raw.date <= "2019-12-31")].copy()
    raw["city_id"] = raw.mchnt_city.str.strip()
    up = pd.read_csv(INTERMEDIATE / "unionpay_city_daily_2017_2019.csv",
                     dtype={"city_id": str}, encoding=ENC, usecols=["city_id", "city_name"])
    up["city_name"] = up.city_name.astype(str).str.strip()
    names = up.drop_duplicates("city_id").set_index("city_id").city_name
    raw["city_name"] = raw.city_id.map(names)
    out = pd.DataFrame({"city_id": raw.city_id, "city_name": raw.city_name, "date": raw.date,
                        "TOT": q(raw.cval), "INB": q(raw.cval_in)})
    out["LOC"] = out.TOT - out.INB
    return out.reset_index(drop=True)


def load_skeleton():
    cols = ["city_event_id", FE, "city_name", "date", "event_time_from_start", "event_time_from_end",
            "is_focal_exposure_day", "is_other_city_event_exposure", "city_event_duration", "cf_complete_end14",
            "full_post_28d", "affected_city_count", "days_until_next_city_event", "peak_rain_mm"]
    sk = pd.read_csv(INTERMEDIATE / "counterfactual_panel.csv", encoding=ENC, usecols=cols)
    sk["date"] = pd.to_datetime(sk.date)
    sk["city_name"] = sk.city_name.astype(str).str.strip()
    return sk


def counterfactual(panel, cols, sk):
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
    info = {}
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
        info[o] = dict(train_n=int(len(sub)), ipf_iter=int(it), ipf_delta=float(dl))
    return days.drop(columns=DIMS), info


def event_metrics_ext(sk, days, panel, comp, events, label):
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
    c = (pre_all.Y / pre_all.Y0).groupby(pre_all.city_event_id).mean()
    fr["Y0t"] = fr.Y0 * fr.city_event_id.map(c)
    b28 = fr[fr.ke.between(-14, 28)].groupby("city_event_id").agg(s=("sup", "sum"), n=("sup", "size"), y=("Y", "count"))
    e["sup28"] = ((b28.s == b28.n) & (b28.n == 43) & (b28.y == 43)).astype(int)
    e["full_post_28d"] = fr.groupby("city_event_id").full_post_28d.first()
    e["other_post28"] = fr[fr.ke.between(1, 28)].groupby("city_event_id").is_other_city_event_exposure.max()
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
        e[f"ND_{h}"] = 100.0 * nde
        e[f"FD_{h}"] = fd
    e = e.reset_index()
    e = e[e.city_event_id.isin(set(events))]
    e = e[e.clean_pre_days >= MIN_CLEAN_PRE]
    e.insert(0, "margin", label)
    return e.reset_index(drop=True)


def mean_test(df, col):
    import pyfixest as pf
    d = df[[col, "city_name", FE]].dropna().rename(columns={col: "y"})
    m = pf.feols("y ~ 1", data=d, vcov=VCOV)
    r = m.tidy().loc["Intercept"]
    G = np.atleast_1d(np.asarray(m._G, float))
    return dict(est=float(r["Estimate"]), se=float(r["Std. Error"]), crv_ci_low=float(r["2.5%"]),
                crv_ci_high=float(r["97.5%"]), crv_p=float(r["Pr(>|t|)"]), N=int(m._N),
                G_city=int(d.city_name.nunique()), G_sys=int(d[FE].nunique()), dof=int(G.min()) - 1)


def boot_weights(G):
    rng = np.random.default_rng(SEED)
    return rng.multinomial(G, np.full(G, 1.0 / G), size=NBOOT).astype(float)


def boot_mean(df, col):
    """Percentile bootstrap over rainfall systems of the event mean."""
    d = df[[col, FE]].dropna()
    g, ug = pd.factorize(d[FE])
    Gs = len(ug)
    S = np.bincount(g, weights=d[col].to_numpy(), minlength=Gs)
    C = np.bincount(g, minlength=Gs).astype(float)
    Wb = boot_weights(Gs)
    bm = (Wb @ S) / (Wb @ C)
    return dict(boot_ci_low=float(np.quantile(bm, 0.025)), boot_ci_high=float(np.quantile(bm, 0.975)),
                boot_se=float(bm.std(ddof=1)))


def sector_cumulative_deficit():
    t0 = time.time()
    city = load_city_daily()
    A = pd.read_parquet(INTERMEDIATE / "sector_sic_daily.parquet")
    names = city.drop_duplicates("city_id").set_index("city_id").city_name
    A["city_name"] = A.mchnt_city.astype(str).str.strip().map(names)
    assert A.city_name.notna().all(), "sector extract: unmatched city codes"
    A = A[A.sic.isin(INPERSON + REMOTE)].copy()
    A["sic"] = A.sic.astype(str)
    # value share of each code within its group and city-day coverage
    grp_of = {s: "inperson" for s in INPERSON} | {s: "remote" for s in REMOTE}
    vs = A.groupby("sic").V.sum()
    share = vs / vs.groupby(lambda s: grp_of[s]).transform("sum")
    cover = A.groupby("sic").size() / (city.city_name.nunique() * city.date.nunique())
    A["grp"] = np.where(A.sic.isin(INPERSON), "INP", "REM")
    G = A.groupby(["city_name", "date", "grp"])["V"].sum().unstack("grp")
    W = A[A.sic.isin(list(SECTORS))].pivot_table(index=["city_name", "date"], columns="sic", values="V", aggfunc="sum")
    W.columns = [f"S_{c}" for c in W.columns]
    city = city.merge(G.reset_index(), on=["city_name", "date"], how="left").merge(W.reset_index(),
                                                                                   on=["city_name", "date"], how="left")
    scols = [f"S_{c}" for c in SECTORS]
    city[["INP", "REM"] + scols] = city[["INP", "REM"] + scols].fillna(0.0)
    del A, G, W
    sk = load_skeleton()
    ev = pd.read_csv(INTERMEDIATE / "events.csv", encoding=ENC, usecols=["city_event_id"]).city_event_id
    nxt = sk.groupby("city_event_id").days_until_next_city_event.first()
    series = ["LOC", "INB", "INP", "REM"] + scols
    days, info = counterfactual(city, series, sk)
    print(f"  counterfactuals fitted for {len(series)} series ({time.time() - t0:.0f} s)", flush=True)
    MARG = {"total": ["LOC", "INB"], "inperson": ["INP"], "remote": ["REM"]} | {c: [f"S_{c}"] for c in SECTORS}
    out = {m: event_metrics_ext(sk, days, city, comp, ev, m) for m, comp in MARG.items()}
    for m, d in out.items():
        d["q4"] = ((d.sup28 == 1) & (d.full_post_28d == 1) & (d.other_post28 == 0)
                   & ~(d.city_event_id.map(nxt) <= 28)).astype(int)
    print(f"  event outcomes built ({time.time() - t0:.0f} s); events per margin: "
          + ", ".join(f"{m} {len(d)}" for m, d in out.items()), flush=True)

    ref = pd.read_parquet(INTERMEDIATE / "event_outcomes.parquet")
    chk = {}
    for m in ("total", "inperson", "remote"):
        x = out[m].merge(ref[ref.margin == m], on="city_event_id", suffixes=("", "_saved"))
        assert len(x) == len(out[m]) == 643, f"{m} event count"
        chk[m] = max(float((x[f"{u}_{h}"] - x[f"{u}_{h}_saved"]).abs().max()) for u in ("ND", "CD", "FD")
                     for h in (0, 3, 7, 14, 21, 28))
        assert chk[m] < 1e-8, f"{m} differs from event_outcomes.parquet by {chk[m]}"

    # paired group frame (in-person minus remote)
    a, b = out["inperson"], out["remote"]
    vcols = [f"{u}_{h}" for u in ("ND", "CD", "FD") for h in HS]
    w = a[["city_event_id", "city_name", FE, "q4"] + vcols].merge(b[["city_event_id"] + vcols], on="city_event_id",
                                                                  suffixes=("_inperson", "_remote"))
    for c in vcols:
        w[f"{c}_diff"] = w[f"{c}_inperson"] - w[f"{c}_remote"]
    frames = {"all": {}, "clean28": {}}
    for mg in ("inperson", "remote", "diff"):
        f_ = w[["city_event_id", "city_name", FE, "q4"] + [f"{c}_{mg}" for c in vcols]].copy()
        f_.columns = ["city_event_id", "city_name", FE, "q4"] + vcols
        frames["all"][mg] = f_
        frames["clean28"][mg] = f_[f_.q4 == 1]
    for m in ["total"] + list(SECTORS):
        frames["all"][m] = out[m]
        frames["clean28"][m] = out[m][out[m].q4 == 1]

    rows = []
    for smp, fd in frames.items():
        for m, d in fd.items():
            units = ("ND", "CD", "FD") if m in ("total", "inperson", "remote", "diff") else ("ND", "CD")
            for u in units:
                for h in HS:
                    col = f"{u}_{h}"
                    if d[col].notna().sum() < 10:
                        continue
                    rows.append(dict(sample=smp, unit=u, margin=m, horizon=h, **mean_test(d, col), **boot_mean(d, col)))
        print(f"  sample {smp}: {len(rows)} rows so far ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(rows)
    lab = {"total": "City total (local + inbound)", "inperson": "In-person sectors",
           "remote": "Remote or large-ticket sectors", "diff": "In-person minus remote"} | SECTORS
    res.insert(3, "margin_label", res.margin.map(lab))
    res.insert(4, "group", res.margin.map(lambda m: grp_of.get(m, {"inperson": "inperson", "remote": "remote"}.get(m, ""))))
    res["value_share_in_group"] = res.margin.map(share)
    res["citDay_coverage"] = res.margin.map(cover)
    res = res.rename(columns={"citDay_coverage": "cityday_coverage"})
    res["unit_label"] = res.unit.map({"ND": "% of a normal day", "CD": "% of cumulative normal spending in the window",
                                      "FD": "% of a normal day, fixed denominator (event days)"})
    res["sample_label"] = res["sample"].map({"all": "all events with support (643; 639 after day 14 for groups)",
                                             "clean28": "clean 28-day post window (T5 Q4 sample)"})

    f4 = pd.read_csv(INTERMEDIATE / "fig4d_sector_cumulative_deficit_daily.csv", encoding=ENC)
    x = res[(res["sample"] == "all") & (res.unit == "ND") & res.margin.isin(["inperson", "remote", "diff"])].merge(
        f4, on=["horizon", "margin"], suffixes=("", "_saved"))
    assert len(x) == 87, "29 horizons x 3 margins expected"
    dB = max(float((x[c] - x[c + "_saved"]).abs().max()) for c in ("est", "crv_ci_low", "crv_ci_high", "boot_ci_low",
                                                                 "boot_ci_high"))
    assert dB < 1e-6 and (x.N == x.N_saved).all(), f"differs from fig4d_sector_cumulative_deficit_daily.csv by {dB}"
    q5 = pd.read_csv(INTERMEDIATE / "T5_Q5_depth_recovery_by_class.csv", encoding=ENC)
    q5 = q5[(q5.margin == "total") & (q5.cls == "all")].set_index("statistic")
    dC = 0.0
    for stat in ("ND_0", "ND_14", "CD_0", "CD_14"):
        u, h = stat.split("_")
        r = res[(res["sample"] == "all") & (res.unit == u) & (res.margin == "total") & (res.horizon == int(h))].iloc[0]
        dC = max(dC, abs(r.est - q5.loc[stat, "est"]), abs(r.crv_ci_low - q5.loc[stat, "ci_low"]),
                 abs(r.crv_ci_high - q5.loc[stat, "ci_high"]))
    q4l = pd.read_csv(INTERMEDIATE / "T5_Q4_levels_clean28.csv", encoding=ENC)
    q4l = q4l[(q4l.margin == "total") & (q4l.cls == "all")]
    for r0 in q4l.itertuples():
        r = res[(res["sample"] == "clean28") & (res.unit == r0.unit) & (res.margin == "total")
                & (res.horizon == r0.horizon)].iloc[0]
        dC = max(dC, abs(r.est - r0.est), abs(r.crv_ci_low - r0.ci_low), abs(r.crv_ci_high - r0.ci_high))
        assert int(r.N) == int(r0.N), "clean-28 total N"
    assert dC < 1e-6, f"totals differ from T5_Q5_depth_recovery_by_class.csv / T5_Q4_levels_clean28.csv by {dC}"
    q4b = pd.read_csv(INTERMEDIATE / "T5_Q4b_sectors.csv", encoding=ENC)
    q4b = q4b[q4b.cls == "all"]
    dD = 0.0
    for r0 in q4b.itertuples():
        smp = "clean28" if r0.spec.startswith("clean-28") else "all"
        r = res[(res["sample"] == smp) & (res.unit == r0.unit) & (res.margin == r0.margin)
                & (res.horizon == r0.horizon)].iloc[0]
        dD = max(dD, abs(r.est - r0.est), abs(r.crv_ci_low - r0.ci_low), abs(r.crv_ci_high - r0.ci_high))
        assert int(r.N) == int(r0.N), f"N differs ({smp} {r0.unit} {r0.margin} {r0.horizon})"
    assert dD < 1e-6, f"groups differ from T5_Q4b_sectors.csv by {dD}"

    res.to_csv(INTERMEDIATE / "SI3_sector_cumdeficit_daily.csv", index=False, encoding=ENC)
    print(f"  wrote SI3_sector_cumdeficit_daily.csv ({len(res)} rows); "
          f"max differences from saved results: events {max(chk.values()):.1e}, groups {dB:.1e}, totals {dC:.1e}, "
          f"sector groups {dD:.1e}; {time.time() - t0:.0f} s")


def main():
    t0 = time.time()
    print("== cluster_inside_outside", flush=True)
    cluster_inside_outside()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== sector_cumulative_deficit", flush=True)
    sector_cumulative_deficit()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
