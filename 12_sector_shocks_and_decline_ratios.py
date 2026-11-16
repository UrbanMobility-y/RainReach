"""Sector-group event shocks and inbound composition, decline ratios of inbound and outbound to local spending, the
dose-response of spending to daily rainfall, and loss depth by rainfall-system extent (Supplementary results
supporting Fig. 1). Writes SI1_sector_group_event_shocks.csv, SI1_inbound_sector_composition.csv,
SI1_decline_ratios.csv, SI1_dose_response.csv and SI1_extent_slopes.csv.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import pyfixest as pf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*singleton fixed effect.*")

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
DATASET = ROOT / "Dataset"
INTERMEDIATE = ROOT / "Intermediate"
# vendor files: daily city-by-sector activity index and its data dictionary (sector codes)
SECTOR_FILE = DATASET / "\u57ce\u5e02\u5206\u884c\u4e1a\u6d3b\u529b\u6307\u6570_\u65e5\u5ea6.csv"
DICTIONARY = DATASET / "\u57ce\u5e02\u6d3b\u529b\u6307\u6570\u6570\u636e\u8bf4\u660e\u9644\u4ef6.xlsx"
DAILY_PANEL = INTERMEDIATE / "unionpay_city_daily_2017_2019.csv"
CF_PANEL = INTERMEDIATE / "counterfactual_panel.csv"
EVENTS = INTERMEDIATE / "events.csv"
ENC = "utf-8-sig"
PRE_DAYS, MIN_CLEAN_PRE = 14, 7


def need(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file missing: {path}")
    return path


GROUPS = {"in-person": {"F52", "H", "H61", "H62", "R86", "R87", "R88", "R89", "O79", "O80"},
          "remote/large-ticket": {"F51", "K70", "C29", "C36", "Y"}}
EVENT_OUTCOMES = INTERMEDIATE / "event_outcomes.parquet"
SIC_EN = {"B07": "Oil and gas extraction", "C14": "Food manufacturing", "C15": "Beverages", "C16": "Tobacco",
          "C17": "Textiles", "C18": "Apparel", "C19": "Leather and footwear", "C20": "Wood products",
          "C21": "Furniture", "C24": "Stationery, sports goods", "C25": "Petroleum processing",
          "C27": "Pharmaceuticals", "C29": "Automobiles (C29)", "C30": "Non-metallic minerals",
          "C33": "Metal products", "C36": "Automobiles", "C37": "Other transport equipment",
          "C38": "Electrical machinery", "C39": "Computers and electronics", "C40": "Instruments",
          "D": "Utilities", "F51": "Wholesale", "F52": "Retail", "G53": "Rail transport", "G54": "Road transport",
          "G55": "Water transport", "G56": "Air transport", "G58": "Transport agency", "G59": "Warehousing",
          "G60": "Postal services", "H": "Accommodation, catering, entertainment (combined code)",
          "H61": "Accommodation", "H62": "Catering", "I63": "Telecommunications", "I64": "Internet services",
          "I65": "Software and IT", "J66": "Banking", "J67": "Capital markets", "J68": "Insurers",
          "J69": "Other finance", "K70": "Real estate", "L71": "Leasing", "L72": "Business services",
          "M74": "Professional services", "N78": "Public facilities", "O79": "Resident services",
          "O80": "Repair services", "P82": "Education", "Q83": "Health care", "Q84": "Social work",
          "R83": "Health care (R83)", "R85": "News and publishing", "R86": "Film and broadcasting",
          "R87": "Culture and arts", "R88": "Sports", "R89": "Entertainment", "X": "Other",
          "Y": "Administrative fees"}
DIMS = ["city_moy", "datef"]


def read_sector_file() -> pd.DataFrame:
    """Columns mchnt_city, sic, date, T, I, O for 2017-2019 (inverted index)."""
    import pyarrow.csv as pacsv
    t0 = time.time()
    tab = pacsv.read_csv(need(SECTOR_FILE), read_options=pacsv.ReadOptions(encoding="utf8"),
                         convert_options=pacsv.ConvertOptions(
                             include_columns=["mchnt_city", "sic", "date", "cval", "cval_in", "cval_out"],
                             column_types={"mchnt_city": "string", "sic": "string", "date": "string"}))
    d = tab.to_pandas()
    del tab
    n_raw = len(d)
    ud = pd.Series(d.date.unique())
    dmap = dict(zip(ud, pd.to_datetime(ud, format="%d%b%Y")))
    d["date"] = d.date.map(dmap)
    d = d.loc[d.date <= pd.Timestamp("2019-12-31")].reset_index(drop=True)
    for c, n in (("cval", "T"), ("cval_in", "I"), ("cval_out", "O")):
        d[n] = np.expm1(pd.to_numeric(d[c], errors="coerce").astype("float64") / 10.0)
    d = d.drop(columns=["cval", "cval_in", "cval_out"])
    print(f"sector file: {n_raw:,} rows read, {len(d):,} rows 2017-2019, {d.mchnt_city.nunique()} cities, "
          f"{d.sic.nunique()} codes ({time.time() - t0:.0f} s)", flush=True)
    return d, n_raw


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
    for _ in range(maxit):
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
    return log_fe


def counterfactual(train: pd.DataFrame, outcome: str, keys: pd.DataFrame) -> np.ndarray:
    sub = train.loc[drop_singletons(train, DIMS)]
    cats, codes, nlev = {}, [], []
    for d in DIMS:
        cat = pd.Categorical(sub[d])
        cats[d] = cat.categories
        codes.append(cat.codes.astype(np.int64))
        nlev.append(len(cat.categories))
    log_fe = fit_ipf(sub[outcome].to_numpy(), codes, nlev)
    total = np.zeros(len(keys))
    ok = np.ones(len(keys), bool)
    for j, d in enumerate(DIMS):
        v = keys[d].map(pd.Series(log_fe[j], index=cats[d]))
        ok &= v.notna().to_numpy()
        total += v.fillna(0.0).to_numpy()
    return np.where(ok, np.exp(total), np.nan)


def shocks(frame: pd.DataFrame, series: pd.DataFrame, cf: pd.DataFrame, label: dict) -> pd.DataFrame:
    """Event shocks for local / inbound / outbound / place-based / resident-based of one series."""
    f = frame.merge(series, on=["city_name", "date"], how="left").merge(cf, on=["city_name", "date"], how="left")
    comps = {"local": (["L"], ["L0"]), "inbound": (["I"], ["I0"]), "outbound": (["O"], ["O0"]),
             "place-based": (["L", "I"], ["L0", "I0"]), "resident-based": (["L", "O"], ["L0", "O0"])}
    pre_mask = f.k.between(-PRE_DAYS, -1) & (f.is_other_city_event_exposure == 0)
    exp_mask = f.is_focal_exposure_day == 1
    out = []
    w_pre = (f.loc[pre_mask, "L"] + f.loc[pre_mask, "I"]).groupby(f.loc[pre_mask, "city_event_id"]).mean()
    for m, (ys, y0s) in comps.items():
        a = 100.0 * (f[ys].sum(axis=1, min_count=len(ys)) / f[y0s].sum(axis=1, min_count=len(y0s)) - 1.0)
        pre = a[pre_mask].groupby(f.loc[pre_mask, "city_event_id"]).mean()
        ex = a[exp_mask].groupby(f.loc[exp_mask, "city_event_id"]).mean()
        r = pd.DataFrame({"CleanPreMean": pre, "ExposureMean": ex})
        r["CurrentShock"] = r.ExposureMean - r.CleanPreMean
        r["margin"] = m
        r["w_pre"] = w_pre
        out.append(r.rename_axis("city_event_id").reset_index())
    o = pd.concat(out, ignore_index=True)
    for k, v in label.items():
        o[k] = v
    return o


def sector_group_shocks():
    t0 = time.time()
    ev = pd.read_csv(need(EVENTS), encoding=ENC, usecols=["city_event_id", "city_name", "meteorological_event_id",
                                                          "CurrentShock_local", "CurrentShock_inbound",
                                                          "CurrentShock_outbound", "CurrentShock_merchant",
                                                          "CurrentShock_resident"])
    ev["city_name"] = ev.city_name.astype(str).str.strip()
    ids = set(ev.city_event_id)
    cfp = pd.read_csv(need(CF_PANEL), encoding=ENC, dtype={"city_id": str},
                      usecols=["city_event_id", "city_name", "date", "event_time_from_start", "is_focal_exposure_day",
                               "is_other_city_event_exposure", "cf_complete_end14", "normal_local_spending",
                               "normal_inbound_spending", "normal_outbound_spending"])
    cfp["city_name"] = cfp.city_name.astype(str).str.strip()
    cfp["date"] = pd.to_datetime(cfp.date)
    keys = cfp[["city_name", "date"]].drop_duplicates().reset_index(drop=True)          # hold-out = all window days
    keys["datef"] = keys.date.dt.strftime("%Y-%m-%d")
    keys["city_moy"] = keys.city_name + "|" + keys.date.dt.month.astype(str)
    frame = cfp.loc[(cfp.cf_complete_end14 == 1) & cfp.city_event_id.isin(ids),
                    ["city_event_id", "city_name", "date", "event_time_from_start", "is_focal_exposure_day",
                     "is_other_city_event_exposure"]].rename(columns={"event_time_from_start": "k"})
    assert frame.city_event_id.nunique() == 643

    up = pd.read_csv(need(DAILY_PANEL), encoding=ENC, dtype={"city_id": str})
    up["city_name"] = up.city_name.astype(str).str.strip()
    up["date"] = pd.to_datetime(up.date)
    up["city_id"] = up.city_id.str.zfill(4)
    grid = up[["city_id", "city_name", "date"]].copy()
    hold = set(zip(keys.city_name, keys.date))
    grid["train"] = [(c, d) not in hold for c, d in zip(grid.city_name, grid.date)]
    grid["datef"] = grid.date.dt.strftime("%Y-%m-%d")
    grid["city_moy"] = grid.city_name + "|" + grid.date.dt.month.astype(str)
    print(f"city panel {len(grid):,} city-days, training {int(grid.train.sum()):,}, window keys {len(keys):,}")

    def series_cf(s: pd.DataFrame, tag: str):
        """s: city_name, date, L, I, O on the full grid -> counterfactual on window keys."""
        g = grid.merge(s, on=["city_name", "date"], how="left")
        g[["L", "I", "O"]] = g[["L", "I", "O"]].fillna(0.0)
        tr = g.loc[g.train]
        cf = keys[["city_name", "date"]].copy()
        for m in ("L", "I", "O"):
            cf[f"{m}0"] = counterfactual(tr, m, keys)
        print(f"   {tag:<32} counterfactual fitted ({time.time() - t0:.0f} s)", flush=True)
        return g[["city_name", "date", "L", "I", "O"]], cf

    rows, checks = [], []
    # --- check 1: city panel series through the same code
    city = up[["city_name", "date"]].assign(L=up.local_spending, I=up.inbound_spending, O=up.outbound_spending)
    s_city, cf_city = series_cf(city, "city panel (check)")
    ref = cfp[["city_name", "date", "normal_local_spending", "normal_inbound_spending",
               "normal_outbound_spending"]].drop_duplicates(["city_name", "date"]).merge(cf_city, on=["city_name", "date"])
    ok = ref.normal_local_spending.notna()
    dlog = max(float(np.nanmax(np.abs(np.log(ref.loc[ok, f"normal_{a}_spending"] / ref.loc[ok, f"{b}0"]))))
               for a, b in (("local", "L"), ("inbound", "I"), ("outbound", "O")))
    checks.append(("city panel Y0 vs counterfactual_panel.csv, max |dlog Y0|", dlog))
    sc = shocks(frame, s_city, cf_city, {"grouping": "city panel", "group": "all sectors"})
    w = sc.pivot(index="city_event_id", columns="margin", values="CurrentShock").join(ev.set_index("city_event_id"))
    sc = sc.assign(grouping="city panel", group="all sectors")
    for m, col in (("local", "CurrentShock_local"), ("inbound", "CurrentShock_inbound"),
                   ("outbound", "CurrentShock_outbound"), ("place-based", "CurrentShock_merchant"),
                   ("resident-based", "CurrentShock_resident")):
        checks.append((f"city panel CurrentShock {m} vs events.csv, max |diff| (pp)",
                       float((w[m] - w[col]).abs().max())))
    rows.append(sc)

    # --- sector file
    d, n_raw = read_sector_file()
    cid = up[["city_id", "city_name"]].drop_duplicates().set_index("city_id").city_name
    d["city_name"] = d.mchnt_city.str.zfill(4).map(cid)
    if d.city_name.isna().any():
        raise SystemExit(f"{d.loc[d.city_name.isna(), 'mchnt_city'].nunique()} sector-file city codes not in panel")

    # composition (inbound and local by sector, 2017-2019 pooled, per city)
    agg = d.groupby(["city_name", "sic"])[["T", "I"]].sum(min_count=1).fillna(0.0)
    agg["L"] = agg["T"] - agg["I"]
    comp = []
    for m in ("I", "L"):
        x = agg[m].unstack(fill_value=0.0)
        sh = x.div(x.sum(axis=1), axis=0)
        comp.append(pd.DataFrame({"sic": sh.columns, f"{m}_median": sh.median().values, f"{m}_p25": sh.quantile(.25).values,
                                  f"{m}_p75": sh.quantile(.75).values, f"{m}_mean": sh.mean().values,
                                  f"{m}_n_cities": sh.notna().sum().values}))
    comp = comp[0].merge(comp[1], on="sic")
    names = pd.read_excel(need(DICTIONARY), sheet_name=None)            # vendor list of the 58 codes (check only)
    sheet = next(v for v in names.values() if v.shape[1] == 2 and v.shape[0] == 58)
    missing = set(sheet.iloc[:, 0].astype(str).str.strip()) - set(SIC_EN)
    assert not missing, f"no English name for {missing}"
    comp["name_en"] = comp.sic.map(SIC_EN)
    comp["group"] = np.where(comp.sic.isin(GROUPS["in-person"]), "in-person",
                             np.where(comp.sic.isin(GROUPS["remote/large-ticket"]), "remote/large-ticket", "other"))
    comp = comp.rename(columns=lambda c: c.replace("I_", "inbound_share_").replace("L_", "local_share_"))
    comp = comp.sort_values("inbound_share_median", ascending=False).reset_index(drop=True)
    ann = INTERMEDIATE / "sector_annual_sums.parquet"
    if ann.exists():
        sa = pd.read_parquet(ann)
        sa = sa.loc[sa.year.between(2017, 2019)]
        g = sa.groupby(["mchnt_city", "sic"]).I_sum.sum().unstack(fill_value=0.0)
        med = g.div(g.sum(axis=1), axis=0).median()
        cmp_ = comp.set_index("sic").inbound_share_median.reindex(med.index)
        checks.append(("inbound composition medians vs sector_annual_sums.parquet, max |diff| (share)",
                       float((cmp_ - med).abs().max())))
    comp.to_csv(INTERMEDIATE / "SI1_inbound_sector_composition.csv", index=False, encoding=ENC)

    # group series
    d[["T", "I", "O"]] = d[["T", "I", "O"]].fillna(0.0)
    g = np.where(d.sic.isin(GROUPS["in-person"]), "in-person",
                 np.where(d.sic.isin(GROUPS["remote/large-ticket"]), "remote/large-ticket", ""))
    by = d.assign(grp=g).loc[g != ""].groupby(["city_name", "date", "grp"])[["T", "I", "O"]].sum().reset_index()
    allx = d.groupby(["city_name", "date"])[["T", "I", "O"]].sum().reset_index()      # all 58 codes (check only)
    del d
    specs = [("sector", gname, by.loc[by.grp == gname, ["city_name", "date", "T", "I", "O"]]) for gname in GROUPS]
    specs.append(("check", "all sector codes", allx))
    for grouping, gname, s_ in specs:
        s_ = s_.assign(L=s_["T"] - s_["I"])[["city_name", "date", "L", "I", "O"]]
        s_g, cf_g = series_cf(s_, f"{grouping} / {gname}")
        rows.append(shocks(frame, s_g, cf_g, {"grouping": grouping, "group": gname}))
    out = pd.concat(rows, ignore_index=True)
    out = out[["grouping", "group", "margin", "city_event_id", "CleanPreMean", "ExposureMean", "CurrentShock", "w_pre"]]
    nonfinite = int((~np.isfinite(out.CurrentShock)).sum())
    checks.append(("non-finite event shocks (all groups, margins)", nonfinite))
    groups_ref = pd.read_csv(need(INTERMEDIATE / "x1_T3_sector_groups.csv"), encoding=ENC)
    a = out.loc[out.grouping == "check"].pivot(index="city_event_id", columns="margin", values="CurrentShock")
    gall = float((a["inbound"] - a["local"]).mean())
    ref_all = float(groups_ref.loc[(groups_ref.label == "ALL full") & (groups_ref.outcome == "G_in") & (groups_ref.weight == "equal"), "est"].iloc[0])
    checks.append((f"sum over all sector codes, G_in (equal) {gall:.4f} vs x1_T3_sector_groups.csv 'ALL full' {ref_all:.4f}, |diff| (pp)",
                   abs(gall - ref_all)))
    outcomes = pd.read_parquet(need(EVENT_OUTCOMES))
    for grp, m5 in (("in-person", "inperson"), ("remote/large-ticket", "remote")):
        mine = out.loc[(out.group == grp) & (out.margin == "place-based")].set_index("city_event_id").CurrentShock
        ref = outcomes.loc[outcomes.margin == m5].set_index("city_event_id").CurrentShock.reindex(mine.index)
        checks.append((f"{grp} place-based change vs event_outcomes.parquet '{m5}' (one total series): correlation", float(mine.corr(ref))))
        checks.append((f"{grp} place-based change vs event_outcomes.parquet '{m5}': mean {mine.mean():.3f} vs {ref.mean():.3f}, |diff| (pp)",
                       abs(mine.mean() - ref.mean())))
    out = out.loc[out.grouping != "check"].reset_index(drop=True)
    out.to_csv(INTERMEDIATE / "SI1_sector_group_event_shocks.csv", index=False, encoding=ENC, float_format="%.9g")

    for k, v in checks:
        print(f"CHECK {k}: {v:.3g}")
    print(f"sector file rows read: {n_raw:,}; wrote SI1_sector_group_event_shocks.csv ({len(out):,} rows) and "
          f"SI1_inbound_sector_composition.csv ({time.time() - t0:.0f} s)")


FE = "meteorological_event_id"


def read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(need(path), encoding=ENC, low_memory=False, **kw)


B, SEED = 1999, 2026


def codes(c1, c2):
    a = pd.factorize(np.asarray(c1))[0]
    b = pd.factorize(np.asarray(c2))[0]
    ab = pd.factorize(pd.Series(a).astype(str) + "_" + pd.Series(b).astype(str))[0]
    return a, b, ab


def ratio_delta(yn, yd, w, city, sysid):
    """R = wmean(yn) / wmean(yd); delta method, two-way CRV1 (city x system), t(G - 1)."""
    yn, yd = np.asarray(yn, float), np.asarray(yd, float)
    w = np.ones_like(yn) if w is None else np.asarray(w, float)
    a, b, ab = codes(city, sysid)
    G = min(a.max() + 1, b.max() + 1)
    sw = w.sum()
    mn, md = (w * yn).sum() / sw, (w * yd).sum() / sw
    R = mn / md
    s = w * ((yn - mn) - R * (yd - md)) / md
    meat = sum(sg * (np.bincount(c, weights=s) ** 2).sum() for c, sg in ((a, 1), (b, 1), (ab, -1)))
    se = np.sqrt(max(G / (G - 1) * meat / sw ** 2, 0))
    t = stats.t.ppf(.975, G - 1)
    return R, se, R - t * se, R + t * se, mn, md


def boot_draws(sysid):
    s = pd.factorize(np.asarray(sysid))[0]
    ng = s.max() + 1
    groups = [np.where(s == g)[0] for g in range(ng)]
    rng = np.random.default_rng(SEED)
    return [np.concatenate([groups[g] for g in rng.integers(0, ng, ng)]) for _ in range(B)]


def decline_ratios():
    ev = read(EVENTS, usecols=["city_event_id", "city_name", FE])
    ev["city_name"] = ev.city_name.astype(str).str.strip()
    sh = read(INTERMEDIATE / "SI1_sector_group_event_shocks.csv")
    up = read(DAILY_PANEL, usecols=["city_name", "local_spending", "inbound_spending"])
    up["city_name"] = up.city_name.astype(str).str.strip()
    size = (up.local_spending + up.inbound_spending).groupby(up.city_name).mean()
    sz = size.loc[sorted(ev.city_name.unique())]
    ev["size"] = ev.city_name.map(sz)
    q1, q2 = np.quantile(sz.to_numpy(), [1 / 3, 2 / 3])          # a city exactly on an edge joins the outer group
    terc = pd.Series(np.where(sz <= q1, "small", np.where(sz >= q2, "large", "middle")), index=sz.index)
    ev["tercile"] = ev.city_name.map(terc)
    ev["quintile"] = ev.city_name.map(pd.qcut(sz, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))
    checks = []
    frame_weights = read(INTERMEDIATE / "x1_event_frame_weights.csv")
    assert (frame_weights.city_event_id.values == ev.city_event_id.values).all()
    checks.append(("city size vs x1_event_frame_weights.csv, max |rel diff|",
                   float((ev["size"].values / frame_weights["size"].values - 1).__abs__().max())))
    tmap = {"T1_small": "small", "T2_mid": "middle", "T3_large": "large"}
    checks.append(("size tercile identical to x1_event_frame_weights.csv (share of events)",
                   float((ev.tercile.astype(str).values == frame_weights.terc.map(tmap).values).mean())))
    for t_ in ("small", "middle", "large"):
        checks.append((f"cities in size tercile '{t_}'", float((terc == t_).sum())))

    series = {("city panel", "all sectors"): "city", ("sector", "in-person"): "in-person",
              ("sector", "remote/large-ticket"): "remote/large-ticket"}
    wide = {}
    for (grouping, group), tag in series.items():
        q = sh.loc[(sh.grouping == grouping) & (sh.group == group)]
        p = q.pivot(index="city_event_id", columns="margin", values="CurrentShock").reindex(ev.city_event_id)
        w = q.loc[q.margin == "local"].set_index("city_event_id").w_pre.reindex(ev.city_event_id)
        assert p.notna().all().all() and w.notna().all(), tag
        wide[tag] = (grouping, group, p, w.to_numpy())
    checks.append(("city-panel w_P vs x1_event_frame_weights.csv, max |rel diff|",
                   float(np.abs(wide["city"][3] / frame_weights.w_P.values - 1).max())))

    subsets = {"all events": np.ones(len(ev), bool)}
    for col, levels in (("tercile", ["small", "middle", "large"]), ("quintile", ["Q1", "Q2", "Q3", "Q4", "Q5"])):
        for lev in levels:
            subsets[f"size {col}: {lev}"] = (ev[col].astype(str) == lev).to_numpy()
    rows, draws_store = [], {}
    for sub, mask in subsets.items():
        e = ev.loc[mask]
        idx = boot_draws(e[FE].to_numpy())
        tags = list(series.values())
        for tag in tags:
            grouping, group, p, w = wide[tag]
            p = p.loc[mask]
            ww = w[mask]
            for num in ("inbound", "outbound"):
                for wname, wt in (("equal", None), ("spending-weighted (w_P)", ww)):
                    yn, yd = p[num].to_numpy(), p["local"].to_numpy()
                    R, se, dlo, dhi, mn, md = ratio_delta(yn, yd, wt, e.city_name, e[FE])
                    wv = np.ones_like(yn) if wt is None else wt
                    bs = np.array([(wv[i] * yn[i]).sum() / (wv[i] * yd[i]).sum() for i in idx])
                    draws_store[(sub, tag, num, wname)] = bs
                    rows.append({"subset": sub, "series": tag, "grouping": grouping, "group": group,
                                 "ratio": f"{num}/local", "weight": wname, "DR": R,
                                 "boot_ci_low": float(np.percentile(bs, 2.5)), "boot_ci_high": float(np.percentile(bs, 97.5)),
                                 "delta_se": se, "delta_ci_low": dlo, "delta_ci_high": dhi,
                                 "mean_shock_num": mn, "mean_shock_local": md, "N": int(mask.sum()),
                                 "n_city": e.city_name.nunique(), "n_sys": e[FE].nunique(),
                                 "size_min": float(e["size"].min()), "size_max": float(e["size"].max()),
                                 "B": B, "seed": SEED})
    out = pd.DataFrame(rows)
    diffs = []
    for sub in subsets:
        for a, b in (("in-person", "remote/large-ticket"),):
            for wname in ("equal", "spending-weighted (w_P)"):
                if (sub, a, "inbound", wname) not in draws_store:
                    continue
                da = draws_store[(sub, a, "inbound", wname)] - draws_store[(sub, b, "inbound", wname)]
                ra = out.loc[(out.subset == sub) & (out.series == a) & (out.ratio == "inbound/local") & (out.weight == wname), "DR"].iloc[0]
                rb = out.loc[(out.subset == sub) & (out.series == b) & (out.ratio == "inbound/local") & (out.weight == wname), "DR"].iloc[0]
                diffs.append({"subset": sub, "series": f"{a} minus {b}", "grouping": "sector",
                              "group": "difference", "ratio": "inbound/local", "weight": wname, "DR": ra - rb,
                              "boot_ci_low": float(np.percentile(da, 2.5)), "boot_ci_high": float(np.percentile(da, 97.5)),
                              "N": int(subsets[sub].sum()), "B": B, "seed": SEED})
    out = pd.concat([out, pd.DataFrame(diffs)], ignore_index=True)
    ref = read(INTERMEDIATE / "x1_T1_diversification_ratio.csv")
    for rname, series_tag, num, wname, wref in (("city inbound/local", "city", "inbound", "equal", "equal"),
                                                ("city inbound/local", "city", "inbound", "spending-weighted (w_P)", "w_P"),
                                                ("city outbound/local", "city", "outbound", "equal", "equal"),
                                                ("city outbound/local", "city", "outbound", "spending-weighted (w_P)", "w_P"),
                                                ):
        r0 = ref.loc[(ref.ratio == rname) & (ref.weight == wref)].iloc[0]
        r1 = out.loc[(out.subset == "all events") & (out.series == series_tag) & (out.ratio == f"{num}/local")
                     & (out.weight == wname)].iloc[0]
        dd = max(abs(r1.DR - r0.DR), abs(r1.boot_ci_low - r0.boot_lo), abs(r1.boot_ci_high - r0.boot_hi),
                 abs(r1.delta_ci_low - r0.delta_lo), abs(r1.delta_ci_high - r0.delta_hi))
        checks.append((f"{rname} ({wref}) {r1.DR:.3f} [{r1.boot_ci_low:.3f}, {r1.boot_ci_high:.3f}] vs x1_T1_diversification_ratio.csv "
                       f"{r0.DR:.3f} [{r0.boot_lo:.3f}, {r0.boot_hi:.3f}], max |diff|", dd))
    out.to_csv(INTERMEDIATE / "SI1_decline_ratios.csv", index=False, encoding=ENC)
    for k, v in checks:
        print(f"CHECK {k}: {v:.3g}")
    show = out.loc[out.weight == "equal", ["subset", "series", "ratio", "DR", "boot_ci_low", "boot_ci_high", "N"]]
    print(show.round(3).to_string(index=False))
    print(f"wrote SI1_decline_ratios.csv ({len(out)} rows)")


WEATHER_DIR = DATASET / "city daily weather"
EDGES, BINS = [-1, 1, 10, 25, 50, 1e4], ["0-1", "1-10", "10-25", "25-50", "50+"]
MARGIN_COLS = {"local": ["local_spending"], "inbound": ["inbound_spending"], "outbound": ["outbound_spending"],
               "place-based": ["local_spending", "inbound_spending"]}


def dose_response():
    t0 = time.time()
    daily = read(DAILY_PANEL, dtype={"city_id": str})
    daily["city_name"] = daily.city_name.astype(str).str.strip()
    daily["date"] = pd.to_datetime(daily.date)
    rain = pd.concat([read(WEATHER_DIR / f"city_daily_weather_{y}.csv", usecols=["city_name", "date", "rain_mm"])
                      for y in (2017, 2018, 2019)])
    rain["city_name"] = rain.city_name.astype(str).str.strip()
    rain["date"] = pd.to_datetime(rain.date)
    d = daily.merge(rain, on=["city_name", "date"]).sort_values(["city_name", "date"])
    d["bin"] = pd.cut(d.rain_mm, EDGES, labels=BINS).astype(str)
    d["lagbin"] = pd.cut(d.groupby("city_name").rain_mm.shift(1), EDGES, labels=BINS).astype(str)
    d = d.loc[d.date.dt.year.isin([2018, 2019])].copy()
    d["cym"] = d.city_name + "_" + d.date.dt.strftime("%Y-%m")
    d["cmoy"] = d.city_name + "_" + d.date.dt.month.astype(str)
    for m, cols in MARGIN_COLS.items():
        d[f"y_{m.replace('-', '_')}"] = d[cols].sum(axis=1)
    n_days, n_cities = len(d), d.city_name.nunique()
    print(f"dose-response panel: {n_days:,} city-days, {n_cities} cities ({time.time() - t0:.0f} s)")
    counts = d.bin.value_counts()
    rows = []
    for spec in ("ols_cym", "ppml_cym", "ppml_cmoy"):
        for m in MARGIN_COLS:
            y = f"y_{m.replace('-', '_')}"
            if spec == "ols_cym":
                mod = pf.feols("ly ~ i(bin, ref='0-1') + i(lagbin, ref='0-1') | cym + date",
                               data=d.assign(ly=np.log(d[y])), vcov={"CRV1": "city_name"})
            else:
                fe = spec.split("_")[1]
                mod = pf.fepois(f"{y} ~ i(bin, ref='0-1') + i(lagbin, ref='0-1') | {fe} + date", data=d,
                                vcov={"CRV1": "city_name"})
            t = mod.tidy()
            for b in BINS[1:]:
                for lag, key in (("same day", f"bin::{b}"), ("previous day", f"lagbin::{b}")):
                    r = t.loc[key]
                    rows.append({"spec": spec, "margin": m, "lag": lag, "bin": b, "coef": float(r["Estimate"]),
                                 "se": float(r["Std. Error"]), "pct": 100 * np.expm1(r["Estimate"]),
                                 "ci_low": 100 * np.expm1(r["2.5%"]), "ci_high": 100 * np.expm1(r["97.5%"]),
                                 "p_value": float(r["Pr(>|t|)"]), "N": int(mod._N),
                                 "n_city_days_in_bin": int(counts.get(b, 0))})
            print(f"   {spec:<10} {m:<12} >50 mm {100 * np.expm1(t.loc['bin::50+', 'Estimate']):+.2f}% "
                  f"(N {mod._N:,}; {time.time() - t0:.0f} s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(INTERMEDIATE / "SI1_dose_response.csv", index=False, encoding=ENC)
    print(f"wrote SI1_dose_response.csv ({len(out)} rows; {time.time() - t0:.0f} s)")


CATALOGUE = INTERMEDIATE / "extreme_rainfall_events_2017_2019.csv"


def fit(data, formula, vcov):
    import pyfixest as pf
    return pf.feols(formula, data=data, vcov=vcov)


def dof_of(model) -> int:
    groups = np.atleast_1d(np.asarray(getattr(model, "_G", []) or [], dtype=float))
    return int(groups.min()) - 1 if groups.size else int(model._N - len(model._coefnames))


def coefficient(model, term: str = "Intercept") -> dict:
    row = model.tidy().loc[term]
    return {"estimate": float(row["Estimate"]), "se": float(row["Std. Error"]), "ci_low": float(row["2.5%"]),
            "ci_high": float(row["97.5%"]), "p_value": float(row["Pr(>|t|)"]), "dof": dof_of(model), "N": int(model._N)}


CONTROLS = {"none": [], "ERA5-Land": ["z_peak_rain_mm", "z_cumulative_rain_mm"], "CPC": ["z_cpc_max"],
            "CHIRPS": ["z_chirps_max"],
            "all (Fig. 1d)": ["z_peak_rain_mm", "z_cumulative_rain_mm", "z_cpc_max", "z_chirps_max"]}
OUTCOMES = {"place-based": "CurrentShock_merchant", "local": "CurrentShock_local"}


def extent_slopes():
    ev = read(EVENTS)
    ev["city_name"] = ev.city_name.astype(str).str.strip()
    cat = read(CATALOGUE, usecols=["city_event_id", "affected_city_count"])
    fl = read(INTERMEDIATE / "event_flags.csv", usecols=["city_event_id", "cpc_max", "chirps_max"])
    d = ev.merge(cat, on="city_event_id", how="left", validate="1:1").merge(fl, on="city_event_id", how="left",
                                                                           validate="1:1")
    assert len(d) == 643 and d[["affected_city_count", "cpc_max", "chirps_max"]].notna().all().all()
    d["ym"] = pd.to_datetime(d.city_event_start).dt.strftime("%Y-%m")
    d["lext"] = np.log(d.affected_city_count)
    for c in ("peak_rain_mm", "cumulative_rain_mm", "cpc_max", "chirps_max"):
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std()
    rows = []
    for oname, y in OUTCOMES.items():
        for cname, cs in CONTROLS.items():
            formula = f"{y} ~ lext" + "".join(f" + {c}" for c in cs) + " | city_name + ym"
            for clus, vc in (("rainfall system", {"CRV1": FE}), ("two-way city and rainfall system",
                                                                 {"CRV1": f"city_name + {FE}"})):
                c = coefficient(fit(d, formula, vc), "lext")
                rows.append({"outcome": oname, "controls": cname, "cluster": clus, **c, "formula": formula})
    out = pd.DataFrame(rows)
    ref = read(INTERMEDIATE / "fig1d_extent_slope.csv")
    checks = []
    for clus_ref, clus in (("system", "rainfall system"), ("two-way city and system", "two-way city and rainfall system")):
        r0 = ref.loc[ref.cluster == clus_ref].iloc[0]
        r1 = out.loc[(out.outcome == "place-based") & (out.controls == "all (Fig. 1d)") & (out.cluster == clus)].iloc[0]
        checks.append((f"all controls, {clus}: {r1.estimate:.3f} [{r1.ci_low:.3f}, {r1.ci_high:.3f}] N {r1.N} vs "
                       f"fig1d_extent_slope.csv {r0.estimate:.3f} [{r0.ci_low:.3f}, {r0.ci_high:.3f}] N {r0.N}, max |diff|",
                       max(abs(r1.estimate - r0.estimate), abs(r1.ci_low - r0.ci_low), abs(r1.ci_high - r0.ci_high),
                           abs(r1.N - r0.N))))
    out.to_csv(INTERMEDIATE / "SI1_extent_slopes.csv", index=False, encoding=ENC)
    for k, v in checks:
        print(f"CHECK {k}: {v:.3g}")
    print(out.drop(columns="formula").round(3).to_string(index=False))


def main():
    t0 = time.time()
    print("== sector_group_shocks", flush=True)
    sector_group_shocks()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== decline_ratios", flush=True)
    decline_ratios()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== dose_response", flush=True)
    dose_response()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== extent_slopes", flush=True)
    extent_slopes()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
