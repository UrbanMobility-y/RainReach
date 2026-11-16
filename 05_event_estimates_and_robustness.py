"""Event-level estimates of the local, inbound and outbound spending response to extreme rainfall with specification
robustness, the decomposition of the estimand, the Figure 1 illustration day and the gradient with rainfall-system
extent. The first step extracts the sector-level spending series used by the later scripts."""
import os
import time
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import pyfixest as pf
import shapely
from scipy import stats
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = r"E:\Extreme Rainfall and UnionPay Spending"
DATASET = os.path.join(ROOT, "Dataset")
EXTERNAL = os.path.join(ROOT, "External data")
INTERMEDIATE = os.path.join(ROOT, "Intermediate")


def sector_extracts():
    """Extract sector-level spending from the licensed city-by-sector daily file: annual sums by city, year and sector
    (sector_annual_sums.parquet), daily series for the in-person and remote sectors used in the incidence and recovery
    analyses (sector_sic_daily.parquet), and daily totals by city for three sector groups: presence-based, remote and other
    (sector_group_daily.parquet). Index values are inverted with q(x) = exp(x/10) - 1."""
    # City-by-sector daily index of the licensed data delivery (file name as delivered).
    SECTOR_DAILY = os.path.join(DATASET, "\u57ce\u5e02\u5206\u884c\u4e1a\u6d3b\u529b\u6307\u6570_\u65e5\u5ea6.csv")

    IN_PERSON = ["F52", "H", "H61", "H62", "R86", "R87", "R88", "R89", "O79", "O80"]
    REMOTE = ["F51", "K70", "C29", "C36", "Y"]
    DAILY_SECTORS = set(IN_PERSON + REMOTE)
    GROUP_REMOTE = {"D", "G56", "I63", "I64", "I65", "J66", "J67", "J68", "J69", "K70", "Y"}
    GROUP_PRESENCE = {"F52", "H", "H61", "H62", "O79", "O80", "R86", "R87", "R88", "R89", "G54", "P82", "Q83", "R83"}

    def invert(x):
        return np.expm1(pd.to_numeric(x, errors="coerce").astype("float64") / 10.0)

    start = time.time()
    annual, daily, groups = [], [], []
    n_rows = 0
    reader = pd.read_csv(SECTOR_DAILY, usecols=["mchnt_city", "sic", "date", "cval", "cval_in", "cval_out", "cvol"],
                         dtype={"mchnt_city": str, "sic": str, "date": str}, chunksize=3_000_000)
    for chunk in reader:
        n_rows += len(chunk)
        grp = chunk[["mchnt_city", "date"]].copy()
        for c, n in (("cval", "T"), ("cval_in", "I"), ("cval_out", "O")):
            grp[n] = invert(chunk[c]).fillna(0.0)
        grp["grp"] = np.where(chunk.sic.isin(GROUP_REMOTE), "REMOTE", np.where(chunk.sic.isin(GROUP_PRESENCE), "PRESENCE", "OTHER"))
        groups.append(grp.groupby(["mchnt_city", "date", "grp"])[["T", "I", "O"]].sum())

        date_map = {d: pd.to_datetime(d, format="%d%b%Y") for d in chunk.date.unique()}
        chunk["date"] = chunk.date.map(date_map)

        sums = chunk[["mchnt_city", "sic", "date"]].copy()
        sums["T"] = invert(chunk.cval)
        sums["I"] = invert(chunk.cval_in)
        sums["O"] = invert(chunk.cval_out)
        sums["year"] = sums.date.dt.year
        annual.append(sums.groupby(["mchnt_city", "year", "sic"])[["T", "I", "O"]].agg(["sum", "count"]))

        keep = chunk[chunk.sic.isin(DAILY_SECTORS)]
        keep = keep[(keep.date >= "2017-01-01") & (keep.date <= "2019-12-31")]
        keep = keep.assign(V=invert(keep.cval), N=invert(keep.cvol))
        daily.append(keep[["mchnt_city", "sic", "date", "V", "N"]])
        print(n_rows, f"{time.time() - start:.0f}", flush=True)

    annual = pd.concat(annual)
    annual.columns = ["_".join(c) for c in annual.columns]
    annual = annual.groupby(level=[0, 1, 2]).sum().reset_index()
    annual.to_parquet(os.path.join(INTERMEDIATE, "sector_annual_sums.parquet"))

    groups = pd.concat(groups).groupby(level=[0, 1, 2]).sum().reset_index()
    groups["date"] = pd.to_datetime(groups.date, format="%d%b%Y")
    groups.to_parquet(os.path.join(INTERMEDIATE, "sector_group_daily.parquet"))

    daily = pd.concat(daily, ignore_index=True)
    daily.to_parquet(os.path.join(INTERMEDIATE, "sector_sic_daily.parquet"), index=False)
    print("rows read", n_rows, "annual rows", len(annual), "daily rows", len(daily), "seconds", f"{time.time() - start:.0f}")


def event_estimates_and_robustness():
    """Robustness of the event-level inbound and outbound spending gaps (G_in, G_out): the event catalogue and
    counterfactual are rebuilt under four extreme-rainfall rules and three counterfactual designs (specification curve and
    sample sizes), events are flagged by CPC/CHIRPS confirmation and tropical-cyclone proximity (event_flags.csv), and the
    pooled gaps are estimated for event subgroups (fig1d_forest.csv)."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    GIS = os.path.join(DATASET, "GIS")
    ENC = "utf-8-sig"
    START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")
    WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])

    def hav(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(a))

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

    def city_geometry():
        """Queen adjacency edges and equal-area centroids of the prefecture boundaries."""
        g = gpd.read_file(os.path.join(GIS, "Prefecture City.shp"))
        if int((~g.geometry.is_valid).sum()):
            rep = shapely.make_valid(g.geometry.array)
            g.geometry = [polygonal_only(x) for x in rep]
        g = gpd.GeoDataFrame({"city_name": g.city_name.astype("string").str.strip(), "geometry": g.geometry},
                             crs=g.crs).dissolve(by="city_name", as_index=False).to_crs(4326)
        q = g.sindex.query(g.geometry, predicate="intersects")
        edges = set()
        for a, b in zip(q[0], q[1]):
            if a != b:
                edges.add(tuple(sorted((str(g.city_name.iloc[a]), str(g.city_name.iloc[b])))))
        edges = pd.DataFrame(sorted(edges), columns=["a", "b"])
        pr = g.to_crs("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs")
        cen = gpd.GeoSeries(pr.geometry.centroid, crs=pr.crs).to_crs(4326)
        cen = pd.DataFrame({"city_name": g.city_name.astype(str).values, "lon": cen.x.values, "lat": cen.y.values}).set_index("city_name")
        return edges, cen

    def ibtracs_fixes():
        ib = pd.read_csv(os.path.join(EXTERNAL, "IBTrACS", "ibtracs.WP.list.v04r01.csv"), skiprows=[1], low_memory=False,
                         usecols=["SID", "SEASON", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "TRACK_TYPE", "USA_SSHS"])
        ib["t"] = pd.to_datetime(ib.ISO_TIME) + pd.Timedelta(hours=8)
        ib = ib[(ib.t >= "2016-11-01") & (ib.t < "2020-02-01")].copy()
        ib = ib[ib.TRACK_TYPE == "main"]
        ib["bjt_date"] = ib.t.dt.strftime("%Y-%m-%d")
        ib = ib[["SID", "NAME", "bjt_date", "LAT", "LON", "NATURE", "USA_SSHS"]].reset_index(drop=True)
        ib["LAT"] = pd.to_numeric(ib.LAT)
        ib["LON"] = pd.to_numeric(ib.LON)
        return ib

    def load_base():
        B = {}
        up = pd.read_csv(os.path.join(INTERMEDIATE, "unionpay_city_daily_2017_2019.csv"),
                         dtype={"city_id": str}, encoding=ENC)
        up["date"] = pd.to_datetime(up.date)
        up["city_name"] = up.city_name.astype(str).str.strip()
        B["up"] = up
        era = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC, usecols=["city_name", "date", "rain_mm"])
                         for y in (2017, 2018, 2019)], ignore_index=True)
        era["city_name"] = era.city_name.astype(str).str.strip()
        era["date"] = pd.to_datetime(era.date)
        B["era"] = era
        th = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC)
        th["city_name"] = th.city_name.str.strip()
        u, s, x, lam = th.pot_threshold_mm, th.gpd_scale_mm, th.gpd_shape, th.cluster_rate_per_year
        T = 5.0
        rl5 = np.where(np.abs(x) > 0, u + s / x * ((T * lam) ** x - 1), u + s * np.log(T * lam))
        th["RL5"] = np.where(th.return_level_3yr_mm.notna(), rl5, np.nan)
        B["th"] = th.set_index("city_name")
        B["edges"], cen = city_geometry()
        subst = pd.read_csv(os.path.join(EXTERNAL, "city_centroid_substitutes.csv"), encoding=ENC)
        for r in subst.itertuples(index=False):
            cen.loc[r.city_name] = cen.loc[r.centroid_of]
        B["cen"] = cen
        audit = pd.read_csv(os.path.join(GIS, "city_boundary_match_audit.csv"), encoding=ENC)
        reg = pd.read_csv(os.path.join(EXTERNAL, "province_region7.csv"), encoding=ENC)
        audit["region"] = audit.province.map(dict(zip(reg.province, reg.region7)))
        B["region"] = audit.set_index("UnionPay city").region
        B["region_group"] = dict(zip(reg.region7, reg.region_group))
        B["ib"] = ibtracs_fixes()
        return B

    def clean_flags(B, rain_mm=25.0, km=500.0):
        """City-day is 'clean' if no city within km had rain >= rain_mm and no cyclone fix lay within km."""
        up, cen, era, ib = B["up"], B["cen"], B["era"], B["ib"]
        ucity = np.array(sorted(up.city_name.unique()))
        allc = np.array(cen.index)
        D = hav(cen.loc[ucity, "lat"].values[:, None], cen.loc[ucity, "lon"].values[:, None],
                cen.loc[allc, "lat"].values[None, :], cen.loc[allc, "lon"].values[None, :])
        idx = {c: i for i, c in enumerate(allc)}
        wet_by = era.loc[era.rain_mm >= rain_mm].groupby("date").city_name.apply(list).to_dict()
        ibd = {d: g for d, g in ib.groupby("bjt_date")}
        rows = []
        for d in pd.date_range(START, END):
            ok = np.ones(len(ucity), bool)
            w = wet_by.get(d)
            if w:
                ok &= D[:, [idx[c] for c in w]].min(axis=1) > km
            g = ibd.get(d.strftime("%Y-%m-%d"))
            if g is not None and len(g):
                dt = hav(cen.loc[ucity, "lat"].values[:, None], cen.loc[ucity, "lon"].values[:, None], g.LAT.values[None, :], g.LON.values[None, :])
                ok &= dt.min(axis=1) > km
            rows.append(pd.DataFrame({"city_name": ucity, "date": d, "clean": ok.astype(int)}))
        return pd.concat(rows, ignore_index=True)

    def extreme_days(B, defn):
        era = B["era"].copy()
        th = B["th"]
        rl3 = era.city_name.map(th.return_level_3yr_mm)
        rl5 = era.city_name.map(th.RL5)
        if defn == "baseline":
            m = (era.rain_mm >= 50) | (era.rain_mm >= rl3.fillna(np.inf))
        elif defn == "RL3":
            m = era.rain_mm >= rl3.fillna(np.inf)
        elif defn == "50mm":
            m = era.rain_mm >= 50
        elif defn == "RL5":
            m = era.rain_mm >= rl5.fillna(np.inf)
        else:
            raise ValueError(defn)
        x = era.loc[m].copy()
        x["return_level_3yr_mm"] = x.city_name.map(th.return_level_3yr_mm)
        return x.sort_values(["city_name", "date"]).reset_index(drop=True)

    class UnionFind:
        def __init__(self, n):
            self.p = list(range(n))

        def find(self, i):
            while self.p[i] != i:
                self.p[i] = self.p[self.p[i]]
                i = self.p[i]
            return i

        def union(self, a, b):
            a, b = self.find(a), self.find(b)
            if a != b:
                self.p[max(a, b)] = min(a, b)

    def catalogue(B, xd):
        """City events = runs of consecutive extreme days; systems = adjacent cities with overlapping or touching events."""
        recs = []
        for c, g in xd.groupby("city_name", sort=True):
            d = g.date.sort_values()
            grp = (d.diff().dt.days.ne(1)).cumsum()
            for _, gg in g.loc[d.index].groupby(grp.values):
                pk = gg.rain_mm.max()
                recs.append(dict(city_name=c, city_event_start=gg.date.min(), city_event_end=gg.date.max(),
                                 city_event_duration=len(gg), peak_rain_mm=pk, peak_date=gg.loc[gg.rain_mm == pk, "date"].min(),
                                 cumulative_rain_mm=gg.rain_mm.sum()))
        ce = pd.DataFrame(recs).sort_values(["city_event_start", "city_event_end", "city_name", "peak_date"], kind="stable").reset_index(drop=True)
        ce.insert(0, "city_event_id", [f"CE2017_2019_{i:06d}" for i in range(1, len(ce) + 1)])
        by = {c: g.index.tolist() for c, g in ce.groupby("city_name")}
        uf = UnionFind(len(ce))
        s = ce.city_event_start.values
        e = ce.city_event_end.values
        one = np.timedelta64(1, "D")
        for a, b in B["edges"][["a", "b"]].itertuples(index=False):
            for i in by.get(a, []):
                for j in by.get(b, []):
                    if s[j] <= e[i] + one and s[i] <= e[j] + one:
                        uf.union(i, j)
        ce["_comp"] = [uf.find(i) for i in range(len(ce))]
        comp = ce.groupby("_comp").agg(ms=("city_event_start", "min"), me=("city_event_end", "max"), first=("city_event_id", "min"),
                                       ncity=("city_name", "nunique"))
        comp = comp.sort_values(["ms", "me", "first"], kind="stable")
        comp["meteorological_event_id"] = [f"ME2017_2019_{i:05d}" for i in range(1, len(comp) + 1)]
        ce["meteorological_event_id"] = ce._comp.map(comp.meteorological_event_id)
        ce["affected_city_count"] = ce._comp.map(comp.ncity)
        ce["multi_city_event"] = (ce.affected_city_count >= 2).astype(int)
        return ce.drop(columns="_comp")

    def windows(ce, ucities, W=28):
        focal = ce.loc[ce.city_name.isin(ucities)].reset_index(drop=True)
        pieces = []
        for r in focal.itertuples(index=False):
            d = pd.date_range(max(r.city_event_start - pd.Timedelta(days=W), START), min(r.city_event_end + pd.Timedelta(days=W), END))
            pieces.append(pd.DataFrame({"city_event_id": r.city_event_id, "city_name": r.city_name, "date": d,
                                        "k_start": (d - r.city_event_start).days, "k_end": (d - r.city_event_end).days}))
        p = pd.concat(pieces, ignore_index=True)
        p["is_exposure"] = ((p.k_start >= 0) & (p.k_end <= 0)).astype(int)
        expo = pd.concat([pd.DataFrame({"city_name": r.city_name, "date": pd.date_range(r.city_event_start, r.city_event_end), "oid": r.city_event_id})
                          for r in ce.itertuples(index=False)], ignore_index=True)
        p = p.merge(expo, on=["city_name", "date"], how="left")
        p["is_other"] = (p.oid.notna() & (p.oid != p.city_event_id)).astype(int)
        return focal, p.drop(columns="oid")

    def drop_singletons(cols):
        keep = np.ones(len(cols[0]), bool)
        while True:
            ch = False
            for c in cols:
                cnt = np.bincount(c[keep], minlength=c.max() + 1)
                bad = keep & (cnt[c] == 1)
                if bad.any():
                    keep &= ~bad
                    ch = True
            if not ch:
                return keep

    def fit_ipf(y, codes, nlev, tol=1e-12, maxit=5000):
        lm = np.zeros(len(y))
        fe = [np.zeros(k) for k in nlev]
        ys = [np.bincount(c, weights=y, minlength=k) for c, k in zip(codes, nlev)]
        for it in range(maxit):
            dl = 0.0
            for j, (c, k) in enumerate(zip(codes, nlev)):
                ms = np.bincount(c, weights=np.exp(lm), minlength=k)
                with np.errstate(divide="ignore", invalid="ignore"):
                    adj = np.where(ms > 0, np.log(ys[j] / ms), 0.0)
                adj[~np.isfinite(adj)] = 0.0
                fe[j] += adj
                lm += adj[c]
                dl = max(dl, float(np.abs(adj).max()))
            if dl < tol:
                break
        return fe, lm, it + 1

    def keys(df, cf, cityfe="moy", region=None, predict=False):
        """Fixed-effect keys: city x month-of-year (or season) and date (CF0), date x clean flag (CF1), date x region (CF2)."""
        if cityfe == "moy":
            k1 = df.city_name + "|" + df.date.dt.month.astype(str)
        else:
            k1 = df.city_name + "|S" + ((df.date.dt.month % 12) // 3).astype(str)
        ds = df.date.dt.strftime("%Y-%m-%d")
        if cf == "CF0":
            k2 = ds
        elif cf == "CF1":
            k2 = (ds + "|1") if predict else (ds + "|" + df.clean.astype(int).astype(str))
        elif cf == "CF2":
            k2 = ds + "|" + df.city_name.map(region)
        else:
            raise ValueError(cf)
        return k1.values, k2.values

    def fit_cf(up, hold, cf, B, cityfe="moy", cleandf=None, outcomes=("local_spending", "inbound_spending", "outbound_spending")):
        tr = up.merge(hold.assign(_h=1), on=["city_name", "date"], how="left")
        tr = tr.loc[tr._h.isna()].drop(columns="_h")
        if cf == "CF1":
            tr = tr.merge(cleandf, on=["city_name", "date"], how="left")
        k1, k2 = keys(tr, cf, cityfe, region=B["region"])
        c1, u1 = pd.factorize(k1)
        c2, u2 = pd.factorize(k2)
        keep = drop_singletons([c1, c2])
        a1, l1 = pd.factorize(k1[keep])
        a2, l2 = pd.factorize(k2[keep])
        maps = {}
        for o in outcomes:
            fe, lm, it = fit_ipf(tr[o].values[keep].astype(float), [a1, a2], [len(l1), len(l2)])
            maps[o] = (pd.Series(fe[0], index=l1), pd.Series(fe[1], index=l2))
        return maps

    def predict(days, maps, cf, B, cityfe="moy"):
        k1, k2 = keys(days, cf, cityfe, region=B["region"], predict=True)
        out = {}
        sup = None
        for o, (m1, m2) in maps.items():
            v1 = pd.Series(k1).map(m1).values
            v2 = pd.Series(k2).map(m2).values
            s = ~(np.isnan(v1) | np.isnan(v2))
            out["normal_" + o] = np.where(s, np.exp(np.nan_to_num(v1) + np.nan_to_num(v2)), np.nan)
            sup = s if sup is None else sup
        out["sup"] = sup.astype(int)
        return pd.DataFrame(out, index=days.index)

    def attach(p, up, preds):
        q = p.merge(up[["city_name", "date", "local_spending", "inbound_spending", "outbound_spending"]], on=["city_name", "date"], how="left")
        return q.merge(preds, on=["city_name", "date"], how="left")

    def complete_end14(q):
        b = q.loc[q.k_end.between(-14, 14)]
        g = b.groupby("city_event_id").sup.agg(["sum", "size"])
        return (g["sum"] == g["size"]) & (g["size"] == 29)

    def shocks(q, focal, ids):
        """Exposure-day deviation from normal minus the clean pre-event mean; G = inbound (outbound) minus local."""
        fr = q.loc[q.city_event_id.isin(ids)].copy()
        for m in ("local", "inbound", "outbound"):
            fr["A_" + m] = 100 * (fr[m + "_spending"] / fr["normal_" + m + "_spending"] - 1)
        fr["G_in_raw"] = fr.A_inbound - fr.A_local
        fr["G_out_raw"] = fr.A_outbound - fr.A_local
        pre = fr.loc[fr.k_start.between(-14, -1) & (fr.is_other == 0)]
        ex = fr.loc[fr.is_exposure == 1]
        ev = focal.set_index("city_event_id").loc[sorted(ids)].copy()
        ev["clean_pre_days"] = pre.groupby("city_event_id").size()
        for m in ("local", "inbound", "outbound"):
            ev["CurrentShock_" + m] = ex.groupby("city_event_id")["A_" + m].mean() - pre.groupby("city_event_id")["A_" + m].mean()
        for n in ("G_in", "G_out"):
            ev[n] = ex.groupby("city_event_id")[n + "_raw"].mean() - pre.groupby("city_event_id")[n + "_raw"].mean()
        ev = ev.reset_index()
        return ev.loc[ev.clean_pre_days >= 7].reset_index(drop=True)

    def _cl_meat(e, codes):
        s = np.bincount(codes, weights=e)
        return (s ** 2).sum()

    def twoway(y, city, sysid):
        """Mean with two-way clustered (city, system) standard error, t(G - 1)."""
        y = np.asarray(y, float)
        n = len(y)
        c1 = pd.factorize(np.asarray(city))[0]
        c2 = pd.factorize(np.asarray(sysid))[0]
        c12 = pd.factorize(pd.Series(c1).astype(str) + "_" + pd.Series(c2).astype(str))[0]
        G = min(c1.max() + 1, c2.max() + 1)
        b = y.mean()
        e = y - b
        meat = _cl_meat(e, c1) + _cl_meat(e, c2) - _cl_meat(e, c12)
        V = G / (G - 1) * meat / n ** 2
        se = np.sqrt(max(V, 0))
        df = G - 1
        t = b / se
        p = 2 * stats.t.sf(abs(t), df)
        cr = stats.t.ppf(0.975, df)
        return dict(est=b, se=se, ci_low=b - cr * se, ci_high=b + cr * se, p=p, df=df, N=n, n_city=c1.max() + 1, n_sys=c2.max() + 1)

    def wcr(y, city, sysid, B=1999, seed=12345, null=0.0):
        """Wild cluster bootstrap p-value (Webb weights by system) for the two-way clustered t statistic."""
        y = np.asarray(y, float)
        n = len(y)
        c1 = pd.factorize(np.asarray(city))[0]
        c2 = pd.factorize(np.asarray(sysid))[0]
        c12 = pd.factorize(pd.Series(c1).astype(str) + "_" + pd.Series(c2).astype(str))[0]
        G = min(c1.max() + 1, c2.max() + 1)
        adj = G / (G - 1)

        def tstat(Y):
            b = Y.mean(1)
            E = Y - b[:, None]
            meat = 0
            for c, sgn in ((c1, 1), (c2, 1), (c12, -1)):
                k = c.max() + 1
                S = np.stack([np.bincount(c, weights=E[i], minlength=k) for i in range(Y.shape[0])])
                meat = meat + sgn * (S ** 2).sum(1)
            se = np.sqrt(np.maximum(adj * meat / n ** 2, 1e-300))
            return (b - null) / se

        t0 = tstat(y[None, :])[0]
        rng = np.random.default_rng(seed)
        ng = c2.max() + 1
        V = WEBB[rng.integers(0, 6, size=(B, ng))]
        ts = tstat(null + (y - null)[None, :] * V[:, c2])
        return float(np.mean(np.abs(ts) >= abs(t0)))

    def r1row(ev, label, wcrB=1999, extra=None, winsor=None):
        d = {"spec": label}
        if extra:
            d.update(extra)
        for n in ("G_in", "G_out"):
            y = ev[n].values.astype(float)
            if winsor:
                lo, hi = np.percentile(y, [winsor, 100 - winsor])
                y = np.clip(y, lo, hi)
            r = twoway(y, ev.city_name.values, ev.meteorological_event_id.values)
            for k in ("est", "se", "ci_low", "ci_high", "p"):
                d[f"{n}_{k}"] = r[k]
            d[f"{n}_wcr_p"] = wcr(y, ev.city_name.values, ev.meteorological_event_id.values, B=wcrB) if wcrB else np.nan
            d.update(N=r["N"], n_city=r["n_city"], n_sys=r["n_sys"], df=r["df"])
        return d

    def run(B, defn, cf, cleandf=None):
        """Rebuild catalogue, counterfactual and event gaps for one rule; events losing support are recovered with a
        14-day hold-out and then with a season fixed effect where possible."""
        up = B["up"]
        uc = set(up.city_name)
        xd = extreme_days(B, defn)
        ce = catalogue(B, xd)
        focal, p = windows(ce, uc, 28)

        def fit_predict(pp, W, cityfe):
            if W == 28:
                hold = pp[["city_name", "date"]].drop_duplicates()
            else:
                hold = pp.loc[pp.k_start.ge(-W) & pp.k_end.le(W), ["city_name", "date"]].drop_duplicates()
            maps = fit_cf(up, hold, cf, B, cityfe=cityfe, cleandf=cleandf)
            days = p[["city_name", "date"]].drop_duplicates().reset_index(drop=True)
            pr = pd.concat([days, predict(days, maps, cf, B, cityfe=cityfe)], axis=1)
            return attach(p, up, pr)

        q = fit_predict(p, 28, "moy")
        comp = complete_end14(q)
        ids = set(comp[comp].index)
        ev = shocks(q, focal, ids)
        res = dict(n_xdays=len(xd), n_catalogue=len(ce), n_focal=len(focal), ev=ev, ce=ce)
        lost = focal.loc[~focal.city_event_id.isin(ids)]
        inb = lost.loc[(lost.city_event_start - pd.Timedelta(days=14) >= START) & (lost.city_event_end + pd.Timedelta(days=14) <= END), "city_event_id"]
        res["n_lost"] = len(lost)
        q1 = fit_predict(p, 14, "moy")
        c1 = complete_end14(q1)
        r1 = set(inb) & set(c1[c1].index)
        rest = set(inb) - r1
        q2 = fit_predict(p, 14, "season")
        c2 = complete_end14(q2)
        r2 = rest & set(c2[c2].index)
        parts = []
        if r1:
            parts.append(shocks(q1, focal, r1).assign(recovery="holdout14"))
        if r2:
            parts.append(shocks(q2, focal, r2).assign(recovery="season_fallback"))
        res["rec"] = pd.concat(parts, ignore_index=True) if parts else ev.iloc[:0].assign(recovery="")
        res["n_rec_h14"] = len(r1)
        res["n_rec_season"] = len(r2)
        res["n_unrecovered"] = len(inb) - len(r1) - len(r2)
        return res

    B = load_base()
    cl = clean_flags(B)
    PRD = list(pd.read_csv(os.path.join(EXTERNAL, "pearl_river_delta_cities.csv"), encoding=ENC).city_name)
    R = {}
    for d in ["baseline", "RL3", "50mm", "RL5"]:
        for c in ["CF0", "CF1", "CF2"]:
            R[(d, c)] = run(B, d, c, cleandf=cl if c == "CF1" else None)
            print(d, c, len(R[(d, c)]["ev"]), flush=True)

    # Specification curve
    rowsB = []
    for d in ["baseline", "RL3", "50mm", "RL5"]:
        for c in ["CF0", "CF1", "CF2"]:
            r = R[(d, c)]
            samples = {"S1_standard": r["ev"], "S2_plus_recovered": pd.concat([r["ev"], r["rec"]], ignore_index=True),
                       "S3_excl_PRD": r["ev"][~r["ev"].city_name.isin(PRD)]}
            for sn, ev in samples.items():
                for w in (None, 1, 5):
                    rowsB.append(r1row(ev, f"{d}|{c}|{sn}|w{w or 0}", winsor=w,
                                       extra=dict(defn=d, cf=c, sample=sn, winsor={None: "none", 1: "1/99", 5: "5/95"}[w])))
    Bsp = pd.DataFrame(rowsB)
    Bsp["in_lt3"] = Bsp.G_in_est.abs() < 3
    Bsp["out_gt3"] = Bsp.G_out_est > 3
    Bsp["in_ci_in_pm3"] = (Bsp.G_in_ci_low > -3) & (Bsp.G_in_ci_high < 3)
    Bsp.to_csv(os.path.join(INTERMEDIATE, "b_spec_curve.csv"), index=False, encoding=ENC)

    ss = pd.DataFrame([{"defn": d, "cf": c, "xdays": R[(d, c)]["n_xdays"], "catalogue": R[(d, c)]["n_catalogue"], "focal": R[(d, c)]["n_focal"],
                        "S1": len(R[(d, c)]["ev"]), "lost": R[(d, c)]["n_lost"], "rec_h14": R[(d, c)]["n_rec_h14"],
                        "rec_season": R[(d, c)]["n_rec_season"], "unrecovered": R[(d, c)]["n_unrecovered"],
                        "S2": len(R[(d, c)]["ev"]) + len(R[(d, c)]["rec"]), "S3": int((~R[(d, c)]["ev"].city_name.isin(PRD)).sum())}
                       for d in ["baseline", "RL3", "50mm", "RL5"] for c in ["CF0", "CF1", "CF2"]])
    ss.to_csv(os.path.join(INTERMEDIATE, "b_sample_sizes.csv"), index=False, encoding=ENC)

    # Event flags: confirmation by CPC and CHIRPS at the ERA5-Land 1981-2016 extreme-day frequency, cyclone proximity
    era_all = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC, usecols=["city_name", "date", "rain_mm"])
                         for y in range(1981, 2020)], ignore_index=True)
    era_all["city_name"] = era_all.city_name.str.strip()
    rl = B["th"].return_level_3yr_mm
    e = era_all[era_all.date < "2017-01-01"].copy()
    e["ext"] = (e.rain_mm >= 50) | (e.rain_mm >= e.city_name.map(rl).fillna(np.inf))
    freq = e.groupby("city_name").ext.mean()
    thr, ext = {}, {}
    for prod, fn in [("cpc", "city_daily_cpc_1981_2019.parquet"), ("chirps", "city_daily_chirps_1981_2019.parquet")]:
        x = pd.read_parquet(os.path.join(INTERMEDIATE, fn))
        x["city_name"] = x.city_name.astype(str).str.strip()
        x["date"] = pd.to_datetime(x.date)
        xh = x[(x.date < "2017-01-01")].dropna(subset=["rain_mm"])
        thr[prod] = pd.Series({c: s.quantile(1 - freq[c]) for c, s in xh.groupby("city_name").rain_mm if c in freq.index})
        ext[prod] = x[(x.date >= "2016-12-01")].set_index(["city_name", "date"]).rain_mm

    ev = R[("baseline", "CF0")]["ev"].copy()

    def window_max(prod, r):
        idx = [(r.city_name, d) for d in pd.date_range(r.city_event_start - pd.Timedelta(days=1), r.city_event_end + pd.Timedelta(days=1))]
        v = ext[prod].reindex(idx)
        return v.max() if v.notna().any() else np.nan

    ev["cpc_max"] = [window_max("cpc", r) for r in ev.itertuples()]
    ev["chirps_max"] = [window_max("chirps", r) for r in ev.itertuples()]
    ev["cpc_thr"] = ev.city_name.map(thr["cpc"])
    ev["chirps_thr"] = ev.city_name.map(thr["chirps"])
    ev["cpc_eq"] = (ev.cpc_max >= ev.cpc_thr).where(ev.cpc_max.notna())
    ev["chirps_eq"] = (ev.chirps_max >= ev.chirps_thr).where(ev.chirps_max.notna() & ev.chirps_thr.notna())
    ev["cpc50"] = (ev.cpc_max >= 50).where(ev.cpc_max.notna())
    ib = B["ib"].copy()
    ib["d"] = pd.to_datetime(ib.bjt_date)
    cen = B["cen"]

    def nearest_cyclone(r):
        g = ib[(ib.d >= r.city_event_start - pd.Timedelta(days=1)) & (ib.d <= r.city_event_end + pd.Timedelta(days=1))]
        if not len(g):
            return np.nan, None
        dd = hav(cen.loc[r.city_name, "lat"], cen.loc[r.city_name, "lon"], g.LAT.values, g.LON.values)
        i = int(np.argmin(dd))
        return float(dd[i]), g.NAME.iloc[i]

    tt = [nearest_cyclone(r) for r in ev.itertuples()]
    ev["TCdist_km"] = [a for a, b in tt]
    ev["TCname"] = [b for a, b in tt]
    for R_ in (300, 500, 800):
        ev[f"TC{R_}"] = (ev.TCdist_km <= R_).astype(int)
    ev["lat"] = ev.city_name.map(cen.lat)
    ev["doy"] = ev.city_event_start.dt.day_of_year
    ev["meiyu"] = ((ev.TC500 == 0) & ev.doy.between(160, 205) & ev.lat.between(26, 35)).astype(int)
    ev["region7"] = ev.city_name.map(B["region"])
    ev["region4"] = ev.region7.map(B["region_group"])
    ev["year"] = ev.city_event_start.dt.year
    ev[["city_event_id", "city_name", "meteorological_event_id", "city_event_start", "city_event_end", "year", "region7", "region4",
        "multi_city_event", "peak_rain_mm", "CurrentShock_local", "CurrentShock_inbound", "CurrentShock_outbound", "G_in", "G_out",
        "cpc_max", "cpc_thr", "cpc_eq", "chirps_max", "chirps_thr", "chirps_eq", "cpc50", "TCdist_km", "TCname", "TC300", "TC500",
        "TC800", "meiyu", "lat", "doy"]].to_csv(os.path.join(INTERMEDIATE, "event_flags.csv"), index=False, encoding=ENC)

    # Subgroup estimates for the forest plot
    F = []

    def add(label, group, e, **kw):
        d = r1row(e, label, **kw)
        d["group"] = group
        F.append(d)

    add("Pooled (643 events)", "Pooled", ev)
    for y in (2017, 2018, 2019):
        add(f"{y} events", "Year", ev[ev.year == y])
    for rg in ("South", "East", "North", "Inland"):
        add(f"{rg}", "Region", ev[ev.region4 == rg])
    add("TC (IBTrACS \u2264500 km)", "Rain type", ev[ev.TC500 == 1])
    add("Non-TC", "Rain type", ev[ev.TC500 == 0])
    add("CPC-confirmed (equal frequency)", "Hazard product", ev[ev.cpc_eq == True])
    add("CPC & CHIRPS confirmed", "Hazard product", ev[(ev.cpc_eq == True) & (ev.chirps_eq == True)])
    add("Far-field date FE (>500 km)", "Counterfactual", R[("baseline", "CF1")]["ev"])
    add("Region \u00d7 date FE", "Counterfactual", R[("baseline", "CF2")]["ev"])
    add("Winsorised 1/99", "Estimator", ev, winsor=1)
    add("Excluding Jul\u2013Sep events", "Sample", ev[~ev.city_event_start.dt.month.isin([7, 8, 9])])
    add("Multi-city systems only", "Sample", ev[ev.multi_city_event == 1])
    add("Excluding PRD (9 cities)", "Sample", ev[~ev.city_name.isin(PRD)])
    cols = ["group", "spec", "N", "n_city", "n_sys", "G_in_est", "G_in_ci_low", "G_in_ci_high", "G_in_p", "G_in_wcr_p",
            "G_out_est", "G_out_ci_low", "G_out_ci_high", "G_out_p", "G_out_wcr_p"]
    pd.DataFrame(F)[cols].to_csv(os.path.join(INTERMEDIATE, "fig1d_forest.csv"), index=False, encoding=ENC)
    print("events", len(ev), "specifications", len(Bsp))


def estimand_decomposition():
    """Event-level estimands of the inbound gap: spending weights for each event (x1_event_frame_weights.csv), the decline
    ratio CD_inbound / CD_local and CD_outbound / CD_local with delta-method and rainfall-system bootstrap intervals
    (x1_T1_diversification_ratio.csv), and inbound and outbound gaps by sector group (x1_T3_sector_groups.csv)."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
    START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")
    WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    warnings.filterwarnings("ignore")

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

    def queen_edges():
        g = gpd.read_file(os.path.join(DATASET, "GIS", "Prefecture City.shp"))
        if int((~g.geometry.is_valid).sum()):
            rep = shapely.make_valid(g.geometry.array)
            g.geometry = [polygonal_only(x) for x in rep]
        g = gpd.GeoDataFrame({"city_name": g.city_name.astype("string").str.strip(), "geometry": g.geometry},
                             crs=g.crs).dissolve(by="city_name", as_index=False).to_crs(4326)
        q = g.sindex.query(g.geometry, predicate="intersects")
        edges = set()
        for a, b in zip(q[0], q[1]):
            if a != b:
                edges.add(tuple(sorted((str(g.city_name.iloc[a]), str(g.city_name.iloc[b])))))
        return pd.DataFrame(sorted(edges), columns=["a", "b"])

    def extreme_days(era, th):
        rl3 = era.city_name.map(th.return_level_3yr_mm)
        x = era.loc[(era.rain_mm >= 50) | (era.rain_mm >= rl3.fillna(np.inf))].copy()
        x["return_level_3yr_mm"] = x.city_name.map(th.return_level_3yr_mm)
        return x.sort_values(["city_name", "date"]).reset_index(drop=True)

    class UnionFind:
        def __init__(self, n):
            self.p = list(range(n))

        def find(self, i):
            while self.p[i] != i:
                self.p[i] = self.p[self.p[i]]
                i = self.p[i]
            return i

        def union(self, a, b):
            a, b = self.find(a), self.find(b)
            if a != b:
                self.p[max(a, b)] = min(a, b)

    def catalogue(edges, xd):
        """City events = runs of consecutive extreme days; systems = adjacent cities with overlapping or touching events."""
        recs = []
        for c, g in xd.groupby("city_name", sort=True):
            d = g.date.sort_values()
            grp = (d.diff().dt.days.ne(1)).cumsum()
            for _, gg in g.loc[d.index].groupby(grp.values):
                pk = gg.rain_mm.max()
                recs.append(dict(city_name=c, city_event_start=gg.date.min(), city_event_end=gg.date.max(),
                                 city_event_duration=len(gg), peak_rain_mm=pk, peak_date=gg.loc[gg.rain_mm == pk, "date"].min(),
                                 cumulative_rain_mm=gg.rain_mm.sum()))
        ce = pd.DataFrame(recs).sort_values(["city_event_start", "city_event_end", "city_name", "peak_date"], kind="stable").reset_index(drop=True)
        ce.insert(0, "city_event_id", [f"CE2017_2019_{i:06d}" for i in range(1, len(ce) + 1)])
        by = {c: g.index.tolist() for c, g in ce.groupby("city_name")}
        uf = UnionFind(len(ce))
        s = ce.city_event_start.values
        e = ce.city_event_end.values
        one = np.timedelta64(1, "D")
        for a, b in edges[["a", "b"]].itertuples(index=False):
            for i in by.get(a, []):
                for j in by.get(b, []):
                    if s[j] <= e[i] + one and s[i] <= e[j] + one:
                        uf.union(i, j)
        ce["_comp"] = [uf.find(i) for i in range(len(ce))]
        comp = ce.groupby("_comp").agg(ms=("city_event_start", "min"), me=("city_event_end", "max"), first=("city_event_id", "min"),
                                       ncity=("city_name", "nunique"))
        comp = comp.sort_values(["ms", "me", "first"], kind="stable")
        comp["meteorological_event_id"] = [f"ME2017_2019_{i:05d}" for i in range(1, len(comp) + 1)]
        ce["meteorological_event_id"] = ce._comp.map(comp.meteorological_event_id)
        ce["affected_city_count"] = ce._comp.map(comp.ncity)
        ce["multi_city_event"] = (ce.affected_city_count >= 2).astype(int)
        return ce.drop(columns="_comp")

    def windows(ce, ucities, W=28):
        focal = ce.loc[ce.city_name.isin(ucities)].reset_index(drop=True)
        pieces = []
        for r in focal.itertuples(index=False):
            d = pd.date_range(max(r.city_event_start - pd.Timedelta(days=W), START), min(r.city_event_end + pd.Timedelta(days=W), END))
            pieces.append(pd.DataFrame({"city_event_id": r.city_event_id, "city_name": r.city_name, "date": d,
                                        "k_start": (d - r.city_event_start).days, "k_end": (d - r.city_event_end).days}))
        p = pd.concat(pieces, ignore_index=True)
        p["is_exposure"] = ((p.k_start >= 0) & (p.k_end <= 0)).astype(int)
        expo = pd.concat([pd.DataFrame({"city_name": r.city_name, "date": pd.date_range(r.city_event_start, r.city_event_end), "oid": r.city_event_id})
                          for r in ce.itertuples(index=False)], ignore_index=True)
        p = p.merge(expo, on=["city_name", "date"], how="left")
        p["is_other"] = (p.oid.notna() & (p.oid != p.city_event_id)).astype(int)
        return focal, p.drop(columns="oid")

    def drop_singletons(cols):
        keep = np.ones(len(cols[0]), bool)
        while True:
            ch = False
            for c in cols:
                cnt = np.bincount(c[keep], minlength=c.max() + 1)
                bad = keep & (cnt[c] == 1)
                if bad.any():
                    keep &= ~bad
                    ch = True
            if not ch:
                return keep

    def fit_ipf(y, codes, nlev, tol=1e-12, maxit=5000):
        lm = np.zeros(len(y))
        fe = [np.zeros(k) for k in nlev]
        ys = [np.bincount(c, weights=y, minlength=k) for c, k in zip(codes, nlev)]
        for it in range(maxit):
            dl = 0.0
            for j, (c, k) in enumerate(zip(codes, nlev)):
                ms = np.bincount(c, weights=np.exp(lm), minlength=k)
                with np.errstate(divide="ignore", invalid="ignore"):
                    adj = np.where(ms > 0, np.log(ys[j] / ms), 0.0)
                adj[~np.isfinite(adj)] = 0.0
                fe[j] += adj
                lm += adj[c]
                dl = max(dl, float(np.abs(adj).max()))
            if dl < tol:
                break
        return fe, lm, it + 1

    def keys(df):
        """City x month-of-year and date fixed-effect keys."""
        return (df.city_name + "|" + df.date.dt.month.astype(str)).values, df.date.dt.strftime("%Y-%m-%d").values

    def fit_cf(up, hold, outcomes=("local_spending", "inbound_spending", "outbound_spending")):
        tr = up.merge(hold.assign(_h=1), on=["city_name", "date"], how="left")
        tr = tr.loc[tr._h.isna()].drop(columns="_h")
        k1, k2 = keys(tr)
        c1, _ = pd.factorize(k1)
        c2, _ = pd.factorize(k2)
        keep = drop_singletons([c1, c2])
        a1, l1 = pd.factorize(k1[keep])
        a2, l2 = pd.factorize(k2[keep])
        maps = {}
        for o in outcomes:
            fe, lm, it = fit_ipf(tr[o].values[keep].astype(float), [a1, a2], [len(l1), len(l2)])
            maps[o] = (pd.Series(fe[0], index=l1), pd.Series(fe[1], index=l2))
        return maps

    def predict(days, maps):
        k1, k2 = keys(days)
        out = {}
        sup = None
        for o, (m1, m2) in maps.items():
            v1 = pd.Series(k1).map(m1).values
            v2 = pd.Series(k2).map(m2).values
            s = ~(np.isnan(v1) | np.isnan(v2))
            out["normal_" + o] = np.where(s, np.exp(np.nan_to_num(v1) + np.nan_to_num(v2)), np.nan)
            sup = s if sup is None else sup
        out["sup"] = sup.astype(int)
        return pd.DataFrame(out, index=days.index)

    def attach(p, up, preds):
        q = p.merge(up[["city_name", "date", "local_spending", "inbound_spending", "outbound_spending"]], on=["city_name", "date"], how="left")
        return q.merge(preds, on=["city_name", "date"], how="left")

    def complete_end14(q):
        b = q.loc[q.k_end.between(-14, 14)]
        g = b.groupby("city_event_id").sup.agg(["sum", "size"])
        return (g["sum"] == g["size"]) & (g["size"] == 29)

    def shocks(q, focal, ids):
        """Exposure-day deviation from normal minus the clean pre-event mean; G = inbound (outbound) minus local."""
        fr = q.loc[q.city_event_id.isin(ids)].copy()
        for m in ("local", "inbound", "outbound"):
            fr["A_" + m] = 100 * (fr[m + "_spending"] / fr["normal_" + m + "_spending"] - 1)
        fr["G_in_raw"] = fr.A_inbound - fr.A_local
        fr["G_out_raw"] = fr.A_outbound - fr.A_local
        pre = fr.loc[fr.k_start.between(-14, -1) & (fr.is_other == 0)]
        ex = fr.loc[fr.is_exposure == 1]
        ev = focal.set_index("city_event_id").loc[sorted(ids)].copy()
        ev["clean_pre_days"] = pre.groupby("city_event_id").size()
        for m in ("local", "inbound", "outbound"):
            ev["CurrentShock_" + m] = ex.groupby("city_event_id")["A_" + m].mean() - pre.groupby("city_event_id")["A_" + m].mean()
        for n in ("G_in", "G_out"):
            ev[n] = ex.groupby("city_event_id")[n + "_raw"].mean() - pre.groupby("city_event_id")[n + "_raw"].mean()
        ev = ev.reset_index()
        return ev.loc[ev.clean_pre_days >= 7].reset_index(drop=True)

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

    def _codes(c1, c2):
        a = pd.factorize(np.asarray(c1))[0]
        b = pd.factorize(np.asarray(c2))[0]
        ab = pd.factorize(pd.Series(a).astype(str) + "_" + pd.Series(b).astype(str))[0]
        return a, b, ab

    def wmean_tw(y, w, city, sysid):
        """Weighted mean with two-way clustered (city, system) standard error, t(G - 1)."""
        y = np.asarray(y, float)
        w = np.ones_like(y) if w is None else np.asarray(w, float)
        a, b, ab = _codes(city, sysid)
        G = min(a.max() + 1, b.max() + 1)
        sw = w.sum()
        est = (w * y).sum() / sw
        s = w * (y - est)
        meat = sum(sg * (np.bincount(c, weights=s) ** 2).sum() for c, sg in ((a, 1), (b, 1), (ab, -1)))
        V = G / (G - 1) * meat / sw ** 2
        se = np.sqrt(max(V, 0))
        df = G - 1
        t = stats.t.ppf(.975, df)
        t90 = stats.t.ppf(.95, df)
        return dict(est=est, se=se, lo=est - t * se, hi=est + t * se, lo90=est - t90 * se, hi90=est + t90 * se,
                    p=2 * stats.t.sf(abs(est / se), df) if se > 0 else np.nan, df=df, N=len(y), n_city=a.max() + 1, n_sys=b.max() + 1)

    def wcr_w(y, w, city, sysid, B=9999, seed=12345):
        """Wild cluster bootstrap p-value for a zero weighted mean (Webb weights by system)."""
        y = np.asarray(y, float)
        w = np.ones_like(y) if w is None else np.asarray(w, float)
        a, b, ab = _codes(city, sysid)
        G = min(a.max() + 1, b.max() + 1)
        sw = w.sum()

        def tst(Y):
            est = (Y * w).sum(1) / sw
            S = w * (Y - est[:, None])
            meat = 0
            for c, sg in ((a, 1), (b, 1), (ab, -1)):
                k = c.max() + 1
                M = np.stack([np.bincount(c, weights=S[i], minlength=k) for i in range(Y.shape[0])])
                meat = meat + sg * (M ** 2).sum(1)
            return est / np.sqrt(np.maximum(G / (G - 1) * meat / sw ** 2, 1e-300))

        t0 = tst(y[None, :])[0]
        rng = np.random.default_rng(seed)
        ng = b.max() + 1
        out = []
        for st in range(0, B, 1000):
            nb = min(1000, B - st)
            V = WEBB[rng.integers(0, 6, size=(nb, ng))]
            out.append(tst(y[None, :] * V[:, b]))
        ts = np.concatenate(out)
        return float(np.mean(np.abs(ts) >= abs(t0)))

    def row(y, w, city, sysid, label, B=9999, **kw):
        ok = np.isfinite(np.asarray(y, float))
        if w is not None:
            ok &= np.isfinite(np.asarray(w, float))
        y = np.asarray(y, float)[ok]
        w2 = None if w is None else np.asarray(w, float)[ok]
        city = np.asarray(city)[ok]
        sysid = np.asarray(sysid)[ok]
        r = wmean_tw(y, w2, city, sysid)
        r["p_wcr"] = wcr_w(y, w2, city, sysid, B=B) if B else np.nan
        r["n_dropped_nonfinite"] = int((~ok).sum())
        return dict(label=label, **kw, **r)

    def ratio_delta(yn, yd, w, city, sysid):
        """R = weighted mean(yn) / weighted mean(yd) with a delta-method two-way clustered interval."""
        yn = np.asarray(yn, float)
        yd = np.asarray(yd, float)
        w = np.ones_like(yn) if w is None else np.asarray(w, float)
        a, b, ab = _codes(city, sysid)
        G = min(a.max() + 1, b.max() + 1)
        sw = w.sum()
        mn = (w * yn).sum() / sw
        md = (w * yd).sum() / sw
        R = mn / md
        s = w * ((yn - mn) - R * (yd - md)) / md
        meat = sum(sg * (np.bincount(c, weights=s) ** 2).sum() for c, sg in ((a, 1), (b, 1), (ab, -1)))
        se = np.sqrt(max(G / (G - 1) * meat / sw ** 2, 0))
        t = stats.t.ppf(.975, G - 1)
        return dict(R=R, se=se, lo=R - t * se, hi=R + t * se, mean_num=mn, mean_den=md)

    def sys_boot_idx(sysid, B=1999, seed=2026):
        s = pd.factorize(np.asarray(sysid))[0]
        ng = s.max() + 1
        groups = [np.where(s == g)[0] for g in range(ng)]
        rng = np.random.default_rng(seed)
        for _ in range(B):
            pick = rng.integers(0, ng, ng)
            yield np.concatenate([groups[g] for g in pick])

    # Rebuild the canonical catalogue, counterfactual and event shocks (baseline rule, city x month-of-year + date FE)
    up0 = pd.read_csv(os.path.join(INTERMEDIATE, "unionpay_city_daily_2017_2019.csv"),
                      dtype={"city_id": str}, encoding=ENC)
    up0["date"] = pd.to_datetime(up0.date)
    up0["city_name"] = up0.city_name.astype(str).str.strip()
    era = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC, usecols=["city_name", "date", "rain_mm"])
                     for y in (2017, 2018, 2019)], ignore_index=True)
    era["city_name"] = era.city_name.astype(str).str.strip()
    era["date"] = pd.to_datetime(era.date)
    th = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC)
    th["city_name"] = th.city_name.str.strip()
    th = th.set_index("city_name")
    ce = catalogue(queen_edges(), extreme_days(era, th))
    focal, p = windows(ce, set(up0.city_name), 28)
    hold = p[["city_name", "date"]].drop_duplicates()
    days = p[["city_name", "date"]].drop_duplicates().reset_index(drop=True)

    def build_q(upx):
        maps = fit_cf(upx, hold)
        pr = pd.concat([days, predict(days, maps)], axis=1)
        return attach(p, upx, pr)

    q0 = build_q(up0)
    comp = complete_end14(q0)
    ev0 = shocks(q0, focal, set(comp[comp].index))
    ids = set(ev0.city_event_id)

    # Sector-group series (presence-based, remote, other, all)
    gg = pd.read_parquet(os.path.join(INTERMEDIATE, "sector_group_daily.parquet"))
    gg = gg[gg.date <= pd.Timestamp("2019-12-31")].copy()
    gg["city_name"] = gg.mchnt_city.map(up0[["city_id", "city_name"]].drop_duplicates().set_index("city_id").city_name)
    grid = up0[["city_name", "date"]]
    GP = {}
    for grp in ["PRESENCE", "REMOTE", "OTHER"]:
        x = grid.merge(gg[gg.grp == grp][["city_name", "date", "T", "I", "O"]], on=["city_name", "date"], how="left").fillna(0.0)
        GP[grp] = pd.DataFrame({"city_name": x.city_name, "date": x.date, "local_spending": x["T"] - x["I"], "inbound_spending": x["I"], "outbound_spending": x["O"]})
    allg = GP["PRESENCE"].copy()
    for c in ["local_spending", "inbound_spending", "outbound_spending"]:
        allg[c] = GP["PRESENCE"][c].values + GP["REMOTE"][c].values + GP["OTHER"][c].values
    GP["ALL"] = allg
    GQ, GE = {}, {}
    for grp in ["PRESENCE", "REMOTE", "OTHER", "ALL"]:
        GQ[grp] = build_q(GP[grp])
        GE[grp] = shocks(GQ[grp], focal, ids).set_index("city_event_id")

    def pre_ex(q):
        fr = q.loc[q.city_event_id.isin(ids)]
        return fr.loc[fr.k_start.between(-14, -1) & (fr.is_other == 0)], fr.loc[fr.is_exposure == 1]

    # Event frame with spending weights: w_P pre-event place-based spending, w_I pre-event inbound, w_Pcf normal spending
    pre, ex = pre_ex(q0)
    E = ev0.set_index("city_event_id")[["city_name", "meteorological_event_id", "G_in", "G_out", "CurrentShock_local", "CurrentShock_inbound", "CurrentShock_outbound"]].copy()
    E["w_P"] = (pre.local_spending + pre.inbound_spending).groupby(pre.city_event_id).mean()
    E["w_I"] = pre.inbound_spending.groupby(pre.city_event_id).mean()
    E["w_Pcf"] = (ex.normal_local_spending + ex.normal_inbound_spending).groupby(ex.city_event_id).mean()
    size = (up0.local_spending + up0.inbound_spending).groupby(up0.city_name).mean()
    E["size"] = E.city_name.map(size)
    sz = size.loc[sorted(E.city_name.unique())]
    E["terc"] = E.city_name.map(pd.qcut(sz, 3, labels=["T1_small", "T2_mid", "T3_large"]))
    top5 = sz.sort_values(ascending=False).index[:int(np.ceil(0.05 * len(sz)))].tolist()
    PRD = set(pd.read_csv(os.path.join(EXTERNAL, "pearl_river_delta_cities.csv"), encoding=ENC).city_name)
    REMOTE_H = remote_heavy_cities(dict(zip(up0.city_id.str.strip(), up0.city_name)))
    E["PRD"] = E.city_name.isin(PRD)
    E["REM"] = E.city_name.isin(REMOTE_H)
    E["TOP5"] = E.city_name.isin(top5)
    E.reset_index().to_csv(os.path.join(INTERMEDIATE, "x1_event_frame_weights.csv"), index=False, encoding=ENC)

    # Decline ratios
    for grp in ["PRESENCE", "REMOTE", "OTHER", "ALL"]:
        pr_, ex_ = pre_ex(GQ[grp])
        GE[grp]["w_P"] = (pr_.local_spending + pr_.inbound_spending).groupby(pr_.city_event_id).mean()
        GE[grp]["w_I"] = pr_.inbound_spending.groupby(pr_.city_event_id).mean()
    P_ = GE["PRESENCE"].loc[E.index]
    city = E.city_name.values
    sysv = E.meteorological_event_id.values
    series = {"city inbound/local": (E.CurrentShock_inbound.values, E.CurrentShock_local.values, {"equal": None, "w_P": E.w_P.values, "w_I": E.w_I.values}),
              "city outbound/local": (E.CurrentShock_outbound.values, E.CurrentShock_local.values, {"equal": None, "w_P": E.w_P.values, "w_I": E.w_I.values}),
              "presence inbound/local": (P_.CurrentShock_inbound.values, P_.CurrentShock_local.values, {"equal": None, "w_P": P_.w_P.values, "w_I": P_.w_I.values})}
    idxs = list(sys_boot_idx(sysv, B=1999, seed=2026))
    T1, BOOT = [], {}
    for nm, (yn, yd, ws) in series.items():
        for wn, w in ws.items():
            r = ratio_delta(yn, yd, w, city, sysv)
            ww = np.ones(len(yn)) if w is None else w
            bs = np.array([(ww[i] * yn[i]).sum() / (ww[i] * yd[i]).sum() for i in idxs])
            BOOT[(nm, wn)] = bs
            T1.append(dict(ratio=nm, weight=wn, DR=r["R"], delta_lo=r["lo"], delta_hi=r["hi"], boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5),
                           mean_num=r["mean_num"], mean_den=r["mean_den"], N=len(yn)))
    for wn in ["equal", "w_P", "w_I"]:
        d = BOOT[("presence inbound/local", wn)] - BOOT[("city outbound/local", wn)]
        pt = ([t for t in T1 if t["ratio"] == "presence inbound/local" and t["weight"] == wn][0]["DR"]
              - [t for t in T1 if t["ratio"] == "city outbound/local" and t["weight"] == wn][0]["DR"])
        T1.append(dict(ratio="DR_in(presence) - DR_out(city)", weight=wn, DR=pt, boot_lo=np.percentile(d, 2.5), boot_hi=np.percentile(d, 97.5), N=643))
    pd.DataFrame(T1).to_csv(os.path.join(INTERMEDIATE, "x1_T1_diversification_ratio.csv"), index=False, encoding=ENC)

    # Inbound and outbound gaps by sector group
    def R(df, yc, wc, label, **kw):
        return row(df[yc].values, None if wc is None else df[wc].values, df.city_name.values, df.meteorological_event_id.values, label,
                   outcome=yc, weight=wc or "equal", **kw)

    T3 = []
    for grp in ["PRESENCE", "REMOTE", "OTHER", "ALL"]:
        Gd = GE[grp].loc[E.index].copy()
        Gd["PRD"] = E.PRD.values
        for lab, msk in [("full", np.ones(len(Gd), bool)), ("excl PRD", ~Gd.PRD.values)]:
            d = Gd[msk]
            for wc in [None, "w_P"]:
                for yc in ["G_in", "G_out"]:
                    r = R(d, yc, wc, f"{grp} {lab}", group=grp, sample=lab)
                    r.update({f"mean_shock_{mg}": np.average(d["CurrentShock_" + mg], weights=None if wc is None else d[wc]) for mg in ["local", "inbound", "outbound"]})
                    T3.append(r)
    pd.DataFrame(T3).to_csv(os.path.join(INTERMEDIATE, "x1_T3_sector_groups.csv"), index=False, encoding=ENC)
    print("events", len(E))


def illustration_day():
    """Select the Figure 1a illustration day: the 2017-2019 day on which the most sample cities are simultaneously inside an
    ERA5-Land extreme-rainfall city event, and list each hit city's prior-year inbound spending, partner count, realised
    share of partner weight hit that day and ex-ante shared-rain exposure S."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
    N_FOCAL = 3

    def city_days(cat):
        rows = [(d, r.city_name, r.meteorological_event_id) for r in cat.itertuples()
                for d in pd.date_range(r.city_event_start, r.city_event_end)]
        return pd.DataFrame(rows, columns=["date", "city_name", "meteorological_event_id"])

    cat = pd.read_csv(os.path.join(INTERMEDIATE, "extreme_rainfall_events_2017_2019.csv"), encoding=ENC)
    cat["city_name"] = cat.city_name.astype(str).str.strip()
    panel = pd.read_csv(os.path.join(INTERMEDIATE, "unionpay_city_daily_2017_2019.csv"),
                        encoding=ENC)
    panel["city_name"] = panel.city_name.astype(str).str.strip()
    panel["date"] = pd.to_datetime(panel.date)
    sample = set(panel.city_name)
    cd = city_days(cat)
    counts = cd.loc[cd.city_name.isin(sample)].groupby("date").city_name.nunique().sort_values(ascending=False)
    top = counts.iloc[0]
    tied = counts.index[counts == top]
    if len(tied) > 1:
        raise SystemExit(f"tie on {list(tied)}")
    day = tied[0]
    year = day.year
    hit_all = set(cd.loc[cd.date == day, "city_name"])
    hit_sample = sorted(hit_all & sample)
    systems = sorted(cd.loc[cd.date == day, "meteorological_event_id"].unique())
    inb = panel.loc[panel.date.dt.year == year - 1].groupby("city_name").inbound_spending.mean()

    net = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC, dtype={"host_id": str, "guest_id": str})
    for c in ("host_name", "guest_name"):
        net[c] = net[c].astype(str).str.strip()
    net = net.loc[net.network_year == year - 1]
    cy = pd.read_csv(os.path.join(INTERMEDIATE, "city_year_network.csv"), encoding=ENC, dtype={"city_id": str})
    cy["city_name"] = cy.city_name.astype(str).str.strip()
    S = cy.loc[cy.network_year == year - 1].set_index("city_name").HDC.rsub(1.0)
    focal = list(inb.reindex(hit_sample).sort_values(ascending=False).index[:N_FOCAL])

    rows = []
    for c in sorted(hit_all, key=lambda x: -inb.get(x, -1)):
        q = net.loc[net.host_name == c].copy()
        q["hit"] = q.guest_name.isin(hit_all).astype(int)
        share = float((q.w * q.hit).sum() / q.w.sum()) if len(q) else np.nan
        rows.append({"date": day.date(), "city_name": c, "in_sample": int(c in sample), "focal": int(c in focal),
                     "focal_rank": focal.index(c) + 1 if c in focal else np.nan,
                     "inbound_spending_mean_prior_year": inb.get(c, np.nan), "network_year": year - 1,
                     "n_partners": len(q), "realised_hit_share_all_partners": share,
                     "S_exante_1981_2016": S.get(c, np.nan)})
    sel = pd.DataFrame(rows)
    sel.insert(1, "n_sample_cities_in_event", int(top))
    sel.insert(2, "runner_up_day", counts.index[1].date())
    sel.insert(3, "runner_up_count", int(counts.iloc[1]))
    sel.insert(4, "rainfall_events", ";".join(systems))
    sel.to_csv(os.path.join(INTERMEDIATE, "fig1a_event_selection.csv"), index=False, encoding=ENC, float_format="%.6g")
    print(f"day {day.date()}: {top} sample cities; {len(hit_all)} prefectures hit; focal {focal}")


def extent_gradient():
    """Place-based spending change and share of partner demand hit by the extent of the rainfall system (cities hit by
    the same multi-city event), with the slope on log extent under own-rain controls and city and year-month fixed effects."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
    FE = "meteorological_event_id"
    VCOV = {"CRV1": f"city_name + {FE}"}
    EXT_LABELS = ["1", "2\u20135", "6\u201315", ">15"]

    def fit(data, formula, vcov=None):
        return pf.feols(formula, data=data, vcov=vcov or VCOV)

    def dof_of(model):
        groups = np.atleast_1d(np.asarray(getattr(model, "_G", []) or [], dtype=float))
        return int(groups.min()) - 1 if groups.size else int(model._N - len(model._coefnames))

    def coefficient(model, term="Intercept"):
        row = model.tidy().loc[term]
        return {"estimate": float(row["Estimate"]), "se": float(row["Std. Error"]), "ci_low": float(row["2.5%"]),
                "ci_high": float(row["97.5%"]), "p_value": float(row["Pr(>|t|)"]), "dof": dof_of(model), "N": int(model._N)}

    ev = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, low_memory=False)
    ev["city_name"] = ev.city_name.astype(str).str.strip()
    cat = pd.read_csv(os.path.join(INTERMEDIATE, "extreme_rainfall_events_2017_2019.csv"), encoding=ENC, low_memory=False, usecols=["city_event_id", "affected_city_count"])
    fl = pd.read_csv(os.path.join(INTERMEDIATE, "event_flags.csv"), encoding=ENC,
                     usecols=["city_event_id", "cpc_max", "chirps_max"])
    d = ev.merge(cat, on="city_event_id", how="left", validate="1:1").merge(fl, on="city_event_id", how="left",
                                                                           validate="1:1")
    d["cls"] = pd.cut(d.affected_city_count, [0, 1.5, 5, 15, 1000], labels=EXT_LABELS)
    d["ym"] = pd.to_datetime(d.city_event_start).dt.strftime("%Y-%m")
    d["lext"] = np.log(d.affected_city_count)
    for c in ("peak_rain_mm", "cumulative_rain_mm", "cpc_max", "chirps_max"):
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std()
    SYS, TWO = {"CRV1": FE}, VCOV
    rows = []
    for series, col in (("place-based spending (pp)", "CurrentShock_merchant"),
                        ("local spending (pp)", "CurrentShock_local"),
                        ("partner demand hit (share)", "AffectedWeight")):
        for lab in EXT_LABELS:
            q = d.loc[(d.cls == lab) & d[col].notna()]
            r = {"series": series, "column": col, "extent_class": lab, "n_events": len(q),
                 "n_systems": q[FE].nunique(), "n_cities": q.city_name.nunique(), "mean": q[col].mean()}
            if q[col].std() > 0:
                s_, t_ = coefficient(fit(q, f"{col} ~ 1", SYS)), coefficient(fit(q, f"{col} ~ 1", TWO))
                r.update(ci_low_system=s_["ci_low"], ci_high_system=s_["ci_high"],
                         ci_low_twoway=t_["ci_low"], ci_high_twoway=t_["ci_high"])
            else:                                                # single-city systems: no partner is hit, by definition
                r.update(ci_low_system=q[col].mean(), ci_high_system=q[col].mean(),
                         ci_low_twoway=q[col].mean(), ci_high_twoway=q[col].mean())
            rows.append(r)
    classes = pd.DataFrame(rows)
    formula = ("CurrentShock_merchant ~ lext + z_peak_rain_mm + z_cumulative_rain_mm + z_cpc_max + z_chirps_max"
               " | city_name + ym")
    srows = []
    for name, vc in (("system", SYS), ("two-way city and system", TWO)):
        c = coefficient(fit(d, formula, vc), "lext")
        srows.append({"outcome": "CurrentShock_merchant", "term": "log(affected_city_count)", "cluster": name,
                      "estimate": c["estimate"], "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                      "p_value": c["p_value"], "dof": c["dof"], "N": c["N"], "formula": formula})
    slope = pd.DataFrame(srows)
    classes.to_csv(os.path.join(INTERMEDIATE, "fig1d_extent_classes.csv"), index=False, encoding=ENC)
    slope.to_csv(os.path.join(INTERMEDIATE, "fig1d_extent_slope.csv"), index=False, encoding=ENC)
    print(classes.to_string(index=False, float_format="%.3f"))
    print(slope.drop(columns="formula").to_string(index=False, float_format="%.4f"))


def main():
    sector_extracts()
    event_estimates_and_robustness()
    estimand_decomposition()
    illustration_day()
    extent_gradient()


if __name__ == "__main__":
    main()
