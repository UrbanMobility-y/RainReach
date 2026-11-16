"""Ex-ante shared-rain exposure S of each city's inbound partners: validation across rainfall products and definitions,
the decay of pairwise co-extreme probability with distance and the national decoupling distance, and two example
partner networks."""
import os
import time
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import pyfixest as pf
import shapely
from scipy.stats import spearmanr, pearsonr
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = r"E:\Extreme Rainfall and UnionPay Spending"
DATASET = os.path.join(ROOT, "Dataset")
EXTERNAL = os.path.join(ROOT, "External data")
INTERMEDIATE = os.path.join(ROOT, "Intermediate")


def shared_rain_validation():
    """Validation of the shared-rain index S (partner-weighted co-exposure): S from ERA5-Land, CPC and CHIRPS under
    equal-frequency and 50 mm rules (city_S_products.csv); the share of partner demand hit during 2018-2019 events
    regressed on S across products, outcomes and samples (H1); agreement between products; stratification by rain regime
    (H3); alternative rainfall-system definitions (H4); bounds for unlisted partners (H5); and calibration (H6)."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
    FE = "meteorological_event_id"
    RNG_SEED = 20261023
    WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    BASELINE_PER_0P1 = 8.0625  # main-text estimate (pp of partner demand hit per +0.1 S), used for magnitude ratios
    warnings.filterwarnings("ignore")

    ev = pd.read_csv(os.path.join(INTERMEDIATE, "events.csv"), encoding=ENC, dtype={"city_id": str}, low_memory=False)
    cpc = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_cpc_1981_2019.parquet"))
    chp = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_chirps_1981_2019.parquet"))
    thr = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC)

    # Daily rainfall matrices (days x 369 units), 1981-2019
    era = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC) for y in range(1981, 2020)], ignore_index=True)
    era["city_name"] = era.city_name.astype(str).str.strip()
    Ew = era.pivot(index="date", columns="city_name", values="rain_mm").sort_index()
    cities = list(Ew.columns)
    dates = pd.to_datetime(Ew.index)

    def towide(d):
        d = d.copy()
        d["city_name"] = d.city_name.astype(str).str.strip()
        w = d.pivot(index="date", columns="city_name", values="rain_mm")
        w.index = pd.to_datetime(w.index)
        return w.reindex(index=dates, columns=cities)

    Cw = towide(cpc)
    Hw = towide(chp)

    # Extreme-day indicators. EF: ERA5-Land rule (>= 50 mm or >= 3-year return level); CPC and CHIRPS at the same
    # 1981-2016 frequency (n-th largest day). 50: >= 50 mm. chi2: two days per year on average.
    thr["city_name"] = thr.city_name.astype(str).str.strip()
    rl = thr.set_index("city_name").return_level_3yr_mm.reindex(cities).fillna(np.inf).to_numpy()
    yrs = dates.year.to_numpy()
    hist = (yrs <= 2016)
    E = Ew.to_numpy(float)
    C = Cw.to_numpy(float)
    Hh = Hw.to_numpy(float)
    P = {"ERA5": E, "CPC": C, "CHIRPS": Hh}
    IND = {}
    IND[("ERA5", "EF")] = ((E >= 50) | (E >= rl[None, :])).astype(np.int8)
    n_i = IND[("ERA5", "EF")][hist].sum(0)

    def kth_thr(X, n):
        Xh = X[hist]
        out = np.full(X.shape[1], np.inf)
        for j in range(X.shape[1]):
            col = Xh[:, j]
            col = col[~np.isnan(col)]
            if n[j] > 0 and len(col) > 0:
                out[j] = np.sort(col)[::-1][n[j] - 1]
        return out

    THR = {}
    for p in ["CPC", "CHIRPS"]:
        THR[(p, "EF")] = kth_thr(P[p], n_i)
        X = np.nan_to_num(P[p], nan=-1)
        IND[(p, "EF")] = (X >= THR[(p, "EF")][None, :]).astype(np.int8)
    for p in P:
        X = np.nan_to_num(P[p], nan=-1)
        IND[(p, "50")] = (X >= 50).astype(np.int8)
        THR[(p, "chi2")] = kth_thr(P[p], np.full(len(cities), 72))
        IND[(p, "chi2")] = (X >= THR[(p, "chi2")][None, :]).astype(np.int8)
    miss_chirps = np.isnan(Hh).all(0)  # unit outside the CHIRPS domain
    for k in IND:
        if k[0] == "CHIRPS":
            IND[k][:, miss_chirps] = 0

    def pair_metrics(Hm):
        """pi_ij = P(j extreme | i extreme), 1981-2016."""
        B = Hm[hist].astype(np.float64)
        own = B.sum(0)
        co0 = B.T @ B
        with np.errstate(invalid="ignore", divide="ignore"):
            pi = np.where(own[:, None] > 0, co0 / own[:, None], np.nan)
        np.fill_diagonal(pi, np.nan)
        return pi

    def chi_metrics(Hm):
        """Same-day (chi0) and +-1-day (chi1) conditional co-occurrence and event synchronisation (es)."""
        B = Hm[hist].astype(np.float64)
        own = B.sum(0)
        Bd = B.copy()
        Bd[1:] = np.maximum(Bd[1:], B[:-1])
        Bd[:-1] = np.maximum(Bd[:-1], B[1:])
        co0 = B.T @ B
        co1 = B.T @ Bd
        cp = B[:-1].T @ B[1:]
        cm = B[1:].T @ B[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            chi0 = np.where(own[:, None] > 0, co0 / own[:, None], np.nan)
            chi1 = np.where(own[:, None] > 0, co1 / own[:, None], np.nan)
            es = np.where((own[:, None] * own[None, :]) > 0, (co0 + cp + cm) / np.sqrt(own[:, None] * own[None, :]), np.nan)
        for a in (chi0, chi1, es):
            np.fill_diagonal(a, np.nan)
        return chi0, chi1, es

    M = {}
    for p in P:
        for rule in ["EF", "50"]:
            M[f"pi_{p}_{rule}"] = pair_metrics(IND[(p, rule)])
        c0, c1, es = chi_metrics(IND[(p, "chi2")])
        M[f"chi0_{p}"] = c0
        M[f"chi1_{p}"] = c1
        M[f"es_{p}"] = es
    mi = np.where(miss_chirps)[0]
    for k in M:
        if "CHIRPS" in k:
            M[k][mi, :] = np.nan
            M[k][:, mi] = np.nan
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    idx = {c: i for i, c in enumerate(cities)}
    fi = hp.focal_city.map(idx).to_numpy()
    gi = hp.partner_city.map(idx).to_numpy()

    # Distance kernels (60 quantile bins of log distance) and partner-level metrics
    net = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC, dtype={"host_id": str, "guest_id": str})
    cy = pd.read_csv(os.path.join(INTERMEDIATE, "city_year_network.csv"), encoding=ENC, dtype={"city_id": str})
    D = np.full((len(cities), len(cities)), np.nan)
    D[fi, gi] = hp.distance_km.to_numpy()
    ld = np.log(hp.distance_km.clip(lower=5))
    edges = np.unique(np.quantile(ld, np.linspace(0, 1, 61)))
    bins = np.clip(np.searchsorted(edges, ld, side="right") - 1, 0, len(edges) - 2)

    def kernel_of(Mm):
        v = Mm[fi, gi]
        return pd.DataFrame({"b": bins, "ld": ld, "v": v}).groupby("b").agg(ld=("ld", "mean"), v=("v", "mean"))

    def kfun(dfk, d):
        return np.interp(np.log(np.clip(np.asarray(d, float), 5, None)), dfk.ld.to_numpy(), dfk.v.to_numpy())

    KER = {k: kernel_of(v) for k, v in M.items()}
    nn = net.copy()
    nn["hi"] = nn.host_name.map(idx)
    nn["gj"] = nn.guest_name.map(idx)
    obs = nn.hi.notna() & nn.gj.notna() & (nn.host_name != nn.guest_name)
    hiI = nn.hi.fillna(0).astype(int).to_numpy()
    gjI = nn.gj.fillna(0).astype(int).to_numpy()
    for k, Mm in M.items():
        v = np.where(obs, Mm[hiI, gjI], np.nan)
        nn[k] = v
        nn[k + "_k"] = np.where(obs & ~np.isnan(v), kfun(KER[k], nn.distance_km.fillna(1.0)), np.nan)

    # City-year S for every metric (S_), its distance-kernel prediction (Sk_) and weight coverage (cov_)
    keys = list(M.keys())
    g = nn.groupby(["host_id", "network_year"])
    out = pd.DataFrame(index=g.size().index)
    for k in keys:
        u = nn[k].notna()
        wsum = nn.w.where(u, 0.0).groupby([nn.host_id, nn.network_year]).sum()
        out["S_" + k] = (nn.w * nn[k]).where(u, 0.0).groupby([nn.host_id, nn.network_year]).sum() / wsum
        out["Sk_" + k] = (nn.w * nn[k + "_k"]).where(u, 0.0).groupby([nn.host_id, nn.network_year]).sum() / wsum
        out["cov_" + k] = wsum
    out = out.reset_index().rename(columns={"host_id": "city_id"})
    out = cy[["city_id", "city_name", "network_year", "coverage", "HDC", "HDC_kernel", "E", "InboundShare", "Near300"]].merge(out, on=["city_id", "network_year"], how="left")
    fr = pd.DataFrame({"city_name": cities, "F50_ERA5": IND[("ERA5", "50")][hist].sum(0) / 36, "F50_CPC": IND[("CPC", "50")][hist].sum(0) / 36,
                       "F50_CHIRPS": np.where(miss_chirps, np.nan, IND[("CHIRPS", "50")][hist].sum(0) / 36), "Fext_ERA5": n_i / 36})
    out = out.merge(fr, on="city_name", how="left")
    cols = (["city_id", "city_name", "network_year", "coverage", "InboundShare", "E", "Near300", "HDC"]
            + [c for c in out.columns if c.startswith(("S_", "Sk_", "cov_"))] + ["F50_ERA5", "F50_CPC", "F50_CHIRPS", "Fext_ERA5"])
    out = out[cols]
    out.to_csv(os.path.join(INTERMEDIATE, "city_S_products.csv"), index=False, encoding=ENC, float_format="%.10g")

    # Event-level outcomes: share of partner weight hit during the event (system rule and same-day rule by product)
    ns = ev.loc[ev.network_sample == 1].reset_index(drop=True).copy()
    dpos = {d: i for i, d in enumerate(dates)}
    netg = {k: v for k, v in nn.groupby(["host_id", "network_year"])}
    ns["s"] = pd.to_datetime(ns.city_event_start)
    ns["e"] = pd.to_datetime(ns.city_event_end)

    def sameday_outcome(p, rule):
        Hm = IND[(p, rule)]
        avail = ~(miss_chirps if p == "CHIRPS" else np.zeros(len(cities), bool))
        res = []
        for r in ns.itertuples():
            b = netg[(r.city_id, r.baseline_year)]
            ok = b.gj.notna() & (b.host_name != b.guest_name)
            j = b.gj.fillna(0).astype(int).to_numpy()
            ok = ok.to_numpy() & avail[j]
            rows = np.arange(dpos[r.s], dpos[r.e] + 1)
            hit = Hm[rows][:, j].max(0) == 1
            res.append(float(b.w.to_numpy()[ok & hit].sum()))
        return np.array(res)

    OUT = {"ERA5sys": ns.AffectedWeight.to_numpy()}
    for p in P:
        for rule in ["EF", "50"]:
            OUT[f"{p}sd_{rule}"] = sameday_outcome(p, rule)
    for k, v in OUT.items():
        ns["Y_" + k] = v
    Scols = [c for c in out.columns if c.startswith(("S_", "Sk_"))]
    ns = ns.drop(columns=[c for c in ns.columns if c in Scols]).merge(
        out[["city_id", "network_year"] + Scols].rename(columns={"network_year": "baseline_year"}), on=["city_id", "baseline_year"], how="left")
    ns["year"] = ns.s.dt.year

    def demean(A, g):
        A = np.asarray(A, float)
        return A - pd.DataFrame(A).groupby(g).transform("mean").to_numpy()

    def wcr_p(d, y, xs, fe, clus, B=9999, seed=RNG_SEED):
        """Wild cluster restricted bootstrap p-value for xs[0] = 0 (Webb weights by cluster; fixed effects demeaned)."""
        g = d[fe].to_numpy()
        Y = demean(d[y].to_numpy()[:, None], g)[:, 0]
        X = demean(d[xs].to_numpy(), g)
        cl = pd.factorize(d[clus])[0]
        G = cl.max() + 1
        N, k = X.shape
        A = np.linalg.inv(X.T @ X)

        def tstat(Ymat):
            bet = A @ X.T @ Ymat
            e = Ymat - X @ bet
            Gm = np.zeros((G, N))
            Gm[cl, np.arange(N)] = 1
            s0 = np.zeros((G, Ymat.shape[1]))
            for j in range(k):
                s0 += A[0, j] * (Gm @ (X[:, [j]] * e))
            v = (G / (G - 1)) * ((N - 1) / (N - k)) * (s0 ** 2).sum(0)
            return bet[0] / np.sqrt(v)

        t_obs = tstat(Y[:, None])[0]
        Xr = X[:, 1:]
        br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
        fit = Xr @ br
        ur = Y - fit
        rng = np.random.default_rng(seed)
        ts = []
        for chunk in np.array_split(np.arange(B), 5):
            v = rng.choice(WEBB, size=(G, len(chunk)))
            ts.append(tstat(fit[:, None] + ur[:, None] * v[cl]))
        ts = np.concatenate(ts)
        return float(np.mean(np.abs(ts) >= abs(t_obs))), float(t_obs)

    def drop_singletons(d, fe):
        c = d.groupby(fe)[fe].transform("size")
        return d.loc[c > 1]

    def est(y, x, d=None, fe=FE, controls=("z_sev",), B=9999, sd_ref=None, quint=True, extra=None):
        """Slope of y on x with severity control and system fixed effects; two-way clustered CI and WCR p-value."""
        d = ns if d is None else d
        xs = [x] + list(controls)
        dd = d.dropna(subset=[y] + xs).copy()
        if fe == FE:
            dd = drop_singletons(dd, fe)
        m = pf.feols(f"{y} ~ {' + '.join(xs)} | {fe}", data=dd, vcov={"CRV1": f"city_name + {FE}"})
        r = m.tidy().loc[x]
        sdv = (ns[x].std() if sd_ref is None else sd_ref)
        p_w, t_w = wcr_p(dd, y, xs, fe, FE, B=B) if B else (np.nan, np.nan)
        res = dict(y=y, x=x, fe=fe, N=int(m._N), G_sys=dd[FE].nunique(), G_city=dd.city_name.nunique(),
                   b_per0p1=float(r.Estimate) * 10, ci_lo=float(r["2.5%"]) * 10, ci_hi=float(r["97.5%"]) * 10,
                   b_per_sd=float(r.Estimate) * 100 * sdv, t=float(r["t value"]), p_2way=float(r["Pr(>|t|)"]), p_wcr=p_w)
        if quint:
            qq = dd.assign(qn=pd.qcut(dd[x], 5, labels=False, duplicates="drop")).groupby("qn")[y].mean()
            res.update(q1=float(qq.iloc[0]) * 100, q5=float(qq.iloc[-1]) * 100, q_ratio=float(qq.iloc[-1] / qq.iloc[0]) if qq.iloc[0] > 0 else np.inf)
        if extra:
            res.update(extra)
        return res

    # H1: grid of S products x outcomes x samples
    Smap = {"ERA5": "S_pi_ERA5_EF", "CPC": "S_pi_CPC_EF", "CHIRPS": "S_pi_CHIRPS_EF"}
    S50 = {"ERA5": "S_pi_ERA5_50", "CPC": "S_pi_CPC_50", "CHIRPS": "S_pi_CHIRPS_50"}
    Ymap = {"ERA5-system": "Y_ERA5sys", "CPC-sameday": "Y_CPCsd_EF", "CHIRPS-sameday": "Y_CHIRPSsd_EF", "ERA5-sameday(suppl)": "Y_ERA5sd_EF"}
    Y50 = {"ERA5-system": "Y_ERA5sys", "CPC-sameday": "Y_CPCsd_50", "CHIRPS-sameday": "Y_CHIRPSsd_50", "ERA5-sameday(suppl)": "Y_ERA5sd_50"}
    grid = []
    for grid_name, SM, YM in [("EF", Smap, Ymap), ("50mm", S50, Y50)]:
        for sp, sx in SM.items():
            for yo, yv in YM.items():
                for samp in ["364", "440", "2018", "2019"]:
                    if samp == "364":
                        r = est(yv, sx)
                    elif samp == "440":
                        r = est(yv, sx, fe="year")
                    else:
                        r = est(yv, sx, d=ns[ns.year == int(samp)])
                    r.update(grid=grid_name, S_product=sp, outcome=yo, sample=samp)
                    grid.append(r)
    G1 = pd.DataFrame(grid)
    G1.to_csv(os.path.join(INTERMEDIATE, "H1_grid.csv"), index=False, encoding=ENC)

    # Agreement of S between products
    sp_rows = []
    for ny in [2017, 2018]:
        o = out[out.network_year == ny]
        for rule, SM in [("EF", Smap), ("50mm", S50), ("chi1", {"ERA5": "S_chi1_ERA5", "CPC": "S_chi1_CPC", "CHIRPS": "S_chi1_CHIRPS"}),
                         ("chi0", {"ERA5": "S_chi0_ERA5", "CPC": "S_chi0_CPC", "CHIRPS": "S_chi0_CHIRPS"})]:
            for a, b_ in [("ERA5", "CPC"), ("ERA5", "CHIRPS"), ("CPC", "CHIRPS")]:
                x = o[[SM[a], SM[b_]]].dropna()
                sp_rows.append(dict(network_year=ny, metric=rule, pair=f"{a}-{b_}", n=len(x), spearman=spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0],
                                    pearson=pearsonr(x.iloc[:, 0], x.iloc[:, 1])[0]))
    pd.DataFrame(sp_rows).to_csv(os.path.join(INTERMEDIATE, "H1_S_spearman_products.csv"), index=False, encoding=ENC)

    # H3: rain regimes (tropical cyclone within 500 km, Meiyu window, other)
    ib = pd.read_csv(os.path.join(EXTERNAL, "IBTrACS", "ibtracs.WP.list.v04r01.csv"), skiprows=[1],
                     usecols=["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "TRACK_TYPE"], low_memory=False)
    ib = ib[pd.to_numeric(ib.SEASON, errors="coerce").between(2016, 2020)].copy()
    ib["LAT"] = pd.to_numeric(ib.LAT)
    ib["LON"] = pd.to_numeric(ib.LON)
    ib["bjt"] = (pd.to_datetime(ib.ISO_TIME) + pd.Timedelta(hours=8)).dt.normalize()
    cen = cy.drop_duplicates("city_name").set_index("city_name")[["lat", "lon"]]  # grid-weighted city centres

    def hav(la1, lo1, la2, lo2):
        p1, p2 = np.radians(la1), np.radians(la2)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(a))

    ibd = {d: g for d, g in ib.groupby("bjt")}
    allev = ev.copy()
    allev["s"] = pd.to_datetime(allev.city_event_start)
    allev["e"] = pd.to_datetime(allev.city_event_end)
    tcd, tcn = [], []
    for r in allev.itertuples():
        la, lo = cen.loc[r.city_name]
        best, nm = np.inf, None
        for d in pd.date_range(r.s - pd.Timedelta(days=1), r.e):
            g = ibd.get(d)
            if g is None:
                continue
            dist = hav(la, lo, g.LAT.to_numpy(), g.LON.to_numpy())
            i = dist.argmin()
            if dist[i] < best:
                best, nm = dist[i], g.NAME.iloc[i]
        tcd.append(best)
        tcn.append(nm)
    allev["TCdist_re"] = tcd
    allev["TC500_re"] = (allev.TCdist_re <= 500).astype(int)
    allev["TCname_re"] = np.where(allev.TC500_re == 1, tcn, None)
    ns = ns.merge(allev[["city_event_id", "TC500_re", "TCdist_re", "TCname_re"]], on="city_event_id", how="left")
    lat_f = ns.city_name.map(cen.lat)
    doy = ns.s.dt.day_of_year
    ns["regime"] = np.where(ns.TC500_re == 1, "TC", np.where(doy.between(160, 205) & lat_f.between(26, 35), "Meiyu", "other"))
    ns["nonTC"] = (ns.TC500_re == 0).astype(int)
    h3 = []
    for yo in ["Y_ERA5sys", "Y_CPCsd_EF"]:
        for nm, dsub in [("all", ns), ("TC", ns[ns.regime == "TC"]), ("nonTC", ns[ns.nonTC == 1]), ("Meiyu", ns[ns.regime == "Meiyu"]), ("other", ns[ns.regime == "other"])]:
            r = est(yo, "S_pi_ERA5_EF", d=dsub)
            r.update(stratum=nm)
            h3.append(r)
    ns["S_x_nonTC"] = ns.S_pi_ERA5_EF * ns.nonTC
    ns["zsev_x_nonTC"] = ns.z_sev * ns.nonTC
    for yo in ["Y_ERA5sys", "Y_CPCsd_EF"]:
        r = est(yo, "S_x_nonTC", controls=("S_pi_ERA5_EF", "z_sev", "zsev_x_nonTC", "nonTC"), quint=False)
        r.update(stratum="interaction S:nonTC (difference nonTC-TC)")
        h3.append(r)
    H3 = pd.DataFrame(h3)
    H3["ratio_vs_baseline"] = H3.b_per0p1 / BASELINE_PER_0P1
    H3.to_csv(os.path.join(INTERMEDIATE, "H3_regime_strata.csv"), index=False, encoding=ENC)

    # H4: rainfall-system definitions (canonical rebuild, same-day connectivity, split of long or wide systems)
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

    gdf = gpd.read_file(os.path.join(DATASET, "GIS", "Prefecture City.shp"))
    if int((~gdf.geometry.is_valid).sum()):
        gdf.geometry = [polygonal_only(x) for x in shapely.make_valid(gdf.geometry.array)]
    gdf = gpd.GeoDataFrame({"city_name": gdf["city_name"].astype("string").str.strip(), "geometry": gdf.geometry},
                           crs=gdf.crs).dissolve(by="city_name", as_index=False).to_crs(4326)
    q = gdf.sindex.query(gdf.geometry, predicate="intersects")
    Eset = set()
    for a, b in zip(q[0], q[1]):
        if a != b:
            Eset.add(tuple(sorted((str(gdf.iloc[int(a)].city_name), str(gdf.iloc[int(b)].city_name)))))
    qe = pd.DataFrame(sorted(Eset), columns=["a", "b"])
    pr = gdf.to_crs("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs")
    c_ = gpd.GeoSeries(pr.geometry.centroid, crs=pr.crs).to_crs(4326)
    acen = pd.DataFrame({"city_name": gdf.city_name.astype(str), "lon": c_.x, "lat": c_.y}).set_index("city_name")

    cat = pd.read_csv(os.path.join(INTERMEDIATE, "extreme_rainfall_events_2017_2019.csv"), encoding=ENC, parse_dates=["city_event_start", "city_event_end", "peak_date"])
    cat["city_name"] = cat.city_name.astype(str).str.strip()
    by_city = {c: g.index.to_numpy() for c, g in cat.groupby("city_name")}
    S_ = cat.city_event_start.to_numpy()
    E_ = cat.city_event_end.to_numpy()
    one = np.timedelta64(1, "D")

    def components(edgefun, nodes=None):
        n = len(cat)
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in zip(qe.a, qe.b):
            for i in by_city.get(a, []):
                for j in by_city.get(b, []):
                    if nodes is not None and not (nodes[i] and nodes[j]):
                        continue
                    if edgefun(i, j):
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
        return np.array([find(i) for i in range(n)])

    canon_edge = lambda i, j: (S_[j] <= E_[i] + one) and (S_[i] <= E_[j] + one)
    same_edge = lambda i, j: (S_[j] <= E_[i]) and (S_[i] <= E_[j])
    comp_c = components(canon_edge)
    comp_a = components(same_edge)

    def comp_stats(lbl):
        d = pd.DataFrame({"c": lbl, "city": cat.city_name, "s": cat.city_event_start, "e": cat.city_event_end})
        st = {}
        for c_, g in d.groupby("c"):
            dur = (g.e.max() - g.s.min()).days + 1
            cs = g.city.unique()
            la = acen.loc[cs, "lat"].to_numpy()
            lo = acen.loc[cs, "lon"].to_numpy()
            span = np.nanmax(hav(la[:, None], lo[:, None], la[None, :], lo[None, :])) if len(cs) > 1 else 0.0
            st[c_] = (dur, span)
        return st

    stc = comp_stats(comp_c)
    viol = {c_ for c_, (du, sp) in stc.items() if du > 3 or sp > 1000}
    nodes = np.isin(comp_c, list(viol))
    sub = components(same_edge, nodes=nodes)
    lbl_b = np.where(nodes, np.char.add("b1_", sub.astype(str)), np.char.add("c_", comp_c.astype(str)))
    st1 = comp_stats(lbl_b)
    viol2 = {c_ for c_, (du, sp) in st1.items() if (du > 3 or sp > 1000)}
    pk = cat.peak_date.dt.strftime("%Y%m%d").to_numpy()
    nodes2 = np.isin(lbl_b, list(viol2))
    pk_edge = lambda i, j: (pk[i] == pk[j]) and (lbl_b[i] == lbl_b[j])
    sub2 = components(pk_edge, nodes=nodes2)
    lbl_b = np.where(nodes2, np.char.add("b2_", sub2.astype(str)), lbl_b)
    cat["sys_a"] = np.char.add("a_", comp_a.astype(str))
    cat["sys_b"] = lbl_b
    cat["sys_c"] = np.char.add("c_", comp_c.astype(str))
    catx = cat.set_index("city_event_id")

    def affected_by(syscol):
        bys = {k: v for k, v in cat.groupby(syscol)}
        res, sysid = [], []
        for r in ns.itertuples():
            b = netg[(r.city_id, r.baseline_year)]
            sid = catx.loc[r.city_event_id, syscol]
            sysid.append(sid)
            g = bys[sid]
            hit = set(g.loc[(g.city_event_start <= r.e) & (g.city_event_end >= r.s) & (g.city_name != r.city_name), "city_name"])
            res.append(float(b.loc[(b.usable == 1) & b.guest_name.isin(hit), "w"].sum()))
        return np.array(res), np.array(sysid)

    for k in ["c", "a", "b"]:
        y, sid = affected_by("sys_" + k)
        ns["Y_sys_" + k] = y
        ns["FE_" + k] = sid
    h4 = []
    for k, lab in [("c", "canonical rebuild"), ("a", "(a) same-day connectivity"), ("b", "(b) split >3 d or >1,000 km")]:
        d = ns.copy()
        d[FE + "_orig"] = d[FE]
        d[FE] = d["FE_" + k]
        r = est("Y_sys_" + k, "S_pi_ERA5_EF", d=d)
        r.update(definition=lab, n_systems_all=len(set(cat["sys_" + k])))
        r["ratio_vs_baseline"] = r["b_per0p1"] / BASELINE_PER_0P1
        h4.append(r)
        r2 = est("AffectedWeight", "S_pi_ERA5_EF", d=d)  # system fixed effects changed, canonical outcome kept
        r2.update(definition=lab + " | FE only (canonical outcome)")
        r2["ratio_vs_baseline"] = r2["b_per0p1"] / BASELINE_PER_0P1
        h4.append(r2)
    pd.DataFrame(h4).to_csv(os.path.join(INTERMEDIATE, "H4_system_definition.csv"), index=False, encoding=ENC)

    # H5: bounds for unlisted partners. LS = listed share of annual inbound spending; unlisted partners are assumed to
    # lie beyond the 9th listed partner's distance with kernel, zero or maximum co-exposure.
    pan = pd.read_csv(os.path.join(INTERMEDIATE, "unionpay_city_daily_2017_2019.csv"),
                      encoding=ENC, usecols=["city_name", "date", "inbound_spending"])
    pan["city_name"] = pan.city_name.astype(str).str.strip()
    pan["network_year"] = pd.to_datetime(pan.date).dt.year
    Iy = pan.groupby(["city_name", "network_year"]).inbound_spending.sum().rename("I").reset_index()
    ql = net.groupby(["host_name", "network_year"]).q.sum().rename("Q").reset_index().rename(columns={"host_name": "city_name"})
    LS = ql.merge(Iy, on=["city_name", "network_year"])
    LS["LS"] = LS.Q / LS.I
    kE = KER["pi_ERA5_EF"]
    bnd = []
    for (h, yv), b in netg.items():
        if yv not in (2017, 2018):
            continue
        name = b.host_name.iloc[0]
        if name not in idx:
            continue
        u = b[b.usable == 1]
        if len(u) == 0:
            continue
        S0 = (u.w * u.pi).sum() / u.w.sum()
        bb = b.dropna(subset=["distance_km"]).sort_values("w", ascending=False)
        dstar = bb.distance_km.iloc[8] if len(bb) >= 9 else bb.distance_km.max()
        i = idx[name]
        listed = set(b.guest_name)
        cand = [j for j, c_ in enumerate(cities) if c_ != name and c_ not in listed and np.isfinite(D[i, j]) and D[i, j] >= dstar]
        pkk = kfun(kE, D[i, cand]).mean() if cand else kfun(kE, [dstar])[0]
        bnd.append(dict(city_name=name, network_year=yv, S=S0, dstar=dstar, n_cand=len(cand), pimiss_kernel=pkk, pimiss_max=u.pi.max()))
    BN = pd.DataFrame(bnd).merge(LS[["city_name", "network_year", "LS"]], on=["city_name", "network_year"], how="left")
    BN["S_b_kernel"] = BN.LS * BN.S + (1 - BN.LS) * BN.pimiss_kernel
    BN["S_b_zero"] = BN.LS * BN.S
    BN["S_b_max"] = BN.LS * BN.S + (1 - BN.LS) * BN.pimiss_max
    ns = ns.drop(columns=[c for c in ns.columns if c.startswith("S_b_") or c == "LS"], errors="ignore").merge(
        BN[["city_name", "network_year", "LS", "S_b_kernel", "S_b_zero", "S_b_max"]].rename(columns={"network_year": "baseline_year"}),
        on=["city_name", "baseline_year"], how="left")
    h5 = []
    for c_ in ["S_pi_ERA5_EF", "S_b_kernel", "S_b_zero", "S_b_max"]:
        r = est("AffectedWeight", c_)
        r.update(spec=c_)
        h5.append(r)
    e364 = drop_singletons(ns, FE)
    cuts = e364.LS.quantile([1 / 3, 2 / 3]).to_numpy()
    ns["LS_terc"] = np.digitize(ns.LS, cuts) + 1
    for t_ in [1, 2, 3]:
        r = est("AffectedWeight", "S_pi_ERA5_EF", d=ns[ns.LS_terc == t_])
        r.update(spec=f"LS tercile {t_}", LS_range=f"{ns[ns.LS_terc == t_].LS.min():.3f}-{ns[ns.LS_terc == t_].LS.max():.3f}")
        h5.append(r)
    pd.DataFrame(h5).to_csv(os.path.join(INTERMEDIATE, "H5_truncation_bounds.csv"), index=False, encoding=ENC)

    # H6: calibration of realised shares against S, and reliability by S decile
    pairs6 = [("ERA5", "S_pi_ERA5_EF", "Y_ERA5sys"), ("ERA5", "S_pi_ERA5_EF", "Y_ERA5sd_EF"), ("CPC", "S_pi_CPC_EF", "Y_CPCsd_EF"),
              ("CHIRPS", "S_pi_CHIRPS_EF", "Y_CHIRPSsd_EF"), ("ERA5-50", "S_pi_ERA5_50", "Y_ERA5sd_50"), ("CPC-50", "S_pi_CPC_50", "Y_CPCsd_50"),
              ("CHIRPS-50", "S_pi_CHIRPS_50", "Y_CHIRPSsd_50")]
    h6, rel = [], []
    for prod, sx, yv in pairs6:
        for samp in ["440", "364", "2018", "2019"]:
            d = ns if samp == "440" else (e364 if samp == "364" else ns[ns.year == int(samp)])
            d = d.dropna(subset=[sx, yv])
            if samp == "364":
                d = drop_singletons(ns.dropna(subset=[sx, yv]), FE)
            m = pf.feols(f"{yv} ~ {sx}", data=d, vcov={"CRV1": f"city_name + {FE}"})
            tb_ = m.tidy()
            sl = tb_.loc[sx]
            ic = tb_.loc["Intercept"]
            h6.append(dict(product=prod, S=sx, outcome=yv, sample=samp, N=len(d), slope=sl.Estimate, slope_lo=sl["2.5%"], slope_hi=sl["97.5%"],
                           intercept=ic.Estimate, int_lo=ic["2.5%"], int_hi=ic["97.5%"], mean_pred=d[sx].mean(), mean_real=d[yv].mean(),
                           mean_bias=d[yv].mean() - d[sx].mean(), calibrated=bool(sl["2.5%"] <= 1 <= sl["97.5%"] and abs(d[yv].mean() - d[sx].mean()) < 0.05)))
            if samp == "440":
                dq = d.assign(dec=pd.qcut(d[sx], 10, labels=False, duplicates="drop") + 1).groupby("dec").agg(
                    mean_S=(sx, "mean"), mean_realised=(yv, "mean"), n=(sx, "size")).reset_index()
                dq["product"] = prod
                dq["outcome"] = yv
                rel.append(dq)
    pd.DataFrame(h6).to_csv(os.path.join(INTERMEDIATE, "H6_calibration.csv"), index=False, encoding=ENC)
    pd.concat(rel).to_csv(os.path.join(INTERMEDIATE, "H6_reliability_deciles.csv"), index=False, encoding=ENC)

    # H1 summary tables (main sample with year-specific checks)
    def h1tab(gname):
        a = G1[(G1.grid == gname)]
        t = a[a["sample"] == "364"].set_index(["S_product", "outcome"])[["N", "b_per0p1", "ci_lo", "ci_hi", "b_per_sd", "p_wcr", "q_ratio"]]
        for s_ in ["2018", "2019", "440"]:
            b_ = a[a["sample"] == s_].set_index(["S_product", "outcome"])
            t[f"b_{s_}"] = b_.b_per0p1
            t[f"p2w_{s_}"] = b_.p_2way
        t["mag_ratio"] = t.b_per0p1 / BASELINE_PER_0P1
        t["pass_cell"] = ((t.b_per0p1 > 0) & (t.p_wcr < 0.01) & (t.mag_ratio.between(0.65, 1.35)) & (t.q_ratio >= 2.5)
                          & (t.p2w_2018 < 0.05) & (t.p2w_2019 < 0.05))
        return t.reset_index()

    h1tab("EF").to_csv(os.path.join(INTERMEDIATE, "H1_summary_EF.csv"), index=False, encoding=ENC)
    h1tab("50mm").to_csv(os.path.join(INTERMEDIATE, "H1_summary_50mm.csv"), index=False, encoding=ENC)
    print("events", len(ns), "H1 rows", len(G1))


def hazard_scale():
    """Spatial scale of shared extreme rainfall: decay of pairwise co-exposure pi with distance (isotonic fit by log-distance
    bin) and the distances L10 / L05 at which it falls below 0.10 / 0.05, nationally, by region, bearing, season and period
    with year bootstrap intervals, for ERA5-Land and CPC; the share of partner weight within those distances; partner
    weight versus co-exposure mass by distance band; and S relative to random national and distance-band partners."""
    WEATHER = os.path.join(DATASET, "city daily weather")
    ENC = "utf-8-sig"
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

    def build_pi(Hm, mask=None):
        b = Hm if mask is None else Hm[mask]
        b = b.astype(np.float64)
        own = b.sum(0)
        co = b.T @ b
        pi = np.divide(co, own[:, None], out=np.full_like(co, np.nan), where=own[:, None] > 0)
        np.fill_diagonal(pi, np.nan)
        return pi

    def hav(lat1, lon1, lat2, lon2):
        p1, p2 = np.radians(lat1), np.radians(lat2)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(a))

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
        """Distance (km) at which the isotonic curve first falls below c, interpolated between bin centres in log distance."""
        g = np.isfinite(fit)
        x, f = ctr[g], fit[g]
        k = np.where(f < c)[0]
        if len(k) == 0:
            return np.inf
        k = k[0]
        if k == 0:
            return 0.0
        return float(np.exp(x[k - 1] + (f[k - 1] - c) * (x[k] - x[k - 1]) / (f[k - 1] - f[k])))

    # Daily extremes 1981-2016: ERA5-Land rule (EF), CPC at the same per-city frequency, and >= 50 mm for both
    era = pd.concat([pd.read_csv(os.path.join(WEATHER, f"city_daily_weather_{y}.csv"), encoding=ENC, usecols=["city_name", "date", "rain_mm"])
                     for y in range(1981, 2017)], ignore_index=True)
    era["city_name"] = era.city_name.str.strip()
    era["date"] = pd.to_datetime(era.date)
    cpc = pd.read_parquet(os.path.join(INTERMEDIATE, "city_daily_cpc_1981_2019.parquet"))
    hp = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    cyn = pd.read_csv(os.path.join(INTERMEDIATE, "city_year_network.csv"), encoding=ENC)
    pn = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC)
    th = pd.read_csv(os.path.join(INTERMEDIATE, "city_rainfall_thresholds.csv"), encoding=ENC)
    th["city_name"] = th.city_name.astype(str).str.strip()
    cut = dict(zip(th.city_name, th.return_level_3yr_mm.fillna(np.inf)))
    ew = era.pivot(index="date", columns="city_name", values="rain_mm").sort_index()
    cities = list(ew.columns)
    c_ = cpc.copy()
    c_["date"] = pd.to_datetime(c_.date)
    c_ = c_[(c_.date >= "1981-01-01") & (c_.date <= "2016-12-31")]
    cw = c_.pivot(index="date", columns="city_name", values="rain_mm").reindex(index=ew.index, columns=cities)
    E = ew.to_numpy()
    C = cw.to_numpy()
    cutv = np.array([cut.get(c, np.inf) for c in cities])
    H = {}
    H["ERA5-EF"] = ((E >= 50) | (E >= cutv[None, :]))
    n_i = H["ERA5-EF"].sum(0)
    Cf = np.where(np.isnan(C), -np.inf, C)
    u = np.array([np.sort(Cf[:, k])[::-1][n_i[k] - 1] if n_i[k] > 0 else np.inf for k in range(len(cities))])
    H["CPC-EF"] = (Cf >= u[None, :])
    H["ERA5-50"] = (E >= 50)
    H["CPC-50"] = (Cf >= 50)
    dates = pd.to_datetime(ew.index)
    yrs = dates.year.to_numpy()
    mon = dates.month.to_numpy()
    ix = {c: i for i, c in enumerate(cities)}

    # Grid-weighted city centres, distances and bearings
    grid = pd.read_parquet(os.path.join(INTERMEDIATE, "city_grid_weights.parquet"))
    grid["city_name"] = grid.city_name.astype(str).str.strip()
    cc = grid.assign(wlat=grid.latitude * grid.weight, wlon=grid.longitude * grid.weight).groupby("city_name")[["wlat", "wlon", "weight"]].sum()
    lat = (cc.wlat / cc.weight).reindex(cities).to_numpy()
    lon = (cc.wlon / cc.weight).reindex(cities).to_numpy()
    D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    p1, p2 = np.radians(lat[:, None]), np.radians(lat[None, :])
    dl = np.radians(lon[None, :] - lon[:, None])
    BR = (np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))) + 360) % 360
    SECT = ((BR + 22.5) // 45).astype(int) % 8
    ld = np.log(np.clip(hp.distance_km, 5, None))
    edges = np.unique(np.quantile(ld, np.linspace(0, 1, 61)))

    # Regions for all 369 units: province of the city list where available, else majority of Queen neighbours,
    # else the nearest listed city, with manual corrections
    xlsx = [f for f in os.listdir(DATASET) if f.endswith(".xlsx")][0]
    dic = pd.read_excel(os.path.join(DATASET, xlsx), sheet_name=5).dropna()
    dic.columns = ["province", "city_name", "city_id"]
    dic["city_name"] = dic.city_name.astype(str).str.strip()
    dic["province"] = dic.province.astype(str).str.strip()
    dprov = dict(zip(dic.city_name, dic.province))
    pr7 = pd.read_csv(os.path.join(EXTERNAL, "province_region7.csv"), encoding=ENC)
    p2group = dict(zip(pr7.province, pr7.region_group))
    reg = {c: p2group[dprov[c]] for c in cities if c in dprov}
    qe = queen_edges()
    nb = {c: set() for c in cities}
    for a, b in qe[["a", "b"]].itertuples(index=False):
        if a in nb and b in nb:
            nb[a].add(b)
            nb[b].add(a)
    src = {c: "dictionary" for c in reg}
    for it in range(50):
        new = {}
        for c in cities:
            if c in reg:
                continue
            lab = [reg[n] for n in nb[c] if n in reg]
            if lab:
                vc = pd.Series(lab).value_counts()
                if len(vc) == 1 or vc.iloc[0] > vc.iloc[1]:
                    new[c] = vc.index[0]
        if not new:
            break
        for c, v in new.items():
            reg[c] = v
            src[c] = f"queen_majority_iter{it + 1}"
    for c in [c for c in cities if c not in reg]:
        i = ix[c]
        lab = [k for k in reg if src[k] == "dictionary"]
        j = min(lab, key=lambda k: D[i, ix[k]])
        reg[c] = reg[j]
        src[c] = "nearest_dictionary_centroid"
    fix = pd.read_csv(os.path.join(EXTERNAL, "unit_region_corrections.csv"), encoding=ENC)
    for c, v in zip(fix.city_name, fix.region):
        reg[c] = v
    REGS = ["South", "East", "North", "Inland"]
    regv = np.array([reg[c] for c in cities])

    # Yearly co-occurrence counts by product and season
    VARS = ["ERA5-EF", "CPC-EF", "ERA5-50", "CPC-50"]
    SEAS = {"all": np.ones(len(dates), bool), "AMJ": np.isin(mon, [4, 5, 6]), "JAS": np.isin(mon, [7, 8, 9])}
    YRS = np.arange(1981, 2017)
    Cy, Oy = {}, {}
    for v in VARS:
        Hv = H[v].astype(np.float32)
        for s, m in SEAS.items():
            Ca = np.zeros((36, 369, 369), np.float32)
            Oa = np.zeros((36, 369), np.float32)
            for yi, y in enumerate(YRS):
                b = Hv[(yrs == y) & m]
                Ca[yi] = b.T @ b
                Oa[yi] = b.sum(0)
            Cy[(v, s)] = Ca
            Oy[(v, s)] = Oa
    offd = ~np.eye(369, dtype=bool)
    I, J = np.where(offd)
    LD = np.log(np.clip(D[I, J], 5, None))

    def mkbins(sel):
        ldv = LD[sel]
        n = sel.sum() if sel.dtype == bool else len(sel)
        nbins = int(min(60, n // 150))
        ed = np.unique(np.quantile(ldv, np.linspace(0, 1, nbins + 1)))
        b = np.clip(np.searchsorted(ed, ldv, side="right") - 1, 0, len(ed) - 2)
        return dict(idx=np.where(sel)[0], b=b, nb=len(ed) - 1, ctr=np.bincount(b, weights=ldv, minlength=len(ed) - 1) / np.bincount(b, minlength=len(ed) - 1))

    SUB = {"national": mkbins(np.ones(len(I), bool))}
    for r in REGS:
        SUB["region:" + r] = mkbins(regv[I] == r)
    SNAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    for k in range(8):
        SUB["bearing:" + SNAMES[k]] = mkbins(SECT[I, J] == k)

    def curve(pivec, sb):
        x = pivec[sb["idx"]]
        ok = np.isfinite(x)
        b = sb["b"]
        n = np.bincount(b[ok], minlength=sb["nb"]).astype(float)
        s = np.bincount(b[ok], weights=x[ok], minlength=sb["nb"])
        mean = np.divide(s, n, out=np.full(sb["nb"], np.nan), where=n > 0)
        g = n > 0
        fit = np.full(sb["nb"], np.nan)
        fit[g] = pava_dec(mean[g], n[g])
        return sb["ctr"], mean, fit, n

    def pi_from(Ca, Oa, k=None):
        co = Ca.sum(0) if k is None else np.tensordot(k, Ca, axes=(0, 0))
        own = Oa.sum(0) if k is None else k @ Oa
        with np.errstate(invalid="ignore", divide="ignore"):
            p = co[I, J] / own[I]
        p[own[I] <= 0] = np.nan
        return p

    # L10 / L05 with year bootstrap (999 draws)
    PER = {"1981-2016": np.arange(36), "1981-1998": np.arange(18), "1999-2016": np.arange(18, 36)}

    def cells_for(season):
        return list(SUB.keys()) if season == "all" else ["national"] + ["region:" + r for r in REGS]

    rng = np.random.default_rng(20261023)
    NB = 999
    point, curves, boot = [], [], {}
    for per, yix in PER.items():
        kd = np.zeros((NB, 36), np.float32)
        for bdraw in range(NB):
            dr = rng.choice(yix, size=len(yix), replace=True)
            kd[bdraw] = np.bincount(dr, minlength=36)
        k0 = np.zeros(36, np.float32)
        k0[yix] = 1
        for v in VARS:
            for s in SEAS:
                Ca, Oa = Cy[(v, s)], Oy[(v, s)]
                pv = pi_from(Ca, Oa, k0)
                for cell in cells_for(s):
                    ctr, mean, fit, n = curve(pv, SUB[cell])
                    point.append(dict(product=v, period=per, season=s, subset=cell, L10=Lcross(ctr, fit, .10), L05=Lcross(ctr, fit, .05), n_pairs=int(n.sum()), pi_first_bin=fit[0]))
                    curves.append(pd.DataFrame(dict(product=v, period=per, season=s, subset=cell, bin=np.arange(len(ctr)), distance_km=np.exp(ctr), pi_mean=mean, pi_iso=fit, n_pairs=n)))
                Bm = np.full((NB, len(cells_for(s)), 2), np.nan)
                for bdraw in range(NB):
                    pv = pi_from(Ca, Oa, kd[bdraw])
                    for ci, cell in enumerate(cells_for(s)):
                        ctr, mean, fit, n = curve(pv, SUB[cell])
                        Bm[bdraw, ci] = [Lcross(ctr, fit, .10), Lcross(ctr, fit, .05)]
                boot[(per, v, s)] = Bm
    PT = pd.DataFrame(point)
    CV = pd.concat(curves)
    rows = []
    for (per, v, s), Bm in boot.items():
        for ci, cell in enumerate(cells_for(s)):
            r = PT[(PT["product"] == v) & (PT.period == per) & (PT.season == s) & (PT.subset == cell)].iloc[0].to_dict()
            for j, L in enumerate(["L10", "L05"]):
                x = Bm[:, ci, j]
                fin = np.isfinite(x) & (x > 0)
                r[L + "_lo"], r[L + "_hi"] = np.percentile(x[fin], [2.5, 97.5]) if fin.sum() > 0 else (np.nan, np.nan)
                r[L + "_boot_out_of_range"] = int((~fin).sum())
            rows.append(r)
    T1 = pd.DataFrame(rows)
    rat = []
    for per in PER:
        for s in SEAS:
            for pair in [("CPC-EF", "ERA5-EF"), ("CPC-50", "ERA5-50")]:
                Bc, Be = boot[(per, pair[0], s)], boot[(per, pair[1], s)]
                for ci, cell in enumerate(cells_for(s)):
                    pc = T1[(T1["product"] == pair[0]) & (T1.period == per) & (T1.season == s) & (T1.subset == cell)].iloc[0]
                    pe_ = T1[(T1["product"] == pair[1]) & (T1.period == per) & (T1.season == s) & (T1.subset == cell)].iloc[0]
                    for j, L in enumerate(["L10", "L05"]):
                        rr = Bc[:, ci, j] / Be[:, ci, j]
                        fin = np.isfinite(rr) & (rr > 0)
                        rat.append(dict(period=per, season=s, subset=cell, threshold=pair[0][-2:], metric=L, ERA5=pe_[L], CPC=pc[L], ratio=pc[L] / pe_[L],
                                        ratio_lo=np.percentile(rr[fin], 2.5), ratio_hi=np.percentile(rr[fin], 97.5), within15=abs(pc[L] / pe_[L] - 1) <= 0.15))
    T1.to_csv(os.path.join(INTERMEDIATE, "T1_decorrelation_L10_L05.csv"), index=False, encoding=ENC)
    pd.DataFrame(rat).to_csv(os.path.join(INTERMEDIATE, "T1_product_ratio_CPC_over_ERA5.csv"), index=False, encoding=ENC)
    CV.to_csv(os.path.join(INTERMEDIATE, "T1_decay_curves_binned.csv"), index=False, encoding=ENC)

    # Share of partner weight within L
    pn["hi"] = pn.host_name.map(ix)
    pn["gi"] = pn.guest_name.map(ix)
    u_ = pn[pn.usable == True].copy()
    u_["wt"] = u_.w / u_.groupby(["host_name", "network_year"]).w.transform("sum")
    pi_my = {v: build_pi(H[v]) for v in VARS}
    for v in VARS:
        u_["pi_" + v] = pi_my[v][u_.hi.astype(int), u_.gi.astype(int)]
    host_reg = u_.host_name.map(reg)
    nat = T1[(T1.period == "1981-2016") & (T1.season == "all") & (T1.subset == "national")].set_index("product")
    L05 = {v: float(nat.loc[v, "L05"]) for v in VARS}
    L10 = {v: float(nat.loc[v, "L10"]) for v in VARS}
    LREG = {v: {r: float(T1[(T1["product"] == v) & (T1.period == "1981-2016") & (T1.season == "all") & (T1.subset == "region:" + r)].L05.iloc[0]) for r in REGS} for v in VARS}

    def share(thr):
        return u_.assign(x=u_.wt * (u_.distance_km <= thr)).groupby(["host_name", "network_year"]).x.sum()

    res = []

    def summ(sv, label, v, Lval):
        for samp, yrsel in [("672 (2017-2019)", [2017, 2018, 2019]), ("448 (2017-2018)", [2017, 2018])]:
            x = sv[sv.index.get_level_values(1).isin(yrsel)]
            res.append(dict(version=label, product=v, L_km=Lval, sample=samp, N=len(x), median=x.median(), p10=x.quantile(.1), p90=x.quantile(.9), mean=x.mean()))

    for v in VARS:
        summ(share(L05[v]), "national L05", v, L05[v])
        summ(share(host_reg.map(LREG[v]).to_numpy()), "regional L05", v, "host region")
        summ(share(L10[v]), "national L10", v, L10[v])
        summ(share(nat.loc[v, "L05_lo"]), "national L05 lower CI", v, nat.loc[v, "L05_lo"])
        summ(share(nat.loc[v, "L05_hi"]), "national L05 upper CI", v, nat.loc[v, "L05_hi"])
    pd.DataFrame(res).to_csv(os.path.join(INTERMEDIATE, "T2_weight_share_inside_L.csv"), index=False, encoding=ENC)

    # Partner weight versus co-exposure mass by distance band (host bootstrap, 999 draws)
    BANDS = [0, 100, 200, 300, 600, np.inf]
    BL = ["0-100", "100-200", "200-300", "300-600", ">600"]
    u_["band"] = pd.cut(u_.distance_km, BANDS, right=True, labels=BL, include_lowest=True)
    rng = np.random.default_rng(20261023)
    out3 = []
    for v in VARS:
        d = u_[np.isfinite(u_["pi_" + v])].copy()
        d["wv"] = d.w / d.groupby(["host_name", "network_year"]).w.transform("sum")
        d["m"] = d.wv * d["pi_" + v]
        for samp, yrsel in [("672 (2017-2019)", [2017, 2018, 2019]), ("448 (2017-2018)", [2017, 2018])]:
            dd = d[d.network_year.isin(yrsel)]
            Wb = dd.pivot_table(index=["host_name", "network_year"], columns="band", values="wv", aggfunc="sum", observed=False).reindex(columns=BL).fillna(0)
            Mb = dd.pivot_table(index=["host_name", "network_year"], columns="band", values="m", aggfunc="sum", observed=False).reindex(columns=BL).fillna(0)
            hosts = Wb.index.get_level_values(0).to_numpy()
            uh = np.unique(hosts)
            hcode = np.searchsorted(uh, hosts)
            Wh = np.zeros((len(uh), 5))
            np.add.at(Wh, hcode, Wb.to_numpy())
            Mh = np.zeros((len(uh), 5))
            np.add.at(Mh, hcode, Mb.to_numpy())
            nh = np.bincount(hcode)

            def stat(cnt):
                W = (cnt[:, None] * Wh).sum(0) / (cnt * nh).sum()
                M = (cnt[:, None] * Mh).sum(0)
                return W, M / M.sum()

            W0, M0 = stat(np.ones(len(uh)))
            BW, BM = [], []
            for b in range(999):
                W, M = stat(np.bincount(rng.integers(0, len(uh), len(uh)), minlength=len(uh)).astype(float))
                BW.append(W)
                BM.append(M)
            BW, BM = np.array(BW), np.array(BM)
            cw_ = Wb.div(Wb.sum(1), axis=0)
            cm_ = Mb.div(Mb.sum(1), axis=0)
            for k, bl in enumerate(BL + ["<=300 (cum)"]):
                if k < 5:
                    w0, m0, bw, bm = W0[k], M0[k], BW[:, k], BM[:, k]
                    medw, medm = cw_[bl].median(), cm_[bl].median()
                else:
                    w0, m0, bw, bm = W0[:3].sum(), M0[:3].sum(), BW[:, :3].sum(1), BM[:, :3].sum(1)
                    medw, medm = cw_[BL[:3]].sum(1).median(), cm_[BL[:3]].sum(1).median()
                out3.append(dict(pi_product=v, sample=samp, band_km=bl, N_cityyears=len(Wb), weight_share=w0, weight_lo=np.percentile(bw, 2.5),
                                 weight_hi=np.percentile(bw, 97.5), mass_share=m0, mass_lo=np.percentile(bm, 2.5), mass_hi=np.percentile(bm, 97.5),
                                 median_cy_weight_share=medw, median_cy_mass_share=medm))
    pd.DataFrame(out3).to_csv(os.path.join(INTERMEDIATE, "T3_distance_band_weight_vs_mass.csv"), index=False, encoding=ENC)

    # S relative to the mean co-exposure of all hosts (E_nat) and of hosts in the same distance band (E_band)
    hosts224 = sorted(cyn.city_name.unique())
    hidx = np.array([ix[c] for c in hosts224])

    def band_of(dkm):
        return np.clip(np.searchsorted(edges, np.log(np.clip(dkm, 5, None)), side="right") - 1, 0, len(edges) - 2)

    NBAND = len(edges) - 1
    out4 = []
    for v in VARS:
        Pm = pi_my[v]
        d = u_[np.isfinite(u_["pi_" + v]) & u_.network_year.isin([2017, 2018])].copy()
        d["wv"] = d.w / d.groupby(["host_name", "network_year"]).w.transform("sum")
        Enat, Eb = {}, {}
        for i in np.unique(d.hi.astype(int)):
            kk = hidx[(hidx != i)]
            pk = Pm[i, kk]
            ok = np.isfinite(pk)
            kk, pk = kk[ok], pk[ok]
            Enat[i] = pk.mean()
            bk = band_of(D[i, kk])
            s = np.bincount(bk, weights=pk, minlength=NBAND)
            n = np.bincount(bk, minlength=NBAND)
            ne = np.where(n > 0)[0]
            mb = np.full(NBAND, np.nan)
            mb[ne] = s[ne] / n[ne]
            for b in range(NBAND):
                if n[b] == 0:
                    mb[b] = mb[ne[np.argmin(np.abs(ne - b))]]
            Eb[i] = mb
        d["bj"] = band_of(d.distance_km.to_numpy())
        d["eb"] = [Eb[int(i)][b] for i, b in zip(d.hi, d.bj)]
        g = d.assign(S=d.wv * d["pi_" + v], EB=d.wv * d.eb).groupby(["host_name", "network_year"]).agg(S=("S", "sum"), E_band=("EB", "sum"), hi=("hi", "first"))
        g["E_nat"] = g.hi.astype(int).map(Enat)
        g["product"] = v
        g["ratio"] = g.S / g.E_nat
        g["geo_share"] = (g.E_band - g.E_nat) / (g.S - g.E_nat)
        out4.append(g.reset_index())
    T4cy = pd.concat(out4)
    T4cy.to_csv(os.path.join(INTERMEDIATE, "T4_B4_cityyear_products.csv"), index=False, encoding=ENC)
    S4 = T4cy.groupby("product").apply(lambda g: pd.Series(dict(
        N=len(g), median_S=g.S.median(), median_E_nat=g.E_nat.median(), ratio_of_medians=g.S.median() / g.E_nat.median(),
        median_of_ratios=g.ratio.median(), aggregate_ratio=g.S.sum() / g.E_nat.sum(), median_E_band=g.E_band.median(),
        median_geo_share=g.geo_share.median(), aggregate_geo_share=(g.E_band.sum() - g.E_nat.sum()) / (g.S.sum() - g.E_nat.sum())))).reindex(VARS)
    S4["PASS_ge4"] = S4.ratio_of_medians >= 4
    S4.to_csv(os.path.join(INTERMEDIATE, "T4_B4_ratio_summary.csv"), encoding=ENC)
    print("national L05", L05)


def example_networks():
    """Example partner networks of Guangzhou and Xi'an (2018): shared-rain exposure S, share of inbound demand within
    300 km and mean co-extreme probability of the cities within 300 km."""
    ENC = "utf-8-sig"
    EXAMPLES = {"Guangzhou": "\u5e7f\u5dde\u5e02", "Xi'an": "\u897f\u5b89\u5e02"}

    net = pd.read_csv(os.path.join(INTERMEDIATE, "partner_network.csv"), encoding=ENC, dtype={"host_id": str, "guest_id": str}, low_memory=False)
    pairs = pd.read_csv(os.path.join(INTERMEDIATE, "hazard_pairs.csv"), encoding=ENC)
    chk = []
    for en, cn in EXAMPLES.items():
        q = net.loc[(net.host_name == cn) & (net.network_year == 2018)]
        us = q.loc[q.usable == 1]
        near = pairs.loc[(pairs.focal_city == cn) & (pairs.distance_km <= 300)]
        chk.append({"city": en, "network_year": 2018, "n_partner_cities_usable": len(us), "S": float((us.w * us.pi).sum() / us.w.sum()),
                    "share_demand_within_300km": float(us.w[us.distance_km <= 300].sum() / us.w.sum()), "n_cities_within_300km": len(near),
                    "mean_cooccurrence_chance_within_300km": float(near.pi.mean())})
    pd.DataFrame(chk).to_csv(os.path.join(INTERMEDIATE, "figS2c_examples_check.csv"), index=False)
    print(pd.DataFrame(chk).to_string(index=False))


def main():
    shared_rain_validation()
    hazard_scale()
    example_networks()


if __name__ == "__main__":
    main()
