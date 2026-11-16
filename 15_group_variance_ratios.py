"""Group variance ratio (synchrony of spending shortfalls) by grouping, weighting and group size n = 2-20 with year-month block-bootstrap intervals, and within-grouping slopes of the variance ratio on group S (Supplementary results supporting Fig. 5). Writes SI4_group_VR_intervals.csv and SI4_group_slopes.csv. Run time about 20-35 min."""
from pathlib import Path
import warnings
import pickle
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
INTERMEDIATE = ROOT / "Intermediate"
ENC = "utf-8-sig"


def need(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file missing: {path}")
    return path


VR_SEED, VR_NB = 20261110, 999
TYPES = ["cluster", "province", "random", "demand", "crossbelt"]
DESIGN_NAME = {"cluster": "City cluster", "province": "Province", "random": "National random",
          "demand": "Demand partners", "crossbelt": "Cross-rain-belt"}
ARRNAME = {"cluster": "Within city cluster", "province": "Within province", "random": "Nationwide, unselected",
           "demand": "Existing demand ties", "crossbelt": "Beyond the rainfall system"}


class Mom:
    """Per-block sums of L and L'L, so that the covariance matrix can be recomputed for block-bootstrap weights."""
    def __init__(self, L, block):
        ub = np.unique(block)
        bid = np.searchsorted(ub, block)
        self.B = len(ub)
        self.n = np.bincount(bid, minlength=self.B).astype(float)
        self.S1 = np.stack([L[bid == b].sum(0) for b in range(self.B)])
        self.S2 = np.stack([L[bid == b].T @ L[bid == b] for b in range(self.B)])

    def cov(self, k=None):
        k = np.ones(self.B) if k is None else k
        n = k @ self.n
        m = (k @ self.S1) / n
        return np.tensordot(k, self.S2, 1) / n - np.outer(m, m)


def prep(pool, N, nmax):
    M = pool["M"][:, :nmax]
    d = pool["draws"]
    size = np.repeat(np.minimum(pool["size"], nmax), d)
    M0 = np.where(M >= 0, M, 0)
    pa = np.array([a for b in range(1, nmax) for a in range(b)])
    pb = np.array([b for b in range(1, nmax) for a in range(b)])
    uf = np.unique(pool["focal"])
    RF = (pool["focal"][:, None] == uf[None, :]).astype(float)
    return dict(idx=(M0[:, pa] * N + M0[:, pb]).T.astype(np.int64).copy(), didx=(M0 * N + M0).T.astype(np.int64).copy(),
                M0T=M0.T.copy(), pa=pa, pb=pb, size=size, nmax=nmax, P=len(M), R=len(pool["focal"]), draws=d,
                focal=uf, RF=RF)


def nested_vr(C, pp, w=None):
    """VR for groups of the first 2..nmax members (nested groups); array (nmax-1, P)."""
    Cf = C.ravel()
    X = Cf.take(pp["idx"])
    Dg = Cf.take(pp["didx"])
    if w is not None:
        wm = w.take(pp["M0T"])
        X = X * wm[pp["pa"]] * wm[pp["pb"]]
        Dg = Dg * wm ** 2
    S = Dg[0].copy()
    Dd = Dg[0].copy()
    pos = 0
    VR = np.full((pp["nmax"] - 1, pp["P"]), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for b in range(1, pp["nmax"]):
            S = S + Dg[b] + 2 * X[pos:pos + b].sum(0)
            pos += b
            Dd = Dd + Dg[b]
            ok = pp["size"] >= b + 1
            VR[b - 1] = np.where(ok & (Dd > 0), np.sqrt(np.clip(S, 0, None)) / np.sqrt(Dd), np.nan)
    return VR


def to_focal(x, pp):
    nn = x.shape[0]
    xr = x.reshape(nn, pp["R"], pp["draws"])
    ok = np.isfinite(xr)
    cnt = ok.sum(2)
    rowm = np.where(cnt > 0, np.where(ok, xr, 0).sum(2) / np.maximum(cnt, 1), np.nan)
    okr = np.isfinite(rowm)
    num_ = np.where(okr, rowm, 0) @ pp["RF"]
    den = okr.astype(float) @ pp["RF"]
    return np.where(den > 0, num_ / np.maximum(den, 1), np.nan)


def med(x):
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def stats(C, PP, w):
    out = {}
    for t in TYPES:
        fo = to_focal(nested_vr(C, PP[t], w), PP[t])
        for n in range(2, PP[t]["nmax"] + 1):
            x = fo[n - 2]
            out[(t, n)] = (med(x), float(np.nanmean(x)) if np.isfinite(x).any() else np.nan, int(np.isfinite(x).sum()))
    return out


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.quantile(x, .025)), float(np.quantile(x, .975))) if len(x) else (np.nan, np.nan)


def group_vr_intervals():
    t0 = time.time()
    with open(need(INTERMEDIATE / "t4_inputs.pkl"), "rb") as f:
        I = pickle.load(f)
    L, block, POOLS, WSP = I["LOSS"]["main"], I["block"], I["POOLS"], I["WSP"]
    T, N = L.shape
    B = len(np.unique(block))
    assert (T, N, B) == (1095, 224, 36)
    KB = np.random.default_rng([VR_SEED, 1, B]).multinomial(B, np.full(B, 1 / B), size=VR_NB)   # block multiplicities per draw
    mom = Mom(L, block)
    PP = {t: prep(POOLS[t], N, 20) for t in TYPES}
    rows = []
    for wname, w in (("equal", None), ("spending", WSP)):
        pt = stats(mom.cov(), PP, w)
        boot = {k: np.empty(VR_NB) for k in pt}
        for r, k in enumerate(KB):
            s = stats(mom.cov(k.astype(float)), PP, w)
            for key in pt:
                boot[key][r] = s[key][0]
            if r % 100 == 0:
                print(f"  {wname}: draw {r} ({time.time() - t0:.0f} s)", flush=True)
        for (t, n), (m, mn, nf) in pt.items():
            lo, hi = ci(boot[(t, n)])
            rows.append({"weighting": wname, "T4_design": t, "arrangement": ARRNAME[t], "n": n, "N_focal": nf,
                         "VR_median": m, "VR_ci_lo": lo, "VR_ci_hi": hi, "VR_mean": mn,
                         "n_boot_finite": int(np.isfinite(boot[(t, n)]).sum())})
    out = pd.DataFrame(rows)
    sc = pd.read_csv(need(INTERMEDIATE / "T4_size_span_curve.csv"))
    e = out.loc[out.weighting == "equal"].copy()
    e["pool_type"] = e.T4_design.map(DESIGN_NAME)
    m = e.merge(sc, on=["pool_type", "n"], suffixes=("", "_T4"))
    assert len(m) == 95, len(m)
    d_eq = max((m.VR_median - m.VR_median_T4).abs().max(), (m.VR_ci_lo - m.VR_median_ci_lo).abs().max(),
               (m.VR_ci_hi - m.VR_median_ci_hi).abs().max())
    assert d_eq < 1e-9, f"equal-weight curve not reproduced: {d_eq}"
    ps = pd.read_csv(need(INTERMEDIATE / "T4_pool_statistics_main.csv"))
    ps = ps.loc[(ps.weights == "spending") & (ps.n == 5) & (ps.statistic == "VR")]
    s5 = out.loc[(out.weighting == "spending") & (out.n == 5)].copy()
    s5["pool_type"] = s5.T4_design.map(DESIGN_NAME)
    m2 = s5.merge(ps, on="pool_type", suffixes=("", "_T4"))
    assert len(m2) == 5
    d_sp = max((m2.VR_median - m2["median"]).abs().max(), (m2.VR_ci_lo - m2.median_ci_lo).abs().max(),
               (m2.VR_ci_hi - m2.median_ci_hi).abs().max())
    assert d_sp < 1e-9, f"spending n = 5 not reproduced: {d_sp}"
    n_mis = int((m.N_focal != m.N_focal_T4).sum() + (m2.N_focal != m2.N_focal_T4).sum())
    print(f"  N_focal mismatches against the saved curves (they count focal cities with a finite risk ratio): {n_mis}")
    out["check"] = np.where(out.weighting.eq("equal"), f"reproduces T4_size_span_curve.csv (max |diff| {d_eq:.1e})",
                            np.where(out.n.eq(5), f"reproduces T4_pool_statistics_main.csv (max |diff| {d_sp:.1e})",
                                     "new (same engine and draws)"))
    out.to_csv(INTERMEDIATE / "SI4_group_VR_intervals.csv", index=False, encoding=ENC)
    print(f"  run time {time.time() - t0:.0f} s")


Z = float(norm.ppf(0.975))
SLOPE_SEED, SLOPE_NB = 20261112, 999
ORDER = ["Within province", "Within city cluster", "Existing demand ties", "Nationwide, unselected",
         "Beyond the rainfall system"]
SV = {"pi 1981-2016": "", "pi_early 1981-1998 (out-of-sample)": "_early"}
FE_NAME = "All five T4 arrangements, arrangement fixed effects"


def ols(x, y):
    x = x - x.mean()
    return (x * (y - y.mean())).sum() / (x * x).sum()


def fe_slope(d, scol, vcol):
    """VR on S with grouping dummies; CR1 SE clustered by focal city."""
    X = np.column_stack([d[scol].to_numpy()] + [(d.arrangement == a).astype(float).to_numpy()
                                                 for a in sorted(d.arrangement.unique())])
    y = d[vcol].to_numpy()
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    cl = d.city.to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cl):
        s = X[cl == c].T @ e[cl == c]
        meat += np.outer(s, s)
    G, (Nn, K) = len(np.unique(cl)), X.shape
    se = np.sqrt((XtXi @ meat @ XtXi)[0, 0] * G / (G - 1) * (Nn - 1) / (Nn - K))
    return 0.1 * b[0], 0.1 * (b[0] - Z * se), 0.1 * (b[0] + Z * se), Nn


def group_slopes():
    gv = pd.read_csv(need(INTERMEDIATE / "fig5c_group_S_VR_by_city.csv"), encoding=ENC)
    ws = pd.read_csv(need(INTERMEDIATE / "fig5c_within_slopes.csv"), encoding=ENC)
    ew = pd.read_csv(need(INTERMEDIATE / "fig5_eastwest_groups_VR.csv"), encoding=ENC)
    rng = np.random.default_rng(SLOPE_SEED)

    def data(arr, w, n, sv):
        col = f"S_{w}{SV[sv]}"
        if arr.startswith("East"):
            d = ew[[col, f"VR_{w}"]].dropna()
        else:
            d = gv.loc[(gv.arrangement == arr) & (gv.n == int(n)), [col, f"VR_{w}"]].dropna()
        return d[col].to_numpy(), d[f"VR_{w}"].to_numpy()

    def one(arr, w, n, sv):
        x, y = data(arr, w, n, sv)
        z = float((x == 0).mean())
        if z >= 0.9:
            return dict(N_focal=len(x), share_S_zero=z, status=f"not estimable: S = 0 for {100 * z:.0f}% of focal cities")
        bs = np.empty(SLOPE_NB)
        for r in range(SLOPE_NB):
            i = rng.integers(0, len(x), len(x))
            bs[r] = ols(x[i], y[i]) * 0.1
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rho, p = spearmanr(x, y)
        return dict(N_focal=len(x), share_S_zero=z, status="estimated", slope_per_0p1S=ols(x, y) * 0.1, ci_lo=lo,
                    ci_hi=hi, spearman=rho, spearman_p=p)

    def fe(w, n, sv):
        col = f"S_{w}{SV[sv]}"
        d = gv.loc[(gv.n == int(n)) & gv.arrangement.isin(ORDER), ["city", "arrangement", col, f"VR_{w}"]].dropna()
        b, lo, hi, Nn = fe_slope(d, col, f"VR_{w}")
        return dict(N_focal=Nn, status="estimated", slope_per_0p1S=b, ci_lo=lo, ci_hi=hi)

    rows = []
    for r in ws.itertuples():
        if r.arrangement == FE_NAME:
            o = fe(r.weighting, r.n, r.S_version)
        elif str(r.status).startswith("not"):
            x, _ = data(r.arrangement, r.weighting, r.n, r.S_version)
            o = dict(N_focal=len(x), share_S_zero=float((x == 0).mean()), status=r.status)
        else:
            o = one(r.arrangement, r.weighting, r.n, r.S_version)
        rows.append({"arrangement": r.arrangement, "weighting": r.weighting, "n": r.n, "S_version": r.S_version, **o,
                     "source": "fig5c_within_slopes.csv row reproduced"})
    rep = pd.DataFrame(rows)
    dif = {c: float(np.nanmax(np.abs(rep[c].to_numpy(float) - ws[c].to_numpy(float))))
           for c in ("slope_per_0p1S", "ci_lo", "ci_hi", "spearman")}
    print("max |difference| from fig5c_within_slopes.csv:", dif)
    # step 2: n = 2 rows, same stream continued
    new = []
    for w in ("spending", "equal"):
        for sv in SV:
            for arr in ORDER:
                new.append({"arrangement": arr, "weighting": w, "n": "2", "S_version": sv, **one(arr, w, 2, sv),
                            "source": "new (n = 2)"})
            new.append({"arrangement": FE_NAME, "weighting": w, "n": "2", "S_version": sv, **fe(w, 2, sv),
                        "source": "new (n = 2)"})
    out = pd.concat([rep, pd.DataFrame(new)], ignore_index=True)
    out["CI_method"] = np.where(out.arrangement.eq(FE_NAME), "CR1 SE clustered by focal city, +/- 1.95996 SE",
                                "999 focal-city bootstrap draws (seed 20261112), percentile")
    out.to_csv(INTERMEDIATE / "SI4_group_slopes.csv", index=False, encoding=ENC)
    print(pd.DataFrame(new)[["arrangement", "weighting", "S_version", "N_focal", "share_S_zero", "status",
                             "slope_per_0p1S", "ci_lo", "ci_hi"]].round(4).to_string())


def main():
    t0 = time.time()
    print("== group_vr_intervals", flush=True)
    group_vr_intervals()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)
    print("== group_slopes", flush=True)
    group_slopes()
    print(f"   done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
