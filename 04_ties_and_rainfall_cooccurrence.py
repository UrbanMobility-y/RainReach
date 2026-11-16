"""Intercity ties and the co-occurrence of extreme rainfall, and the event table that combines them with spending.

pi_ij is the share of city i's extreme-rain days in 1981-2016 on which city j also had extreme rain (ERA5-Land, and
CPC and CHIRPS aggregated to the same units), partner weights are annual inbound spending by origin city, and
HDC = 1 - sum_j w_j pi_ij / coverage over partners with defined pi.
Each event row holds its spending shocks (event-day mean of 100 (Y / Y0 - 1) minus the clean pre-event mean over days
-14..-1), the previous year's network measures and the partner weight hit by the same rainfall system.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyfixest as pf
import shapely
import xarray as xr
from scipy import sparse

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
DATASET = ROOT / "Dataset"
EXTERNAL = ROOT / "External data"
INTERMEDIATE = ROOT / "Intermediate"

SHAPEFILE = DATASET / "GIS" / "Prefecture City.shp"
INBOUND_NETWORK = DATASET / "城市关联度指数_副城市流向主城市.csv"     # provider file: inbound flows by origin city
DICTIONARY = DATASET / "城市活力指数数据说明附件.xlsx"                # provider data dictionary; sheet 6 lists city codes
THRESHOLDS = INTERMEDIATE / "city_rainfall_thresholds.csv"
GRID_WEIGHTS = INTERMEDIATE / "city_grid_weights.parquet"
DAILY_PANEL = INTERMEDIATE / "unionpay_city_daily_2017_2019.csv"
CATALOGUE = INTERMEDIATE / "extreme_rainfall_events_2017_2019.csv"
CF_PANEL = INTERMEDIATE / "counterfactual_panel.csv"
PARTNER_NETWORK = INTERMEDIATE / "partner_network.csv"
CITY_YEAR_NETWORK = INTERMEDIATE / "city_year_network.csv"
EVENT_TABLE = INTERMEDIATE / "events.csv"

ENC = "utf-8-sig"
HISTORY_YEARS = range(1981, 2017)
EARLY, LATE = (1981, 1998), (1999, 2016)
NETWORK_YEARS = (2017, 2018, 2019)
HEAVY_MM = 50.0
MAX_GAP_DAYS = 1
NEAR_KM = 300.0
KERNEL_BINS = 60

# alternative rainfall products
PRODUCTS = {"cpc": (EXTERNAL / "CPC", "precip.{year}.nc"),
            "chirps": (EXTERNAL / "CHIRPS", "chirps-v2.0.{year}.days_p25.nc")}
PRODUCT_YEARS = range(1981, 2020)
EQUAL_AREA_CRS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
MIN_COVER = 0.5                              # unit-day missing when reporting cells carry <= half the weight
CHINA_BOX = (72.0, 17.0, 136.0, 55.0)       # lon/lat prefilter; every unit lies inside 73.5-135.1 E, 18.1-53.6 N

# event table
SYSTEM = "meteorological_event_id"
PRE_DAYS, MIN_CLEAN_PRE = 14, 7
EVENT_NETWORK_YEARS = (2017, 2018)                       # events of year t use the network of year t - 1
FLOWS = {"local": ("local_spending", "normal_local_spending"),
         "inbound": ("inbound_spending", "normal_inbound_spending"),
         "outbound": ("outbound_spending", "normal_outbound_spending"),
         "merchant": ("merchant_place_spending", "normal_merchant_place_spending"),
         "resident": ("resident_associated_spending", "normal_resident_associated_spending")}
MAIN_MODEL = f"CurrentShock_inbound ~ E + E:h + Unknown + z_sev + z_sev:E + z_sev:E:h + z_sev:Unknown | {SYSTEM}"
ID_COLUMNS = {"city_id": str, "host_id": str, "guest_id": str}


def write(frame, name):
    frame.to_csv(INTERMEDIATE / f"{name}.csv", index=False, encoding=ENC, float_format="%.15g", lineterminator="\n")
    print(f"wrote {name}.csv ({len(frame):,} rows)")


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def load_weather(years):
    frames = []
    for year in years:
        block = pd.read_csv(INTERMEDIATE / f"city_daily_weather_{year}.csv", encoding=ENC)
        block["city_name"] = block.city_name.astype(str).str.strip()
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def conditional_cooccurrence(H, years, window=None):
    mask = np.ones(H.shape[0], bool) if window is None else (years >= window[0]) & (years <= window[1])
    block = H[mask]
    own = block.sum(axis=0)
    co = block.T @ block
    pi = np.divide(co, own[:, None], out=np.full_like(co, np.nan), where=own[:, None] > 0)
    np.fill_diagonal(pi, np.nan)
    return pi

def cooccurrence_and_network():
    thresholds = pd.read_csv(THRESHOLDS, encoding=ENC)
    thresholds["city_name"] = thresholds.city_name.astype(str).str.strip()
    level = dict(zip(thresholds.city_name, thresholds.return_level_3yr_mm.fillna(np.inf)))
    weather = load_weather(HISTORY_YEARS)
    weather["H"] = ((weather.rain_mm >= HEAVY_MM) | (weather.rain_mm >= weather.city_name.map(level))).astype(np.int8)
    wide = weather.pivot(index="date", columns="city_name", values="H").sort_index()
    if wide.isna().any().any():
        raise SystemExit("missing city-days in the 1981-2016 rainfall record")
    cities = list(wide.columns)
    H = wide.to_numpy(dtype=np.float32)
    years = pd.to_datetime(wide.index).year.to_numpy()
    print(f"rainfall record {H.shape[0]:,} days x {H.shape[1]} units | extreme city-days {int(H.sum()):,}")
    pis = {"pi": conditional_cooccurrence(H, years), "pi_early": conditional_cooccurrence(H, years, EARLY),
           "pi_late": conditional_cooccurrence(H, years, LATE)}

    grid = pd.read_parquet(GRID_WEIGHTS)
    grid["city_name"] = grid.city_name.astype(str).str.strip()
    centroid = (grid.assign(wlat=grid.latitude * grid.weight, wlon=grid.longitude * grid.weight)
                .groupby("city_name")[["wlat", "wlon", "weight"]].sum())
    lat, lon = centroid.wlat / centroid.weight, centroid.wlon / centroid.weight
    la, lo = lat.reindex(cities).to_numpy(), lon.reindex(cities).to_numpy()
    D = haversine(la[:, None], lo[:, None], la[None, :], lo[None, :])
    off = ~np.eye(len(cities), dtype=bool) & np.isfinite(pis["pi"]) & np.isfinite(D)
    ii, jj = np.where(off)
    pairs = pd.DataFrame({"focal_city": np.array(cities)[ii], "partner_city": np.array(cities)[jj],
                          "distance_km": D[off], **{k: v[off] for k, v in pis.items()}})

    # distance-only benchmark: mean pi in 60 quantile bins of log distance, linearly interpolated
    log_d = np.log(pairs.distance_km.clip(lower=5))
    edges = np.unique(np.quantile(log_d, np.linspace(0, 1, KERNEL_BINS + 1)))
    bins = np.clip(np.searchsorted(edges, log_d, side="right") - 1, 0, len(edges) - 2)
    kernel = (pairs.assign(ld=log_d, bin=bins).groupby("bin")
              .agg(distance_km=("distance_km", "median"), log_distance=("ld", "mean"), pi_mean=("pi", "mean"),
                   pi_p10=("pi", lambda s: s.quantile(.10)), pi_p90=("pi", lambda s: s.quantile(.90)),
                   n_pairs=("pi", "size")).reset_index())

    def pi_kernel(distance_km):
        return np.interp(np.log(np.clip(np.asarray(distance_km, float), 5, None)), kernel.log_distance, kernel.pi_mean)

    pairs["pi_kernel"] = pi_kernel(pairs.distance_km)
    kernel["r2_all_pairs"] = 1 - np.var(pairs.pi - pairs.pi_kernel) / np.var(pairs.pi)
    print(f"ordered pairs {len(pairs):,} | mean pi {pairs.pi.mean():.4f} | "
          f"distance-only R2 {kernel.r2_all_pairs.iloc[0]:.3f}")

    extreme = weather.loc[weather.H == 1, ["city_name", "date", "rain_mm"]].copy()
    extreme["date"] = pd.to_datetime(extreme.date)
    extreme = extreme.sort_values(["city_name", "date"]).reset_index(drop=True)
    gap = extreme.groupby("city_name").date.diff().dt.days
    extreme["event_index"] = (gap.isna() | (gap > MAX_GAP_DAYS)).astype(int).groupby(extreme.city_name).cumsum()
    history = extreme.groupby(["city_name", "event_index"], as_index=False).agg(peak_rain_mm=("rain_mm", "max"))
    hazard = history.groupby("city_name", as_index=False).agg(n_historical_events=("event_index", "size"),
                                                              median_event_peak_rain_mm=("peak_rain_mm", "median"))
    hazard["HistoricalExtremeFrequency"] = hazard.n_historical_events / len(HISTORY_YEARS)
    per_year = pd.Series(H.sum(axis=0) / len(HISTORY_YEARS), index=cities, name="extreme_days_per_year")
    hazard = pd.DataFrame({"city_name": cities}).merge(hazard, on="city_name", how="left").merge(
        per_year.rename_axis("city_name").reset_index(), on="city_name")
    hazard["n_historical_events"] = hazard.n_historical_events.fillna(0).astype(int)
    hazard["HistoricalExtremeFrequency"] = hazard.HistoricalExtremeFrequency.fillna(0.0)
    hazard["lat"], hazard["lon"] = hazard.city_name.map(lat), hazard.city_name.map(lon)

    codes = pd.read_excel(DICTIONARY, sheet_name=5).dropna()
    codes.columns = ["province", "city_name", "city_id"]
    codes["city_id"] = codes.city_id.astype(int).astype(str).str.zfill(4)
    names = dict(zip(codes.city_id, codes.city_name.astype(str).str.strip()))
    province = dict(zip(codes.city_name.astype(str).str.strip(), codes.province.astype(str).str.strip()))
    with_weather = set(cities) | set(load_weather(NETWORK_YEARS).city_name)

    network = pd.read_csv(INBOUND_NETWORK, encoding=ENC, dtype={"hostcity": str, "guestcity": str})
    network["q"] = np.expm1(network.cval_gtoh / 10.0)
    annual = network.groupby(["hostcity", "year", "guestcity"], as_index=False).q.sum()
    annual["w"] = annual.q / annual.groupby(["hostcity", "year"]).q.transform("sum")
    annual = annual.rename(columns={"hostcity": "host_id", "guestcity": "guest_id", "year": "network_year"})
    annual = annual.loc[annual.network_year.isin(NETWORK_YEARS)].reset_index(drop=True)
    annual["host_name"], annual["guest_name"] = annual.host_id.map(names), annual.guest_id.map(names)
    position = {name: i for i, name in enumerate(cities)}
    hi, gi = annual.host_name.map(position), annual.guest_name.map(position)
    observable = hi.notna() & gi.notna() & (annual.host_name != annual.guest_name)
    for key, matrix in pis.items():
        values = np.full(len(annual), np.nan)
        values[observable.to_numpy()] = matrix[hi[observable].astype(int), gi[observable].astype(int)]
        annual[key] = values
    annual["host_lat"], annual["host_lon"] = annual.host_name.map(lat), annual.host_name.map(lon)
    annual["guest_lat"], annual["guest_lon"] = annual.guest_name.map(lat), annual.guest_name.map(lon)
    annual["distance_km"] = haversine(annual.host_lat, annual.host_lon, annual.guest_lat, annual.guest_lon)
    annual["pi_kernel"] = np.where(annual.distance_km.notna(), pi_kernel(annual.distance_km.fillna(1.0)), np.nan)
    annual["usable"] = (observable & annual.pi.notna()).astype(int)
    annual["usable_geo"] = ((annual.usable == 1) & annual.distance_km.notna()).astype(int)
    annual["weather_observed"] = annual.guest_name.isin(with_weather).astype(int)
    annual["same_province"] = (annual.host_name.map(province) == annual.guest_name.map(province)).astype(int)

    usable = annual.loc[annual.usable == 1]
    g = usable.groupby(["host_id", "network_year"])
    cy = pd.DataFrame({"coverage": g.w.sum(),
                       "hdc_raw": g.apply(lambda d: (d.w * (1 - d.pi)).sum()),
                       "hdc_kernel_raw": g.apply(lambda d: (d.w * (1 - d.pi_kernel)).sum()),
                       "near_raw": g.apply(lambda d: (d.w * (d.distance_km < NEAR_KM)).sum()),
                       "dist_raw": g.apply(lambda d: (d.w * d.distance_km).sum()),
                       "EffectiveOrigins_obs": 1.0 / g.apply(lambda d: (d.w ** 2).sum())})
    cy["HDC"] = cy.hdc_raw / cy.coverage
    cy["HDC_kernel"] = cy.hdc_kernel_raw / cy.coverage
    cy["Near300"] = cy.near_raw / cy.coverage
    cy["MeanPartnerDistance"] = cy.dist_raw / cy.coverage
    for label in ("early", "late"):
        m = usable.loc[usable[f"pi_{label}"].notna()].groupby(["host_id", "network_year"])
        cy[f"HDC_{label}"] = m.apply(lambda d: (d.w * (1 - d[f"pi_{label}"])).sum()) / m.w.sum()
    cy["EffectiveOrigins"] = 1.0 / annual.groupby(["host_id", "network_year"]).apply(lambda d: (d.w ** 2).sum())
    cy["n_partners"] = annual.groupby(["host_id", "network_year"]).size()
    cy["n_usable_partners"] = g.size()
    cy = cy.drop(columns=["hdc_raw", "hdc_kernel_raw", "near_raw", "dist_raw"]).reset_index()
    cy = cy.rename(columns={"host_id": "city_id"})
    cy["city_name"] = cy.city_id.map(names)
    cy["lat"], cy["lon"] = cy.city_name.map(lat), cy.city_name.map(lon)

    daily = pd.read_csv(DAILY_PANEL, encoding=ENC, dtype={"city_id": str})
    daily["city_name"] = daily.city_name.astype(str).str.strip()
    daily["network_year"] = pd.to_datetime(daily.date).dt.year
    totals = daily.groupby(["city_name", "network_year"], as_index=False).agg(
        I=("inbound_spending", "sum"), M=("merchant_place_spending", "sum"),
        O=("outbound_spending", "sum"), R=("resident_associated_spending", "sum"))
    totals["InboundShare"] = totals.I / totals.M
    totals["BaselineOutboundShare"] = totals.O / totals.R
    cy = cy.merge(totals[["city_name", "network_year", "InboundShare", "BaselineOutboundShare"]],
                  on=["city_name", "network_year"], how="left")
    cy["E"] = cy.InboundShare * cy.coverage
    cy["Decoupled"] = cy.E * cy.HDC
    cy["Correlated"] = cy.E * (1 - cy.HDC)
    cy["Unknown"] = cy.InboundShare * (1 - cy.coverage)
    cy = cy[["city_id", "city_name", "network_year", "lat", "lon", "n_partners", "n_usable_partners", "coverage",
             "HDC", "HDC_early", "HDC_late", "HDC_kernel", "EffectiveOrigins", "EffectiveOrigins_obs",
             "MeanPartnerDistance", "Near300", "InboundShare", "BaselineOutboundShare", "E", "Decoupled",
             "Correlated", "Unknown"]].sort_values(["network_year", "city_id"])
    print(f"partner rows {len(annual):,} | city-years {len(cy)} | "
          + " | ".join(f"{y}: median HDC {cy.loc[cy.network_year == y, 'HDC'].median():.3f}" for y in NETWORK_YEARS))

    annual = annual[["host_id", "host_name", "network_year", "guest_id", "guest_name", "q", "w", "pi", "pi_early",
                     "pi_late", "pi_kernel", "distance_km", "usable", "usable_geo", "weather_observed",
                     "same_province", "host_lat", "host_lon", "guest_lat", "guest_lon"]]
    write(pairs, "hazard_pairs")
    write(kernel, "hazard_kernel")
    write(annual, "partner_network")
    write(cy, "city_year_network")
    write(hazard, "city_hazard")


def product_file(product, year):
    folder, pattern = PRODUCTS[product]
    return folder / pattern.format(year=year)


def spatial_dims(da):
    return ("lat" if "lat" in da.dims else "latitude"), ("lon" if "lon" in da.dims else "longitude")


def grid_cells(product):
    with xr.open_dataset(product_file(product, PRODUCT_YEARS[0])) as ds:
        da = ds["precip"]
        lat_name, lon_name = spatial_dims(da)
        lat, lon = ds[lat_name].values.astype(float), ds[lon_name].values.astype(float)
        valid = np.isfinite(da.transpose("time", lat_name, lon_name).values).any(axis=0)
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    II, JJ = np.meshgrid(np.arange(len(lat)), np.arange(len(lon)), indexing="ij")
    LO = np.where(LO > 180, LO - 360, LO)
    west, south, east, north = CHINA_BOX
    keep = valid & (LO >= west) & (LO <= east) & (LA >= south) & (LA <= north)
    cells = gpd.GeoDataFrame(
        {"i": II[keep], "j": JJ[keep], "lat": LA[keep], "lon": LO[keep]},
        geometry=shapely.box(LO[keep] - dlon / 2, LA[keep] - dlat / 2, LO[keep] + dlon / 2, LA[keep] + dlat / 2),
        crs=4326)
    return cells, len(lon)


def overlap_weights(cells, units):
    overlay = gpd.overlay(cells.to_crs(EQUAL_AREA_CRS), units.to_crs(EQUAL_AREA_CRS), how="intersection")
    overlay["area_m2"] = overlay.geometry.area
    overlay = overlay[overlay.area_m2 > 0].copy()
    overlay["weight"] = overlay.area_m2 / overlay.groupby("city_name").area_m2.transform("sum")
    return pd.DataFrame(overlay[["city_name", "i", "j", "weight"]])


def aggregate_year(product, year, weights, n_lon):
    names = sorted(weights.city_name.unique())
    rows = weights.city_name.map({n: r for r, n in enumerate(names)}).to_numpy()
    cells, columns = np.unique(weights.i.to_numpy() * n_lon + weights.j.to_numpy(), return_inverse=True)
    matrix = sparse.csr_matrix((weights.weight.to_numpy(), (rows, columns)), shape=(len(names), len(cells)))
    with xr.open_dataset(product_file(product, year)) as ds:
        da = ds["precip"]
        lat_name, lon_name = spatial_dims(da)
        values = da.transpose("time", lat_name, lon_name).values.reshape(ds.sizes["time"], -1)[:, cells]
        values = values.astype(np.float64)
        dates = pd.to_datetime(ds["time"].values).date
    reported = np.isfinite(values)
    total = (matrix @ np.where(reported, values, 0.0).T).T
    cover = (matrix @ reported.T.astype(np.float64)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        rain = total / cover
    rain[cover <= MIN_COVER] = np.nan
    return pd.DataFrame({"date": np.repeat(dates, len(names)), "city_name": np.tile(names, len(dates)),
                         "rain_mm": rain.ravel()})

def alternative_rainfall_products():
    units = gpd.read_file(SHAPEFILE)[["city_name", "geometry"]]
    for product in PRODUCTS:
        cells, n_lon = grid_cells(product)
        weights = overlap_weights(cells, units)
        daily = pd.concat([aggregate_year(product, year, weights, n_lon) for year in PRODUCT_YEARS], ignore_index=True)
        daily.to_parquet(INTERMEDIATE / f"city_daily_{product}_1981_2019.parquet", index=False)
        print(f"{product}: {weights.city_name.nunique()} units, {len(weights):,} unit-cell overlaps, "
              f"{len(daily):,} unit-days, {int(daily.rain_mm.isna().sum()):,} missing", flush=True)


def event_shocks():
    panel = pd.read_csv(CF_PANEL, encoding=ENC, dtype={"city_id": str})
    fr = panel.loc[panel.cf_complete_end14 == 1].copy()
    fr["city_name"] = fr.city_name.astype(str).str.strip()
    fr["k_start"] = fr.event_time_from_start.astype(int)
    for label, (observed, normal) in FLOWS.items():
        fr[f"A_{label}"] = 100.0 * (fr[observed] / fr[normal] - 1.0)
    fr["G_out_raw"] = fr.A_outbound - fr.A_local
    fr["G_in_raw"] = fr.A_inbound - fr.A_local
    ev = (fr.groupby("city_event_id")
          .agg(city_name=("city_name", "first"), city_id=("city_id", "first"), meteorological_event_id=(SYSTEM, "first"),
               city_event_start=("city_event_start", "first"), city_event_end=("city_event_end", "first"),
               city_event_duration=("city_event_duration", "first"), peak_rain_mm=("peak_rain_mm", "first"),
               peak_relative_to_return_level=("peak_relative_to_return_level", "first"),
               cumulative_rain_mm=("cumulative_rain_mm", "first"),
               return_level_3yr_mm=("return_level_3yr_mm", "first")).sort_index())
    pre = fr.loc[fr.k_start.between(-PRE_DAYS, -1) & (fr.is_other_city_event_exposure == 0)]
    exposed = fr.loc[fr.is_focal_exposure_day == 1]
    ev["clean_pre_days"] = pre.groupby("city_event_id").size()
    for name in ("G_out", "G_in"):
        ev[name] = (exposed.groupby("city_event_id")[f"{name}_raw"].mean()
                    - pre.groupby("city_event_id")[f"{name}_raw"].mean())
    for label in FLOWS:
        ev[f"CleanPreMean_{label}"] = pre.groupby("city_event_id")[f"A_{label}"].mean()
        ev[f"ExposureMean_{label}"] = exposed.groupby("city_event_id")[f"A_{label}"].mean()
        ev[f"CurrentShock_{label}"] = ev[f"ExposureMean_{label}"] - ev[f"CleanPreMean_{label}"]
    ev = ev.reset_index()
    ev["city_id"] = ev.city_id.astype(str).str.zfill(4)
    ev["baseline_year"] = pd.to_datetime(ev.city_event_start).dt.year - 1
    return ev.loc[ev.clean_pre_days >= MIN_CLEAN_PRE].reset_index(drop=True)


def partner_weight_hit(events, network):
    """Partner weight hit by the same rainfall system (overlapping dates), not hit, or unknown.

    A partner is observable when its pi is defined ('usable' rule) or, for the '_weather' variant, when it has daily
    rainfall data.
    """
    catalogue = pd.read_csv(CATALOGUE, encoding=ENC)
    catalogue["city_name"] = catalogue.city_name.astype(str).str.strip()
    catalogue["s"], catalogue["e"] = pd.to_datetime(catalogue.city_event_start), pd.to_datetime(catalogue.city_event_end)
    by_system = {k: v for k, v in catalogue.groupby(SYSTEM)}
    by_host = {k: v for k, v in network.groupby(["host_id", "network_year"])}
    out = {k: [] for k in ("AffectedWeight", "UnaffectedWeight", "UnknownWeight", "AffectedWeight_weather",
                           "UnaffectedWeight_weather", "UnknownWeight_weather")}
    for r in events.itertuples(index=False):
        block = by_host.get((r.city_id, r.baseline_year))
        if block is None:
            for values in out.values():
                values.append(np.nan)
            continue
        system = by_system.get(r.meteorological_event_id)
        start, end = pd.Timestamp(r.city_event_start), pd.Timestamp(r.city_event_end)
        hit = set() if system is None else set(system.loc[(system.s <= end) & (system.e >= start)
                                                           & (system.city_name != r.city_name), "city_name"])
        affected = block.guest_name.isin(hit)
        for suffix, observable in (("", block.usable == 1), ("_weather", block.weather_observed == 1)):
            out[f"AffectedWeight{suffix}"].append(float(block.loc[observable & affected, "w"].sum()))
            out[f"UnaffectedWeight{suffix}"].append(float(block.loc[observable & ~affected, "w"].sum()))
            out[f"UnknownWeight{suffix}"].append(float(block.loc[~observable, "w"].sum()))
    for key, values in out.items():
        events[key] = values
    return events

def event_table():
    ev = event_shocks()
    print(f"events {len(ev)} | cities {ev.city_name.nunique()} | systems {ev[SYSTEM].nunique()}")
    network = pd.read_csv(PARTNER_NETWORK, encoding=ENC, dtype=ID_COLUMNS, low_memory=False)
    cy = pd.read_csv(CITY_YEAR_NETWORK, encoding=ENC, dtype=ID_COLUMNS, low_memory=False)
    cy = cy.rename(columns={"network_year": "baseline_year", "lat": "focal_lat", "lon": "focal_lon"})
    keep = ["city_id", "baseline_year", "focal_lat", "focal_lon", "coverage", "HDC", "HDC_early", "HDC_late",
            "HDC_kernel", "EffectiveOrigins", "MeanPartnerDistance", "Near300", "InboundShare",
            "BaselineOutboundShare", "E", "Decoupled", "Correlated", "Unknown"]
    ev = ev.merge(cy[keep], on=["city_id", "baseline_year"], how="left")
    shocks = [f"CurrentShock_{m}" for m in FLOWS]
    ev["network_sample"] = (ev.HDC.notna() & ev.baseline_year.isin(EVENT_NETWORK_YEARS)
                            & ev[shocks].notna().all(axis=1)).astype(int)
    ev["h"] = np.where(ev.network_sample == 1, ev.Decoupled / ev.E, np.nan)
    ns = ev.network_sample == 1
    if float((ev.loc[ns, "Decoupled"] + ev.loc[ns, "Correlated"] + ev.loc[ns, "Unknown"]
              - ev.loc[ns, "InboundShare"]).abs().max()) > 1e-10:
        raise SystemExit("decoupled + correlated + unknown differs from the inbound share")
    ev = partner_weight_hit(ev, network)
    for suffix in ("", "_weather"):
        total = ev[f"AffectedWeight{suffix}"] + ev[f"UnaffectedWeight{suffix}"] + ev[f"UnknownWeight{suffix}"]
        if float((total.loc[ev[f"AffectedWeight{suffix}"].notna()] - 1).abs().max()) > 1e-9:
            raise SystemExit("partner weights do not sum to one")

    s = ev.loc[ns]
    for name, column in [("z_sev", "peak_rain_mm"), ("z_sev_alt", "peak_relative_to_return_level"),
                         ("z_HDC", "HDC"), ("z_EO", "EffectiveOrigins"), ("z_dist", "MeanPartnerDistance"),
                         ("z_lat", "focal_lat"), ("z_lon", "focal_lon"), ("z_IS", "InboundShare"),
                         ("z_D", "Decoupled"), ("z_C", "Correlated"), ("z_X", "Unknown")]:
        ev[name] = np.where(ns, (ev[column] - s[column].mean()) / s[column].std(), np.nan)
    for column, name in [("h", "HDC"), ("EffectiveOrigins", "EffectiveOrigins"),
                         ("MeanPartnerDistance", "MeanPartnerDistance")]:
        ev[f"s_{name}"] = np.where(ns, ev[column] / float(s[column].std()), np.nan)

    # the estimation sample is the network sample after singleton rainfall systems are dropped
    model = pf.feols(MAIN_MODEL, data=ev.loc[ns], vcov={"CRV1": f"city_name + {SYSTEM}"})
    used = set(model._data.city_event_id)
    ev["in_estimation_sample"] = ev.city_event_id.isin(used).astype(int)
    print(f"network sample {int(ns.sum())} events in {ev.loc[ns, 'city_name'].nunique()} cities and "
          f"{ev.loc[ns, SYSTEM].nunique()} systems | estimation sample {len(used)}")
    ev.to_csv(EVENT_TABLE, index=False, encoding=ENC, float_format="%.15g", lineterminator="\n")
    print(f"wrote {EVENT_TABLE.name} ({len(ev):,} rows)")



def main():
    cooccurrence_and_network()
    alternative_rainfall_products()
    event_table()


if __name__ == "__main__":
    main()
