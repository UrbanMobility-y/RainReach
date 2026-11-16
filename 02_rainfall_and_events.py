"""Daily rainfall, extreme-rainfall thresholds and events for 369 prefecture-level units.

ERA5-Land precipitation is aggregated to UTC+8 calendar days with overlap-area weights, 3-year return levels are
fitted on 1981-2016 by peaks over threshold, and the extreme city-days of 2017-2019 (>= 50 mm or >= the return level)
are grouped into city events and rainfall systems of Queen-adjacent units with overlapping or consecutive dates.
"""
import argparse
import calendar
import json
import math
import os
import re
import time
import warnings
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import CRS
from scipy import sparse
from scipy.stats import genpareto
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
DATASET = ROOT / "Dataset"
INTERMEDIATE = ROOT / "Intermediate"

RAW_DIR = DATASET / "city daily weather" / "raw"
API_FILE = DATASET / "city daily weather" / "API.txt"
SHAPEFILE = DATASET / "GIS" / "Prefecture City.shp"
WEIGHTS = INTERMEDIATE / "city_grid_weights.parquet"
THRESHOLDS = INTERMEDIATE / "city_rainfall_thresholds.csv"
EXTREME_DAYS = INTERMEDIATE / "extreme_rainfall_city_days_2017_2019.csv"
EVENTS = INTERMEDIATE / "extreme_rainfall_events_2017_2019.csv"

# rainfall aggregation
START_YEAR, END_YEAR = 1981, 2020
CDS_DATASET = "reanalysis-era5-land"
CDS_VARIABLE = "total_precipitation"
DOWNLOAD_TIMES = ["00:00", "16:00"]
LOCAL_MIDNIGHT_UTC_HOUR = 16
MARGIN_DEGREES = 0.2
EQUAL_AREA_CRS = CRS.from_proj4("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs")

# return levels: rainfall peaks above the 95th percentile of wet-day rain (peaks separated by >= 3 days without an
# exceedance), generalised Pareto fit with location 0; fewer than 30 peaks -> 50-mm criterion only
FIT_END_YEAR = 2016
TARGET_START, TARGET_END = pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")
HEAVY_MM = 50.0
WET_DAY_MM = 1.0
POT_QUANTILE = 0.95
RUN_NONEXCEEDANCE_DAYS = 3
MIN_WET_DAYS = 365
MIN_CLUSTER_PEAKS = 30
LOW_SAMPLE_WARNING_BELOW = 50
RETURN_PERIOD_YEARS = 3.0
DAYS_PER_YEAR = 365.2425
THRESHOLD_COLUMNS = ["city_name", "return_level_3yr_mm", "pot_threshold_mm", "gpd_shape", "gpd_scale_mm",
                     "cluster_rate_per_year", "n_cluster_peaks", "fit_status"]
EXTREME_COLUMNS = ["city_name", "date", "rain_mm", "return_level_3yr_mm", "exceed_50mm", "exceed_3yr_return",
                   "is_extreme_rainfall", "excess_over_50mm", "excess_over_return_level", "relative_to_return_level"]
EVENT_COLUMNS = [
    "city_event_id", "meteorological_event_id", "city_name", "city_event_start", "city_event_end",
    "city_event_duration", "peak_date", "peak_rain_mm", "cumulative_rain_mm", "n_extreme_days", "return_level_3yr_mm",
    "peak_relative_to_return_level", "max_excess_over_return_level", "has_50mm_day", "has_3yr_return_day",
    "previous_city_event_id", "previous_city_event_end", "days_since_previous_city_event", "next_city_event_id",
    "next_city_event_start", "days_until_next_city_event", "repeat_within_3d", "repeat_within_7d",
    "repeat_within_14d", "repeat_within_30d", "meteorological_event_start", "meteorological_event_end",
    "meteorological_event_duration", "meteorological_peak_date", "meteorological_peak_rain_mm",
    "affected_city_count", "city_event_count", "multi_city_event"]


def polygonal_only(geom):
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        parts = [p for p in (polygonal_only(g) for g in geom.geoms) if p is not None and not p.is_empty]
        return unary_union(parts) if parts else Polygon()
    return Polygon()


def load_units():
    gdf = gpd.read_file(SHAPEFILE)
    invalid = gdf.geometry.notna() & ~gdf.geometry.is_empty & ~gdf.geometry.is_valid
    names = gdf["city_name"].astype("string").str.strip()
    gdf = gpd.GeoDataFrame({"city_name": names, "geometry": gdf.geometry}, crs=gdf.crs)
    if invalid.any():
        repaired = shapely.make_valid(gdf.loc[invalid, "geometry"].array)
        gdf.loc[invalid, "geometry"] = [polygonal_only(x) for x in repaired]
        still_bad = gdf.geometry.notna() & ~gdf.geometry.is_empty & ~gdf.geometry.is_valid
        if still_bad.any():
            gdf.loc[still_bad, "geometry"] = gdf.loc[still_bad, "geometry"].buffer(0)
    if names.duplicated().any():
        gdf = gdf.dissolve(by="city_name", as_index=False)
    gdf = gdf.to_crs(4326)
    if (~gdf.geometry.is_valid).any() or gdf.geometry.is_empty.any():
        raise ValueError("geometry repair incomplete")
    return gdf.sort_values("city_name", kind="stable").reset_index(drop=True)


def download_area(units):
    west, south, east, north = map(float, units.total_bounds)
    return [math.ceil((north + MARGIN_DEGREES) * 10) / 10, math.floor((west - MARGIN_DEGREES) * 10) / 10,
            math.floor((south - MARGIN_DEGREES) * 10) / 10, math.ceil((east + MARGIN_DEGREES) * 10) / 10]


def read_credentials(path):
    text = path.read_text(encoding="utf-8-sig")
    values = {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            values.update({str(k).lower(): str(v).strip() for k, v in parsed.items()})
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        match = re.match(r"^(url|key|token)\s*[:=]\s*['\"]?(.+?)['\"]?$", line.strip().strip(","), re.IGNORECASE)
        if match:
            values[match.group(1).lower()] = match.group(2).strip()
    url, key = values.get("url"), values.get("key") or values.get("token")
    if not url or not key:
        raise ValueError(f"{path.name} must contain url and key fields")
    return url, key


def month_path(year, month):
    return RAW_DIR / f"{year:04d}_{month:02d}.nc"


def month_is_complete(path, year, month):
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            times = pd.DatetimeIndex(ds["valid_time"].values)
        return times.nunique() == calendar.monthrange(year, month)[1] * len(DOWNLOAD_TIMES)
    except (OSError, ValueError, KeyError):
        return False


def download_missing_months(units, years):
    import cdsapi
    url, key = read_credentials(API_FILE)
    client = cdsapi.Client(url=url, key=key, quiet=True, timeout=600, retry_max=10, sleep_max=120)
    area = download_area(units)
    months = [(years[0] - 1, 12)] + [(y, m) for y in years for m in range(1, 13)]
    for year, month in months:
        target = month_path(year, month)
        if month_is_complete(target, year, month):
            continue
        request = {"product_type": ["reanalysis"], "variable": [CDS_VARIABLE], "year": [f"{year:04d}"],
                   "month": [f"{month:02d}"],
                   "day": [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
                   "time": DOWNLOAD_TIMES, "area": area, "data_format": "netcdf", "download_format": "unarchived"}
        for attempt in range(1, 4):
            part = target.with_suffix(".nc.part")
            try:
                client.retrieve(CDS_DATASET, request, str(part))
                os.replace(part, target)
                if month_is_complete(target, year, month):
                    print(f"downloaded {target.name}", flush=True)
                    break
                raise ValueError("incomplete file")
            except Exception as exc:
                part.unlink(missing_ok=True)
                print(f"download {year:04d}-{month:02d} failed (attempt {attempt}): {exc}", flush=True)
                time.sleep(15 * attempt)


def grid_edges(centers):
    mids = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[centers[0] - (centers[1] - centers[0]) / 2], mids,
                           [centers[-1] + (centers[-1] - centers[-2]) / 2]])


def build_weights(units, sample_file):
    with xr.open_dataset(sample_file, engine="netcdf4") as ds:
        lat = ds["latitude"].values.astype(float)
        lon = ds["longitude"].values.astype(float)
        valid = np.isfinite(ds["tp"].isel(valid_time=0).transpose("latitude", "longitude").values).reshape(-1)
    lat_edges, lon_edges = grid_edges(lat), grid_edges(lon)
    lat_idx = np.repeat(np.arange(len(lat), dtype=np.int32), len(lon))[valid]
    lon_idx = np.tile(np.arange(len(lon), dtype=np.int32), len(lat))[valid]
    west = np.tile(np.minimum(lon_edges[:-1], lon_edges[1:]), len(lat))[valid]
    east = np.tile(np.maximum(lon_edges[:-1], lon_edges[1:]), len(lat))[valid]
    south = np.repeat(np.minimum(lat_edges[:-1], lat_edges[1:]), len(lon))[valid]
    north = np.repeat(np.maximum(lat_edges[:-1], lat_edges[1:]), len(lon))[valid]
    grid = gpd.GeoDataFrame({"lat_idx": lat_idx, "lon_idx": lon_idx, "latitude": lat[lat_idx],
                             "longitude": lon[lon_idx]},
                            geometry=shapely.box(west, south, east, north), crs=4326).to_crs(EQUAL_AREA_CRS)
    projected = units.to_crs(EQUAL_AREA_CRS)
    index = grid.sindex
    records = []
    for row in projected.itertuples(index=False):
        candidates = index.query(row.geometry, predicate="intersects")
        areas = np.asarray(shapely.area(shapely.intersection(grid.geometry.iloc[candidates].array, row.geometry)))
        keep = np.isfinite(areas) & (areas > 0)
        candidates, areas = np.asarray(candidates)[keep], areas[keep]
        if areas.sum() <= 0:
            raise ValueError(f"no grid cell overlaps {row.city_name}")
        cells = grid.iloc[candidates]
        for li, lj, la, lo, ar in zip(cells.lat_idx, cells.lon_idx, cells.latitude, cells.longitude, areas):
            records.append((row.city_name, int(li), int(lj), float(la), float(lo), float(ar), float(ar / areas.sum())))
    weights = pd.DataFrame.from_records(records, columns=["city_name", "lat_idx", "lon_idx", "latitude", "longitude",
                                                          "overlap_area_m2", "weight"])
    return weights, lat, lon


def read_snapshots(path, lat, lon):
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if not (np.allclose(ds["latitude"].values, lat) and np.allclose(ds["longitude"].values, lon)):
            raise ValueError(f"grid mismatch in {path.name}")
        da = ds["tp"].squeeze(drop=True)
        if da.attrs.get("units") != "m":
            raise ValueError(f"unexpected precipitation units in {path.name}")
        times = pd.DatetimeIndex(ds["valid_time"].values)
        data = np.asarray(da.transpose("valid_time", "latitude", "longitude").values, dtype=np.float64) * 1000.0
    snapshots = {}
    for i, stamp in enumerate(times):
        if stamp.hour in (0, LOCAL_MIDNIGHT_UTC_HOUR):
            values = data[i].reshape(-1)
            snapshots[stamp] = np.where(np.isfinite(snapshots[stamp]), snapshots[stamp], values) \
                if stamp in snapshots else values
    return snapshots


def area_weighted(daily_grid, matrix):
    valid = np.isfinite(daily_grid)
    numerator = matrix @ np.where(valid, daily_grid, 0.0)
    coverage = matrix @ valid.astype(float)
    out = np.full(matrix.shape[0], np.nan)
    out[coverage > 0] = numerator[coverage > 0] / coverage[coverage > 0]
    return out


def process_year(year, weights, lat, lon, names):
    rows = weights.city_name.map({n: i for i, n in enumerate(names)}).to_numpy()
    columns = weights.lat_idx.to_numpy(np.int64) * len(lon) + weights.lon_idx.to_numpy(np.int64)
    matrix = sparse.csr_matrix((weights.weight.to_numpy(float), (rows, columns)), shape=(len(names), len(lat) * len(lon)))
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    result = np.full((len(dates), len(names)), np.nan)
    previous = month_path(year - 1, 12)
    prev16 = None
    if month_is_complete(previous, year - 1, 12):
        prev16 = read_snapshots(previous, lat, lon).get(pd.Timestamp(year - 1, 12, 31, LOCAL_MIDNIGHT_UTC_HOUR))
    negative, components = 0, 0
    for month in range(1, 13):
        path = month_path(year, month)
        if not month_is_complete(path, year, month):
            print(f"{path.name} unavailable; its days stay missing", flush=True)
            prev16 = None
            continue
        snapshots = read_snapshots(path, lat, lon)
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            a00 = snapshots.get(pd.Timestamp(year, month, day, 0))
            a16 = snapshots.get(pd.Timestamp(year, month, day, LOCAL_MIDNIGHT_UTC_HOUR))
            if a00 is not None and a16 is not None and prev16 is not None:
                tail = a00 - prev16
                finite = tail[np.isfinite(tail)]
                negative += int((finite < -0.01).sum())
                components += len(finite)
                tail = np.where((tail < 0) & (tail >= -0.01), 0.0, tail)
                daily = tail + a16
                daily = np.where((daily < 0) & (daily >= -0.01), 0.0, daily)
                result[(date(year, month, day) - date(year, 1, 1)).days, :] = area_weighted(daily, matrix)
            prev16 = a16
    if components and negative / components > 0.001:
        raise ValueError(f"{year}: more than 0.1% of de-accumulated components are below -0.01 mm")
    result = np.where((result < 0) & (result >= -0.01), 0.0, result)
    out = pd.concat([pd.DataFrame({"city_name": city, "date": dates.strftime("%Y-%m-%d"),
                                   "rain_mm": np.round(result[:, j], 4)}) for j, city in enumerate(names)],
                    ignore_index=True).sort_values(["city_name", "date"], kind="stable")
    out.to_csv(INTERMEDIATE / f"city_daily_weather_{year}.csv", index=False, encoding="utf-8-sig",
               float_format="%.4f", lineterminator="\n")
    return len(out), int(out.rain_mm.isna().sum())


def year_file_complete(year, n_units):
    path = INTERMEDIATE / f"city_daily_weather_{year}.csv"
    if not path.exists():
        return False
    rain = pd.read_csv(path, encoding="utf-8-sig", usecols=["rain_mm"]).rain_mm
    return len(rain) == n_units * (366 if calendar.isleap(year) else 365) and not rain.isna().any()


def aggregate_rainfall(download, overwrite):
    years = list(range(START_YEAR, END_YEAR + 1))
    units = load_units()
    if download:
        download_missing_months(units, years)
    names = units.city_name.tolist()
    pending_years = [y for y in years if overwrite or not year_file_complete(y, len(names))]
    if not pending_years and WEIGHTS.exists():
        print("yearly rainfall files complete; aggregation skipped (use --overwrite to rebuild)", flush=True)
        return
    weights, lat, lon = build_weights(units, month_path(2019, 1))
    weights.to_parquet(WEIGHTS, index=False, compression="zstd")
    print(f"weights: {weights.city_name.nunique()} units, {len(weights):,} unit-cell overlaps", flush=True)
    incomplete = []
    for year in pending_years:
        n_rows, n_missing = process_year(year, weights, lat, lon, names)
        print(f"{year}: {n_rows:,} unit-days, {n_missing} missing", flush=True)
        if n_missing:
            incomplete.append(year)
    if incomplete:
        raise SystemExit(f"rainfall missing in {incomplete}; the first day of a year needs the previous December's "
                         f"raw file (run with --download)")


def read_weather():
    frames, cities = [], None
    for year in range(START_YEAR, TARGET_END.year + 1):
        df = pd.read_csv(INTERMEDIATE / f"city_daily_weather_{year}.csv", encoding="utf-8-sig",
                         dtype={"city_name": "string", "rain_mm": "float64"}, parse_dates=["date"])
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        year_cities = sorted(df.city_name.unique().tolist())
        cities = cities or year_cities
        if year_cities != cities:
            raise ValueError(f"{year}: the set of units differs from {START_YEAR}")
        if (len(df) != len(cities) * len(dates) or df.duplicated(["city_name", "date"]).any()
                or df.rain_mm.isna().any() or (df.rain_mm < 0).any()):
            raise ValueError(f"{year}: incomplete, duplicated, missing or negative rainfall")
        df["city_name"] = pd.Categorical(df.city_name, categories=cities)
        frames.append(df[["city_name", "date", "rain_mm"]])
    data = pd.concat(frames, ignore_index=True)
    data["city_name"] = data.city_name.astype("category")
    return data, cities


def rainfall_peak_excesses(dates, rainfall, threshold):
    exceed = np.flatnonzero(rainfall > threshold)
    if exceed.size == 0:
        return np.array([])
    days = dates.to_numpy(dtype="datetime64[D]")
    peaks, peak, previous = [], float(rainfall[exceed[0]]), days[exceed[0]]
    for i in exceed[1:]:
        if int((days[i] - previous).astype(int)) > RUN_NONEXCEEDANCE_DAYS:
            peaks.append(peak)
            peak = float(rainfall[i])
        else:
            peak = max(peak, float(rainfall[i]))
        previous = days[i]
    peaks.append(peak)
    return np.asarray(peaks) - threshold


def fit_city(city, block):
    fit = dict(city_name=city, return_level_3yr_mm=math.nan, pot_threshold_mm=math.nan, gpd_shape=math.nan,
               gpd_scale_mm=math.nan, cluster_rate_per_year=math.nan, n_cluster_peaks=0, fit_status="not_fitted")
    block = block.sort_values("date", kind="stable")
    rainfall = block.rain_mm.to_numpy(dtype=float)
    wet = rainfall[rainfall >= WET_DAY_MM]
    if wet.size < MIN_WET_DAYS:
        fit["fit_status"] = f"failed_insufficient_wet_days:{wet.size}<{MIN_WET_DAYS}"
        return fit
    threshold = float(np.quantile(wet, POT_QUANTILE, method="linear"))
    excesses = rainfall_peak_excesses(block.date, rainfall, threshold)
    excesses = excesses[np.isfinite(excesses) & (excesses > 0)]
    fit.update(pot_threshold_mm=threshold, n_cluster_peaks=int(excesses.size),
               cluster_rate_per_year=excesses.size / (len(block) / DAYS_PER_YEAR))
    if excesses.size < MIN_CLUSTER_PEAKS:
        fit["fit_status"] = f"failed_insufficient_cluster_peaks:{excesses.size}<{MIN_CLUSTER_PEAKS}"
        return fit
    if float(np.ptp(excesses)) <= 0:
        fit["fit_status"] = "failed_zero_excess_variation"
        return fit
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shape, loc, scale = (float(v) for v in genpareto.fit(excesses, floc=0.0))
    except Exception as exc:
        fit["fit_status"] = f"failed_optimizer:{type(exc).__name__}:{str(exc)[:120]}"
        return fit
    fit.update(gpd_shape=shape, gpd_scale_mm=scale)
    if not all(np.isfinite([shape, loc, scale])) or abs(loc) > 1e-10 or scale <= 0:
        fit["fit_status"] = "failed_invalid_gpd_parameters"
        return fit
    rate_factor = RETURN_PERIOD_YEARS * fit["cluster_rate_per_year"]
    if not np.isfinite(rate_factor) or rate_factor <= 1:
        fit["fit_status"] = f"failed_return_period_below_threshold:factor={rate_factor:.6g}"
        return fit
    if abs(shape) < 1e-7:
        level = threshold + scale * math.log(rate_factor)
    else:
        level = threshold + scale / shape * (rate_factor ** shape - 1.0)
    if not np.isfinite(level) or level <= threshold:
        fit["fit_status"] = f"failed_invalid_return_level:{level}"
        return fit
    if shape < 0:
        endpoint = threshold - scale / shape
        if level > endpoint * (1 + 1e-10) or rainfall.max() > endpoint * (1 + 1e-7):
            fit["fit_status"] = "failed_negative_shape_endpoint_violation"
            return fit
    fit["return_level_3yr_mm"] = float(level)
    notes = []
    if excesses.size < LOW_SAMPLE_WARNING_BELOW:
        notes.append(f"low_cluster_peaks:{excesses.size}<{LOW_SAMPLE_WARNING_BELOW}")
    if shape <= -0.5 or shape >= 0.5:
        notes.append(f"shape_outside_typical_range:{shape:.6f}")
    messages = [str(w.message).replace("\n", " ")[:100] for w in caught]
    if messages:
        notes.append("optimizer:" + " | ".join(messages))
    fit["fit_status"] = "warning_" + " | ".join(notes) if notes else "ok"
    return fit


def fit_thresholds():
    data, cities = read_weather()
    history = data.loc[data.date.dt.year <= FIT_END_YEAR]
    grouped = {str(k): v for k, v in history.groupby("city_name", observed=True, sort=False)}
    thresholds = pd.DataFrame([fit_city(city, grouped[city]) for city in cities])
    failed = thresholds.loc[thresholds.fit_status.str.startswith("failed"), "city_name"].tolist()
    print(f"return levels fitted for {thresholds.return_level_3yr_mm.notna().sum()} of {len(thresholds)} units; "
          f"failed: {failed}")

    target = data.loc[data.date.between(TARGET_START, TARGET_END)].copy()
    target["city_name"] = target.city_name.astype("string")
    lookup = thresholds[["city_name", "return_level_3yr_mm"]].astype({"city_name": "string"})
    target = target.merge(lookup, on="city_name", how="left", validate="many_to_one")
    target["exceed_50mm"] = (target.rain_mm >= HEAVY_MM).astype("int8")
    has_level = target.return_level_3yr_mm.notna() & (target.return_level_3yr_mm > 0)
    target["exceed_3yr_return"] = (has_level & (target.rain_mm >= target.return_level_3yr_mm)).astype("int8")
    target["is_extreme_rainfall"] = ((target.exceed_50mm == 1) | (target.exceed_3yr_return == 1)).astype("int8")
    target["excess_over_50mm"] = target.rain_mm - HEAVY_MM
    target["excess_over_return_level"] = target.rain_mm - target.return_level_3yr_mm
    target["relative_to_return_level"] = target.rain_mm / target.return_level_3yr_mm
    extreme = target.loc[target.is_extreme_rainfall == 1].sort_values(["city_name", "date"], kind="stable")
    print(f"extreme city-days 2017-2019: {len(extreme):,} in {extreme.city_name.nunique()} units "
          f"(50 mm {int(extreme.exceed_50mm.sum())}, return level {int(extreme.exceed_3yr_return.sum())})")

    options = dict(index=False, encoding="utf-8-sig", float_format="%.6f", date_format="%Y-%m-%d", na_rep="",
                   lineterminator="\n")
    thresholds.to_csv(THRESHOLDS, columns=THRESHOLD_COLUMNS, **options)
    extreme.to_csv(EXTREME_DAYS, columns=EXTREME_COLUMNS, **options)



def read_extreme_days():
    thresholds = pd.read_csv(THRESHOLDS, encoding="utf-8-sig")
    extreme = pd.read_csv(EXTREME_DAYS, encoding="utf-8-sig", parse_dates=["date"])
    no_level = set(thresholds.loc[thresholds.return_level_3yr_mm.isna(), "city_name"])
    if extreme.city_name.isin(no_level).any():
        raise ValueError("extreme days in a unit without a return level")
    if extreme.duplicated(["city_name", "date"]).any() or not (extreme.is_extreme_rainfall == 1).all():
        raise ValueError("extreme-day input is not one row per extreme city-day")
    mapped = extreme.city_name.map(thresholds.set_index("city_name").return_level_3yr_mm)
    if not np.allclose(mapped.to_numpy(), extreme.return_level_3yr_mm.to_numpy(), atol=1e-6):
        raise ValueError("return levels differ from the threshold file")
    return thresholds, extreme.sort_values(["city_name", "date"], kind="stable").reset_index(drop=True)


def queen_neighbours(expected_cities):
    gdf = gpd.read_file(SHAPEFILE)
    if (~gdf.geometry.is_valid).any():
        gdf.geometry = [polygonal_only(x) for x in shapely.make_valid(gdf.geometry.array)]
    gdf = gpd.GeoDataFrame({"city_name": gdf["city_name"].astype("string").str.strip(), "geometry": gdf.geometry},
                           crs=gdf.crs)
    gdf = gdf.dissolve(by="city_name", as_index=False).to_crs(4326)
    if set(gdf.city_name) != expected_cities:
        raise ValueError("boundary units differ from the rainfall units")
    left, right = gdf.sindex.query(gdf.geometry, predicate="intersects")
    names = gdf.city_name.astype(str).to_numpy()
    return {tuple(sorted((names[a], names[b]))) for a, b in zip(left, right) if a != b}


def city_events(extreme):
    records = []
    for _, block in extreme.groupby("city_name", sort=True):
        block = block.sort_values("date")
        run = block.date.diff().dt.days.ne(1).cumsum()
        for _, event in block.groupby(run, sort=True):
            start, end = event.date.iloc[0], event.date.iloc[-1]
            peak = float(event.rain_mm.max())
            level = event.return_level_3yr_mm.drop_duplicates()
            if len(level) != 1:
                raise ValueError("return level varies within a city event")
            records.append({
                "city_name": str(event.city_name.iloc[0]), "city_event_start": start, "city_event_end": end,
                "city_event_duration": int((end - start).days + 1),
                "peak_date": event.loc[event.rain_mm == peak, "date"].min(), "peak_rain_mm": peak,
                "cumulative_rain_mm": float(event.rain_mm.sum()), "n_extreme_days": int(len(event)),
                "return_level_3yr_mm": float(level.iloc[0]),
                "peak_relative_to_return_level": peak / float(level.iloc[0]),
                "max_excess_over_return_level": float(event.excess_over_return_level.max()),
                "has_50mm_day": int((event.exceed_50mm == 1).any()),
                "has_3yr_return_day": int((event.exceed_3yr_return == 1).any())})
    events = pd.DataFrame.from_records(records).sort_values(
        ["city_event_start", "city_event_end", "city_name", "peak_date"], kind="stable").reset_index(drop=True)
    events.insert(0, "city_event_id", [f"CE2017_2019_{i:06d}" for i in range(1, len(events) + 1)])
    return events


def add_repeat_timing(events):
    events = events.sort_values(["city_name", "city_event_start", "city_event_end", "city_event_id"],
                                kind="stable").copy()
    by_city = events.groupby("city_name", sort=False)
    events["previous_city_event_id"] = by_city.city_event_id.shift(1)
    events["previous_city_event_end"] = by_city.city_event_end.shift(1)
    events["days_since_previous_city_event"] = ((events.city_event_start - events.previous_city_event_end).dt.days
                                                - 1).astype("Int64")
    events["next_city_event_id"] = by_city.city_event_id.shift(-1)
    events["next_city_event_start"] = by_city.city_event_start.shift(-1)
    events["days_until_next_city_event"] = ((events.next_city_event_start - events.city_event_end).dt.days
                                            - 1).astype("Int64")
    for days in (3, 7, 14, 30):
        events[f"repeat_within_{days}d"] = (events.days_since_previous_city_event.notna()
                                            & (events.days_since_previous_city_event <= days)).astype("int8")
    return events.sort_values("city_event_id", kind="stable").reset_index(drop=True)


class UnionFind:
    def __init__(self, size):
        self.parent, self.rank = list(range(size)), [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def rainfall_systems(events, edges):
    events = events.reset_index(drop=True).copy()
    by_city = {city: block.index.tolist() for city, block in events.groupby("city_name", sort=False)}
    links = UnionFind(len(events))
    one_day = pd.Timedelta(days=1)
    starts, ends = events.city_event_start, events.city_event_end
    for city_a, city_b in sorted(edges):
        for i in by_city.get(city_a, []):
            for j in by_city.get(city_b, []):
                if starts[j] <= ends[i] + one_day and starts[i] <= ends[j] + one_day:
                    links.union(i, j)
    events["_component"] = [links.find(i) for i in range(len(events))]
    systems = []
    for component, block in events.groupby("_component", sort=False):
        start, end = block.city_event_start.min(), block.city_event_end.max()
        peak = block.sort_values(["peak_rain_mm", "peak_date", "city_event_id"], ascending=[False, True, True],
                                 kind="stable").iloc[0]
        systems.append({"_component": component, "_first_event": block.city_event_id.min(),
                        "meteorological_event_start": start, "meteorological_event_end": end,
                        "meteorological_event_duration": int((end - start).days + 1),
                        "affected_city_count": block.city_name.nunique(), "city_event_count": len(block),
                        "meteorological_peak_rain_mm": float(peak.peak_rain_mm),
                        "meteorological_peak_date": peak.peak_date,
                        "multi_city_event": int(block.city_name.nunique() >= 2)})
    systems = pd.DataFrame(systems).sort_values(
        ["meteorological_event_start", "meteorological_event_end", "_first_event"], kind="stable").reset_index(drop=True)
    systems.insert(0, "meteorological_event_id", [f"ME2017_2019_{i:05d}" for i in range(1, len(systems) + 1)])
    events["meteorological_event_id"] = events._component.map(systems.set_index("_component").meteorological_event_id)
    events = events.drop(columns="_component").merge(systems.drop(columns=["_component", "_first_event"]),
                                                      on="meteorological_event_id", how="left", validate="many_to_one")
    return events.sort_values("city_event_id", kind="stable").reset_index(drop=True)


def build_events():
    thresholds, extreme = read_extreme_days()
    edges = queen_neighbours(set(thresholds.city_name))
    events = rainfall_systems(add_repeat_timing(city_events(extreme)), edges)
    if int(events.n_extreme_days.sum()) != len(extreme):
        raise SystemExit("city events do not cover every extreme city-day exactly once")
    print(f"Queen edges {len(edges):,} | extreme city-days {len(extreme):,} | city events {len(events):,} | "
          f"rainfall systems {events.meteorological_event_id.nunique():,}")
    events.to_csv(EVENTS, index=False, columns=EVENT_COLUMNS, encoding="utf-8-sig", float_format="%.6f",
                  date_format="%Y-%m-%d", na_rep="", lineterminator="\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="retrieve missing monthly ERA5-Land files first")
    parser.add_argument("--overwrite", action="store_true", help="rebuild yearly rainfall files that already exist")
    args = parser.parse_args()
    aggregate_rainfall(args.download, args.overwrite)
    fit_thresholds()
    build_events()


if __name__ == "__main__":
    main()
