"""Event-time panel and normal (counterfactual) spending for every city event in a UnionPay city, 2017-2019.

Each event contributes the dates [start - 28, end + 28] with spending, rainfall, calendar and other-event flags.
Normal local, inbound and outbound spending come from Poisson models with city x month-of-year and date fixed effects,
log mu = alpha(city, month) + delta(date), fitted on all city-days outside every event window.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
DATASET = ROOT / "Dataset"
INTERMEDIATE = ROOT / "Intermediate"

DAILY_PANEL = INTERMEDIATE / "unionpay_city_daily_2017_2019.csv"
CATALOGUE = INTERMEDIATE / "extreme_rainfall_events_2017_2019.csv"
EXTREME_DAYS = INTERMEDIATE / "extreme_rainfall_city_days_2017_2019.csv"
CALENDAR = DATASET / "China_calendar_2017_2019" / "China_calendar_2017_2019.csv"
EVENT_PANEL = INTERMEDIATE / "event_time_panel.csv"
CF_PANEL = INTERMEDIATE / "counterfactual_panel.csv"
QC_OUTPUT = INTERMEDIATE / "counterfactual_qc.csv"

ENC = "utf-8-sig"
YEARS = (2017, 2018, 2019)
WINDOW = 28
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")
SPENDING = ["local_spending", "inbound_spending", "outbound_spending", "resident_associated_spending",
            "merchant_place_spending"]
CALENDAR_COLUMNS = ["weekday", "is_weekend", "is_public_holiday", "is_adjusted_workday", "is_effective_workday",
                    "holiday_name"]
EVENT_COLUMNS = [
    "city_event_id", "meteorological_event_id", "city_event_start", "city_event_end", "city_event_duration",
    "focal_event_duration", "focal_n_extreme_days", "peak_date", "peak_rain_mm", "cumulative_rain_mm",
    "n_extreme_days", "return_level_3yr_mm", "peak_relative_to_return_level", "max_excess_over_return_level",
    "has_50mm_day", "has_3yr_return_day", "rl_only_event", "affected_city_count", "multi_city_event",
    "previous_city_event_id", "days_since_previous_city_event", "next_city_event_id", "days_until_next_city_event",
    "repeat_within_3d", "repeat_within_7d", "repeat_within_14d", "repeat_within_30d",
    "previous_same_meteorological_system", "previous_new_system", "next_same_meteorological_system",
    "next_new_system", "pre_window_days_observed", "post_window_days_observed", "full_pre_28d", "full_post_28d",
    "full_28d_window"]
OUTPUT_COLUMNS = [
    "city_event_id", "meteorological_event_id", "city_id", "city_name", "date", "city_event_start", "city_event_end",
    "city_event_duration", "event_time_from_start", "event_time_from_end", "is_pre_event", "is_focal_exposure_day",
    "is_post_event", "focal_event_duration", "focal_n_extreme_days", "peak_date", "peak_rain_mm", "cumulative_rain_mm",
    "n_extreme_days", "return_level_3yr_mm", "peak_relative_to_return_level", "max_excess_over_return_level",
    "has_50mm_day", "has_3yr_return_day", "rl_only_event", "affected_city_count", "multi_city_event",
    "previous_city_event_id", "days_since_previous_city_event", "next_city_event_id", "days_until_next_city_event",
    "repeat_within_3d", "repeat_within_7d", "repeat_within_14d", "repeat_within_30d",
    "previous_same_meteorological_system", "previous_new_system", "next_same_meteorological_system", "next_new_system",
    "is_other_city_event_exposure", "is_same_system_other_exposure", "is_new_system_other_exposure",
    "other_city_event_id", "days_to_nearest_other_city_event", "rain_mm", *CALENDAR_COLUMNS, *SPENDING,
    "pre_window_days_observed", "post_window_days_observed", "full_pre_28d", "full_post_28d", "full_28d_window"]
DIMS = ["city_moy", "datef"]
MAX_LOG_DEVIATION = 1e-5
COMPONENTS = {"local_spending": "normal_local_spending", "inbound_spending": "normal_inbound_spending",
              "outbound_spending": "normal_outbound_spending"}
COMPLETENESS = {"cf_complete_pre14": ("event_time_from_start", -14, -1),
                "cf_complete_start14": ("event_time_from_start", -14, 14),
                "cf_complete_end14": ("event_time_from_end", -14, 14)}

warnings.filterwarnings("ignore", message=".*singleton fixed effect.*")


def load_catalogue():
    events = pd.read_csv(CATALOGUE, encoding=ENC)
    events["city_name"] = events.city_name.astype(str).str.strip()
    for column in ("city_event_start", "city_event_end", "peak_date", "previous_city_event_end",
                   "next_city_event_start", "meteorological_event_start", "meteorological_event_end",
                   "meteorological_peak_date"):
        events[column] = pd.to_datetime(events[column], errors="coerce")
    return events


def same_system_flag(other_system, own_system, same):
    flag = (other_system == own_system) if same else (other_system != own_system)
    return np.where(other_system.isna(), np.nan, flag.astype(float))

def build_event_time_panel():
    spending = pd.read_csv(DAILY_PANEL, dtype={"city_id": "string"}, encoding=ENC)
    spending["date"] = pd.to_datetime(spending.date, format="%Y-%m-%d")
    spending["city_name"] = spending.city_name.astype(str).str.strip()
    events = load_catalogue()
    weather = pd.concat([pd.read_csv(INTERMEDIATE / f"city_daily_weather_{y}.csv", encoding=ENC) for y in YEARS],
                        ignore_index=True)
    weather["city_name"] = weather.city_name.astype(str).str.strip()
    weather["date"] = pd.to_datetime(weather.date, format="%Y-%m-%d")
    calendar = pd.read_csv(CALENDAR, encoding=ENC)
    calendar["date"] = pd.to_datetime(calendar.date, format="%Y-%m-%d")

    matched = set(spending.city_name.unique()) & set(events.city_name.unique())
    focal = events.loc[events.city_name.isin(matched)].copy()
    if not (focal.city_event_duration == focal.n_extreme_days).all():
        raise SystemExit("city_event_duration differs from n_extreme_days")
    focal["rl_only_event"] = ((focal.has_50mm_day == 0) & (focal.has_3yr_return_day == 1)).astype("int64")
    system_of = events.set_index("city_event_id").meteorological_event_id
    previous_system = focal.previous_city_event_id.map(system_of)
    next_system = focal.next_city_event_id.map(system_of)
    focal["previous_same_meteorological_system"] = same_system_flag(previous_system, focal.meteorological_event_id, True)
    focal["previous_new_system"] = same_system_flag(previous_system, focal.meteorological_event_id, False)
    focal["next_same_meteorological_system"] = same_system_flag(next_system, focal.meteorological_event_id, True)
    focal["next_new_system"] = same_system_flag(next_system, focal.meteorological_event_id, False)
    focal["focal_event_duration"] = focal.city_event_duration
    focal["focal_n_extreme_days"] = focal.n_extreme_days
    window_start = focal.city_event_start - pd.Timedelta(days=WINDOW)
    window_end = focal.city_event_end + pd.Timedelta(days=WINDOW)
    pre_days = ((focal.city_event_start - pd.Timedelta(days=1) - window_start.clip(lower=START)).dt.days + 1).clip(lower=0)
    post_days = ((window_end.clip(upper=END) - (focal.city_event_end + pd.Timedelta(days=1))).dt.days + 1).clip(lower=0)
    focal["pre_window_days_observed"] = pre_days.astype("int64")
    focal["post_window_days_observed"] = post_days.astype("int64")
    focal["full_pre_28d"] = (pre_days == WINDOW).astype("int64")
    focal["full_post_28d"] = (post_days == WINDOW).astype("int64")
    focal["full_28d_window"] = (focal.full_pre_28d & focal.full_post_28d).astype("int64")
    print(f"catalogue {len(events):,} city events in {events.city_name.nunique()} units | "
          f"{len(focal):,} events in {len(matched)} UnionPay cities")

    panel = pd.concat([pd.DataFrame({"city_event_id": r.city_event_id, "city_name": r.city_name,
                                     "date": pd.date_range(max(r.city_event_start - pd.Timedelta(days=WINDOW), START),
                                                           min(r.city_event_end + pd.Timedelta(days=WINDOW), END))})
                       for r in focal.itertuples(index=False)], ignore_index=True)
    panel = panel.merge(focal[EVENT_COLUMNS], on="city_event_id", how="left", validate="many_to_one")
    panel["event_time_from_start"] = (panel.date - panel.city_event_start).dt.days
    panel["event_time_from_end"] = (panel.date - panel.city_event_end).dt.days
    panel["is_pre_event"] = (panel.date < panel.city_event_start).astype("int64")
    panel["is_focal_exposure_day"] = ((panel.date >= panel.city_event_start)
                                      & (panel.date <= panel.city_event_end)).astype("int64")
    panel["is_post_event"] = (panel.date > panel.city_event_end).astype("int64")

    exposure = pd.concat([pd.DataFrame({"city_name": r.city_name,
                                        "date": pd.date_range(r.city_event_start, r.city_event_end),
                                        "other_city_event_id": r.city_event_id,
                                        "other_meteorological_event_id": r.meteorological_event_id})
                          for r in events.itertuples(index=False)], ignore_index=True)
    if exposure.duplicated(["city_name", "date"]).any():
        raise SystemExit("two events of one city share a date")
    panel = panel.merge(exposure, on=["city_name", "date"], how="left", validate="many_to_one")
    other = panel.other_city_event_id.notna() & (panel.other_city_event_id != panel.city_event_id)
    same = other & (panel.other_meteorological_event_id == panel.meteorological_event_id)
    panel["is_other_city_event_exposure"] = other.astype("int64")
    panel["is_same_system_other_exposure"] = same.astype("int64")
    panel["is_new_system_other_exposure"] = (other & ~same).astype("int64")
    panel.loc[~other, "other_city_event_id"] = pd.NA
    panel = panel.drop(columns="other_meteorological_event_id")

    # distance in days from each panel date to the nearest other event of the same city (0 inside it)
    starts, ends, ids = {}, {}, {}
    for city, block in events.groupby("city_name"):
        starts[city] = block.city_event_start.to_numpy("datetime64[D]")
        ends[city] = block.city_event_end.to_numpy("datetime64[D]")
        ids[city] = block.city_event_id.to_numpy()
    nearest = np.full(len(panel), np.nan)
    dates = panel.date.to_numpy("datetime64[D]")
    own = panel.city_event_id.to_numpy()
    for city, index in panel.groupby("city_name").indices.items():
        day = dates[index][:, None]
        gap = np.maximum(np.maximum(starts[city][None, :] - day, day - ends[city][None, :]),
                         np.timedelta64(0, "D")).astype("timedelta64[D]").astype(float)
        gap[own[index][:, None] == ids[city][None, :]] = np.inf
        best = gap.min(axis=1)
        best[np.isinf(best)] = np.nan
        nearest[index] = best
    panel["days_to_nearest_other_city_event"] = nearest

    n = len(panel)
    panel = panel.merge(spending[["city_id", "city_name", "date", *SPENDING]], on=["city_name", "date"], how="left",
                        validate="many_to_one")
    if len(panel) != n or panel.city_id.isna().any() or panel[SPENDING].isna().any().any():
        raise SystemExit("spending merge incomplete")
    panel = panel.merge(weather[["city_name", "date", "rain_mm"]], on=["city_name", "date"], how="left",
                        validate="many_to_one")
    panel = panel.merge(calendar[["date", *CALENDAR_COLUMNS]], on="date", how="left", validate="many_to_one")
    if panel.rain_mm.isna().any() or panel.weekday.isna().any():
        raise SystemExit("rainfall or calendar merge incomplete")

    extreme = pd.read_csv(EXTREME_DAYS, encoding=ENC)
    extreme_keys = set(zip(extreme.city_name.astype(str).str.strip(), pd.to_datetime(extreme.date)))
    exposed = panel.loc[panel.is_focal_exposure_day == 1]
    if not set(zip(exposed.city_name, exposed.date)) <= extreme_keys:
        raise SystemExit("an event day is not an extreme city-day")
    days = exposed.groupby("city_event_id").size()
    if not (days == focal.set_index("city_event_id").loc[days.index, "city_event_duration"]).all():
        raise SystemExit("event days differ from city_event_duration")

    for column in ("date", "city_event_start", "city_event_end", "peak_date"):
        panel[column] = panel[column].dt.strftime("%Y-%m-%d")
    for column in ("previous_same_meteorological_system", "previous_new_system", "next_same_meteorological_system",
                   "next_new_system", "days_since_previous_city_event", "days_until_next_city_event",
                   "days_to_nearest_other_city_event"):
        panel[column] = panel[column].astype("Float64").astype("Int64")
    panel = panel[OUTPUT_COLUMNS].sort_values(["city_event_id", "date"], kind="mergesort").reset_index(drop=True)
    panel.to_csv(EVENT_PANEL, index=False, encoding=ENC, float_format="%.9g", lineterminator="\n")
    print(f"event-time rows {len(panel):,} | city events {panel.city_event_id.nunique():,} | "
          f"unique city-days {len(panel.drop_duplicates(['city_name', 'date'])):,}")


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


def fit_ipf(y, codes, n_levels, tol=1e-12, max_iterations=5000):
    """Poisson fixed-effects fit by iterative proportional fitting (cyclic maximisation over the effects)."""
    y = np.asarray(y, dtype="float64")
    log_mu = np.zeros(len(y))
    log_fe = [np.zeros(k) for k in n_levels]
    y_sum = [np.bincount(c, weights=y, minlength=k) for c, k in zip(codes, n_levels)]
    for iteration in range(max_iterations):
        delta = 0.0
        for j, (c, k) in enumerate(zip(codes, n_levels)):
            mu_sum = np.bincount(c, weights=np.exp(log_mu), minlength=k)
            with np.errstate(divide="ignore", invalid="ignore"):
                adjust = np.where(mu_sum > 0, np.log(y_sum[j] / mu_sum), 0.0)
            adjust[~np.isfinite(adjust)] = 0.0
            log_fe[j] += adjust
            log_mu += adjust[c]
            delta = max(delta, float(np.abs(adjust).max()))
        if delta < tol:
            break
    return log_fe, log_mu, iteration + 1


def fit_component(training, outcome, qc):
    sample = training.loc[drop_singletons(training, DIMS)]
    categories, codes, n_levels = {}, [], []
    for d in DIMS:
        cat = pd.Categorical(sample[d])
        categories[d] = cat.categories
        codes.append(cat.codes.astype(np.int64))
        n_levels.append(len(cat.categories))
    log_fe, log_mu, iterations = fit_ipf(sample[outcome].to_numpy(), codes, n_levels)
    reference = pf.fepois(f"{outcome} ~ 1 | " + " + ".join(DIMS), data=training, iwls_tol=1e-11)
    link = np.asarray(reference._Y_hat_link).ravel()
    deviation = float(np.abs(log_mu - link).max()) if len(link) == len(sample) else float("nan")
    if not deviation < MAX_LOG_DEVIATION:
        raise SystemExit(f"{outcome}: IPF and pyfixest fits differ ({deviation:.3e})")
    print(f"{outcome}: training n={len(sample):,}, {iterations} iterations, max |dlog mu| vs fepois {deviation:.2e}")
    qc += [("estimation", f"{outcome}_training_n", len(sample)), ("estimation", f"{outcome}_ipf_iterations", iterations),
           ("estimation", f"{outcome}_max_abs_dlogmu_vs_fepois", deviation)]
    return {d: pd.Series(log_fe[j], index=categories[d]) for j, d in enumerate(DIMS)}


def predict(keys, effects):
    total = np.zeros(len(keys))
    supported = np.ones(len(keys), bool)
    per_dim = {}
    for d in DIMS:
        values = keys[d].map(effects[d])
        per_dim[d] = values.notna().to_numpy()
        supported &= per_dim[d]
        total += values.fillna(0.0).to_numpy()
    return np.where(supported, np.exp(total), np.nan), supported, per_dim


def complete_flag(frame, clock, low, high):
    block = frame.loc[frame[clock].between(low, high)]
    g = block.groupby("city_event_id").counterfactual_supported.agg(["sum", "size"])
    return ((g["sum"] == g["size"]) & (g["size"] == high - low + 1)).astype("int64")

def estimate_counterfactual():
    qc = []
    daily = pd.read_csv(DAILY_PANEL, dtype={"city_id": "string"}, encoding=ENC)
    daily["date"] = pd.to_datetime(daily.date, format="%Y-%m-%d")
    daily["city_name"] = daily.city_name.astype(str).str.strip()
    events = pd.read_csv(EVENT_PANEL, encoding=ENC, dtype={"city_id": "string"})
    events["date"] = pd.to_datetime(events.date, format="%Y-%m-%d")
    event_columns = list(events.columns)

    window_days = events[["city_name", "date"]].drop_duplicates()
    holdout = set(map(tuple, window_days.values))
    daily["event_window_holdout"] = np.fromiter(((c, d) in holdout for c, d in zip(daily.city_name, daily.date)),
                                                bool, len(daily)).astype("int64")
    daily["datef"] = daily.date.dt.strftime("%Y-%m-%d")
    daily["city_moy"] = daily.city_name + "|" + daily.date.dt.month.astype(str)
    training = daily.loc[daily.event_window_holdout == 0].copy()
    print(f"city-days {len(daily):,} | held out {int(daily.event_window_holdout.sum()):,} | "
          f"training {len(training):,} | events {events.city_event_id.nunique()}")
    qc += [("training", "total_city_days", len(daily)), ("training", "holdout_city_days", len(holdout)),
           ("training", "training_city_days", len(training)),
           ("training", "focal_events", events.city_event_id.nunique())]

    effects = {outcome: fit_component(training, outcome, qc) for outcome in COMPONENTS}
    days = window_days.sort_values(["city_name", "date"]).reset_index(drop=True)
    days["datef"] = days.date.dt.strftime("%Y-%m-%d")
    days["city_moy"] = days.city_name + "|" + days.date.dt.month.astype(str)
    support = None
    for outcome, column in COMPONENTS.items():
        values, supported, per_dim = predict(days[DIMS], effects[outcome])
        days[column] = values
        if support is None:
            support, dim_support = supported, per_dim
        elif not np.array_equal(support, supported):
            raise SystemExit("support differs across components")
    days["counterfactual_supported"] = support.astype("int64")
    days["counterfactual_support_reason"] = np.where(
        support, "SUPPORTED",
        np.where(~dim_support["city_moy"] & ~dim_support["datef"], "UNSUPPORTED_CITY_MOY_AND_DATE",
                 np.where(~dim_support["city_moy"], "UNSUPPORTED_CITY_MOY", "UNSUPPORTED_DATE")))
    days["normal_resident_associated_spending"] = days.normal_local_spending + days.normal_outbound_spending
    days["normal_merchant_place_spending"] = days.normal_local_spending + days.normal_inbound_spending
    n_supported = int(support.sum())
    print(f"event-window city-days {len(days):,} | supported {n_supported:,} ({n_supported / len(days):.4%})")
    qc += [("support", "unique_event_window_city_days", len(days)), ("support", "supported_city_days", n_supported),
           ("support", "coverage", n_supported / len(days))]

    new_columns = [*COMPONENTS.values(), "normal_resident_associated_spending", "normal_merchant_place_spending",
                   "counterfactual_supported", "counterfactual_support_reason"]
    out = events.merge(days[["city_name", "date", *new_columns]], on=["city_name", "date"], how="left",
                       validate="many_to_one")
    if len(out) != len(events) or out.counterfactual_supported.isna().any():
        raise SystemExit("merge into the event-time panel failed")
    out["counterfactual_supported"] = out.counterfactual_supported.astype("int64")
    for name, (clock, low, high) in COMPLETENESS.items():
        out[name] = out.city_event_id.map(complete_flag(out, clock, low, high)).fillna(0).astype("int64")
        n = int(out.drop_duplicates("city_event_id")[name].sum())
        print(f"{name}: {n} of {events.city_event_id.nunique()} events")
        qc.append(("event_flags", name, n))
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    out = out[[*event_columns, *new_columns, *COMPLETENESS]]
    out.to_csv(CF_PANEL, index=False, encoding=ENC, float_format="%.9g", lineterminator="\n")
    pd.DataFrame(qc, columns=["block", "key", "value"]).to_csv(QC_OUTPUT, index=False, encoding=ENC,
                                                               float_format="%.15g", lineterminator="\n")



def main():
    build_event_time_panel()
    estimate_counterfactual()


if __name__ == "__main__":
    main()
