"""Build the city-day UnionPay spending panel for 225 cities, 2017-2019.

The provider's amount indices are log-transformed, cval = 10 ln(1000 A / B + 1), with one common benchmark B. Each
index is inverted with expm1(cval / 10) so that local, inbound, outbound and total spending are additive.
"""
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(r"E:\Extreme Rainfall and UnionPay Spending")
DATASET = ROOT / "Dataset"
INTERMEDIATE = ROOT / "Intermediate"

RAW_INDEX = DATASET / "城市综合活力指数_日度.csv"          # provider file: daily city vitality indices
DICTIONARY = DATASET / "城市活力指数数据说明附件.xlsx"      # provider data dictionary; sheet 6 lists city codes
OUTPUT = INTERMEDIATE / "unionpay_city_daily_2017_2019.csv"

START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")
N_CITIES, N_DAYS = 225, 1095
VALUE_COLUMNS = ["local_spending", "inbound_spending", "outbound_spending", "resident_associated_spending",
                 "merchant_place_spending"]


def load_city_codes():
    workbook = openpyxl.load_workbook(DICTIONARY, read_only=True, data_only=True)
    codes = pd.DataFrame(list(workbook.worksheets[5].values)[1:], columns=["province", "city_name", "city_id"])
    workbook.close()
    codes = codes.dropna(subset=["city_id", "city_name"]).copy()
    codes["city_id"] = codes.city_id.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    codes["city_name"] = codes.city_name.astype(str).str.strip()
    if codes.city_id.duplicated().any() or codes.city_name.duplicated().any():
        raise ValueError("city-code dictionary is not one-to-one")
    return dict(zip(codes.city_id, codes.city_name))


def invert(index_values):
    return np.expm1(index_values.to_numpy(dtype="float64") / 10.0)


def main():
    names = load_city_codes()
    raw = pd.read_csv(RAW_INDEX, usecols=["mchnt_city", "date", "cval", "cval_in", "cval_out"],
                      dtype={"mchnt_city": "string"})
    raw["city_id"] = raw.mchnt_city.str.strip()
    raw["date"] = pd.to_datetime(raw.date, format="%d%b%Y")
    for column in ("cval", "cval_in", "cval_out"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    panel = raw.loc[raw.date.between(START, END)].copy()
    panel["city_name"] = panel.city_id.map(names)
    if panel.city_name.isna().any():
        raise ValueError(f"city codes missing from the dictionary: {sorted(panel.loc[panel.city_name.isna(), 'city_id'].unique())}")

    panel["merchant_place_spending"] = invert(panel.cval)
    panel["inbound_spending"] = invert(panel.cval_in)
    panel["outbound_spending"] = invert(panel.cval_out)
    panel["local_spending"] = panel.merchant_place_spending - panel.inbound_spending
    panel["resident_associated_spending"] = panel.local_spending + panel.outbound_spending
    panel = panel.sort_values(["city_name", "date"], kind="mergesort").reset_index(drop=True)

    if (panel.city_id.nunique(), panel.date.nunique(), len(panel)) != (N_CITIES, N_DAYS, N_CITIES * N_DAYS):
        raise SystemExit("the panel is not a balanced 225 x 1,095 city-day panel")
    if panel.duplicated(["city_id", "date"]).any():
        raise SystemExit("duplicate city-days")
    shares = pd.DataFrame({"inbound_share": panel.inbound_spending / panel.merchant_place_spending,
                           "outbound_share": panel.outbound_spending / panel.resident_associated_spending})
    print(f"rows {len(panel):,} | negative local spending {int((panel.local_spending < 0).sum())} | "
          f"median inbound share {shares.inbound_share.median():.6f} | "
          f"median outbound share {shares.outbound_share.median():.6f} | "
          f"2017 mean total {panel.loc[panel.date.dt.year == 2017, 'merchant_place_spending'].mean():.6f}")

    out = panel[["city_id", "city_name", "date", *VALUE_COLUMNS]].copy()
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT, index=False, encoding="utf-8", float_format="%.9g", lineterminator="\n")
    print(f"wrote {OUTPUT.name} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
