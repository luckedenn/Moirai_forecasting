# prepare_datasets.py
import os
import io
import sys
import math
import time
import zipfile
from datetime import datetime

import pandas as pd
import requests

# --- Utils ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def time_split(df, ts_col, frac_train=0.70, frac_val=0.15):
    """Split by time order: 70% train, 15% val, 15% test."""
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test

def save_splits(df, domain_name):
    outdir = os.path.join(DATA_DIR, domain_name)
    os.makedirs(outdir, exist_ok=True)
    train, val, test = time_split(df, "timestamp")
    df.to_csv(os.path.join(outdir, f"{domain_name}_full.csv"), index=False)
    train.to_csv(os.path.join(outdir, f"{domain_name}_train.csv"), index=False)
    val.to_csv(os.path.join(outdir, f"{domain_name}_val.csv"), index=False)
    test.to_csv(os.path.join(outdir, f"{domain_name}_test.csv"), index=False)
    print(f"[{domain_name}] rows: full={len(df)}, train={len(train)}, val={len(val)}, test={len(test)}  "
          f"range: {df['timestamp'].min().date()} .. {df['timestamp'].max().date()}")

# --- 1) WEATHER: Daily Minimum Temperatures (Melbourne) ---
def prep_weather():
    # Brownlee raw CSV (tiny, ~30 KB)
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Standardize columns
    # Input columns: Date, Temp
    df = df.rename(columns={"Date": "timestamp", "Temp": "value"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "value"]].dropna()
    # Optional: keep as daily frequency as-is
    save_splits(df, "weather_melbourne")

# --- 2) CO2: Mauna Loa monthly means (NOAA GML) ---
def prep_co2():
    import io, re
    import pandas as pd
    import requests

    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    raw = r.text

    # 1) Buang baris komentar (#) dan baris kosong
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

    # 2) NOAA biasanya meletakkan header kolom di baris pertama non-komentar.
    #    Pastikan pemisah adalah koma (kadang spasi + koma).
    clean = "\n".join(lines)
    df = pd.read_csv(
        io.StringIO(clean),
        sep=r"\s*,\s*",  # tolerate spaces around commas
        engine="python"
    )

    # 3) Normalisasi nama kolom (beragam tergantung versi file)
    #    Skema umum: year, month, decimal_date, average, interpolated, trend, days
    lower_cols = {c.lower().strip(): c for c in df.columns}
    # Pastikan kolom year & month ada
    if "year" not in lower_cols or "month" not in lower_cols:
        # Kadang header tidak sesuai; coba pakai header manual jika jumlah kolom >= 7
        df = pd.read_csv(
            io.StringIO(clean),
            sep=r"\s*,\s*",
            engine="python",
            header=None
        )
        if df.shape[1] >= 7:
            df = df.iloc[:, :7]
            df.columns = ["year", "month", "decimal_date", "average", "interpolated", "trend", "days"]
        else:
            raise ValueError("Format CSV NOAA tidak terduga; periksa file secara manual.")

    # 4) Coerce tipe numerik dan seleksi kolom “average”
    for col in df.columns:
        if col.lower() in ("year", "month", "decimal_date", "average", "interpolated", "trend", "days"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Gunakan 'average' bila tersedia, kalau tidak pakai 'interpolated'
    val_col = "average" if "average" in df.columns else ("interpolated" if "interpolated" in df.columns else None)
    if val_col is None:
        raise ValueError("Kolom nilai CO₂ ('average' atau 'interpolated') tidak ditemukan.")

    # Buang nilai sentinel -99.99 dan NaN
    df = df[(df[val_col].notna()) & (df[val_col] != -99.99)]
    # 5) Bangun timestamp (pakai hari=1)
    #    Pastikan kolom year & month ada (setelah normalisasi di atas)
    ycol = "year" if "year" in df.columns else lower_cols.get("year")
    mcol = "month" if "month" in df.columns else lower_cols.get("month")
    if ycol is None or mcol is None:
        raise ValueError("Kolom 'year' atau 'month' tidak ditemukan setelah parsing.")

    df["timestamp"] = pd.to_datetime(
        df[[ycol, mcol]].rename(columns={ycol: "year", mcol: "month"}).assign(day=1),
        errors="coerce"
    )
    df = df[["timestamp", val_col]].rename(columns={val_col: "value"}).dropna().sort_values("timestamp")

    save_splits(df, "co2_maunaloa_monthly")


# --- 3) FINANCE: AAPL daily close (Yahoo Finance via yfinance) ---
def prep_finance():
    try:
        import yfinance as yf
    except Exception:
        print("[finance] yfinance not installed. Installing...")
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf

    # Date range: a few years to keep it lightweight
    df = yf.download("AAPL", start="2015-01-01")  # daily by default
    if df.empty:
        raise RuntimeError("yfinance returned empty DataFrame (check internet/firewall).")
    df = df.reset_index().rename(columns={"Date": "timestamp", "Close": "value"})
    df = df[["timestamp", "value"]].dropna()
    save_splits(df, "finance_aapl")

if __name__ == "__main__":
    print("Preparing datasets into ./data ...")
    prep_weather()
    prep_co2()
    prep_finance()
    print("Done. Files saved under ./data/<domain>/*.csv")
