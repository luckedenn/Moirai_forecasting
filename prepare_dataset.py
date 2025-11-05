import os
import io
from datetime import datetime
import pandas as pd
import requests

# Folder untuk menyimpan data hasil preprocessing
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# fungsi membagi data menjadi train, val, test berdasarkan waktu
def time_split(df, ts_col, frac_train=0.70, frac_val=0.15):
    # Urutkan dataset berdasarkan kolom waktu
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)

    #Iris data menjadi train, val, test
    train = df.iloc[:n_train]               #70%
    val = df.iloc[n_train:n_train + n_val]  #15%
    test = df.iloc[n_train + n_val:]        #15%
    return train, val, test

#Fungsi menyimpan hasil split ke file CSV
def save_splits(df, domain_name):
    outdir = os.path.join(DATA_DIR, domain_name)
    os.makedirs(outdir, exist_ok=True)

    # Membagi data
    train, val, test = time_split(df, "timestamp")

    #Simpan menjadi 4 file CSV
    df.to_csv(os.path.join(outdir, f"{domain_name}_full.csv"), index=False)
    train.to_csv(os.path.join(outdir, f"{domain_name}_train.csv"), index=False)
    val.to_csv(os.path.join(outdir, f"{domain_name}_val.csv"), index=False)
    test.to_csv(os.path.join(outdir, f"{domain_name}_test.csv"), index=False)

    # print ringkasan ukuran data
    print(f"[{domain_name}] rows: full={len(df)}, train={len(train)}, val={len(val)}, test={len(test)}  "
          f"range: {df['timestamp'].min().date()} .. {df['timestamp'].max().date()}")

# Dataset Weather (Harian)
def prep_weather():
    # URL dataset cuaca harian dari Brownlee
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    # Download CSV
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # Baca CSV ke DataFrame
    df = pd.read_csv(io.StringIO(r.text))
    # Standarisasi nama kolom
    df = df.rename(columns={"Date": "timestamp", "Temp": "value"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Pilih kolom yang diperlukan dan buang baris kosong
    df = df[["timestamp", "value"]].dropna()
    # Simpan hasil split ke file CSV
    save_splits(df, "weather_melbourne")

# Dataset CO2 (Bulanan)
def prep_co2():
    import io, re
    import pandas as pd
    import requests
    # URL dataset CO2 bulanan dari NOAA GML
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    raw = r.text

    # Hapus baris komentar (#) dan baris kosong
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    # Gabungkan kembali menjadi CSV bersih
    clean = "\n".join(lines)
    df = pd.read_csv(io.StringIO(clean), sep=r"\s*,\s*", engine="python")

    # Pastikan ada kolom year dan month
    lower_cols = {c.lower().strip(): c for c in df.columns}
    if "year" not in lower_cols or "month" not in lower_cols:
        # Jika header hilang, kita buat manual
        df = pd.read_csv(io.StringIO(clean), sep=r"\s*,\s*", engine="python", header=None)
        if df.shape[1] >= 7:
            df = df.iloc[:, :7]
            df.columns = ["year", "month", "decimal_date", "average", "interpolated", "trend", "days"]
        else:
            raise ValueError("Format CSV NOAA tidak terduga; periksa file secara manual.")

    # Konversi kolom numerik ke tipe numerik
    for col in df.columns:
        if col.lower() in ("year", "month", "decimal_date", "average", "interpolated", "trend", "days"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Gunakan 'average' bila tersedia, kalau tidak pakai 'interpolated'
    val_col = "average" if "average" in df.columns else ("interpolated" if "interpolated" in df.columns else None)
    if val_col is None:
        raise ValueError("Kolom nilai CO₂ ('average' atau 'interpolated') tidak ditemukan.")
    # Buang baris dengan nilai kosong atau -99.99
    df = df[(df[val_col].notna()) & (df[val_col] != -99.99)]
    # Buat kolom timestamp dari year dan month
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


# Dataset Saham AAPL (Harian)
def prep_finance():
    try:
        import yfinance as yf
    except Exception:
        print("[finance] yfinance not installed. Installing...")
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf

    # Download data saham AAPL 
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
