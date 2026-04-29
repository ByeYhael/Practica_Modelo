"""
Paso 4: Analisis de estacionaridad y transformaciones.
Pruebas ADF y KPSS sobre Close, log(Close), diferencias y log-retornos.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import os

RAW_PATH = "db/raw/sbux.csv"
TABLE_DIR = "output/tables"
TABLE_SUFFIX = "v2"


def load_window(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()
    return df


def adf_test(series: pd.Series, label: str) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "serie": label,
        "test": "ADF",
        "statistic": round(result[0], 6),
        "pvalue": round(result[1], 6),
        "critical_1pct": round(result[4]["1%"], 6),
        "critical_5pct": round(result[4]["5%"], 6),
        "critical_10pct": round(result[4]["10%"], 6),
        "is_stationary": result[1] < 0.05,
        "n_lags": result[2],
    }


def kpss_test(series: pd.Series, label: str) -> dict:
    result = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "serie": label,
        "test": "KPSS",
        "statistic": round(result[0], 6),
        "pvalue": round(result[1], 6),
        "critical_5pct": round(result[3]["5%"], 6),
        "is_stationary": result[1] > 0.05,
    }


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    series_list = {
        "Close": df["Close"],
        "log(Close)": np.log(df["Close"]),
        "diff(Close)": df["Close"].diff(),
        "log_return": np.log(df["Close"]).diff(),
    }
    results = []
    for name, series in series_list.items():
        results.append(adf_test(series, name))
        results.append(kpss_test(series, name))
    return pd.DataFrame(results)


def interpret_results(df_results: pd.DataFrame) -> None:
    print("\n=== INTERPRETACION (ADF + KPSS combinados) ===")
    for serie in df_results["serie"].unique():
        row_adf = df_results[(df_results["serie"] == serie) & (df_results["test"] == "ADF")]
        row_kpss = df_results[(df_results["serie"] == serie) & (df_results["test"] == "KPSS")]
        adf_s = row_adf["is_stationary"].values[0]
        kpss_s = row_kpss["is_stationary"].values[0]
        if adf_s and kpss_s:
            conclusion = "ESTACIONARIA"
        elif not adf_s and not kpss_s:
            conclusion = "NO ESTACIONARIA"
        else:
            conclusion = "AMBIGUO"
        print(f"  {serie}: {conclusion}")
        print(f"    ADF  stat={row_adf['statistic'].values[0]:.4f}, p={row_adf['pvalue'].values[0]:.6f}")
        print(f"    KPSS stat={row_kpss['statistic'].values[0]:.4f}, p={row_kpss['pvalue'].values[0]:.6f}")
        print()


def plot_transformations(df: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df.index, df["Close"], linewidth=0.7, color="steelblue")
    axes[0].set_ylabel("Close ($)")
    axes[0].set_title("Original: Close")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df.index, np.log(df["Close"]), linewidth=0.7, color="green")
    axes[1].set_ylabel("log(Close)")
    axes[1].set_title("Transformacion: log(Close)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(df.index, df["Close"].diff(), linewidth=0.7, color="red")
    axes[2].set_ylabel("diff(Close)")
    axes[2].set_title("Diferenciacion: diff(Close)")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(df.index, np.log(df["Close"]).diff(), linewidth=0.7, color="purple")
    axes[3].set_ylabel("log-return")
    axes[3].set_title("Log-retornos: diff(log(Close))")
    axes[3].set_xlabel("Fecha")
    axes[3].grid(True, alpha=0.3)
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    path = f"output/figures/eda_transformations_{TABLE_SUFFIX}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


if __name__ == "__main__":
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs("output/figures", exist_ok=True)

    df = load_window(RAW_PATH, "2025-02-01", "2026-01-31")
    results = run_tests(df)
    print(results.to_string(index=False))
    print()
    interpret_results(results)
    results.to_csv(os.path.join(TABLE_DIR, f"stationarity_tests_{TABLE_SUFFIX}.csv"), index=False)
    plot_transformations(df)
