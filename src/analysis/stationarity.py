"""
Paso 4: Analisis de estacionaridad y transformaciones (2015-2026).
Pruebas ADF y KPSS sobre Close, log(Close), diferencias y log-retornos.
Guarda resultados en output/tables/.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import os

RAW_PATH = "db/raw/sbux.csv"
TABLE_DIR = "output/tables"
START_DATE = "2015-01-01"


def load_window(path: str, start: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[df.index >= start].copy()
    return df


def adf_test(series: pd.Series, label: str) -> dict:
    """Augmented Dickey-Fuller: H0 = serie tiene raiz unitaria (no estacionaria)."""
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
    """KPSS: H0 = serie es estacionaria (opuesto a ADF)."""
    result = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "serie": label,
        "test": "KPSS",
        "statistic": round(result[0], 6),
        "pvalue": round(result[1], 6),
        "critical_5pct": round(result[3]["5%"], 6),
        "is_stationary": result[1] > 0.05,  # KPSS: p > 0.05 = no rechazar H0 (estacionaria)
    }


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta ADF y KPSS sobre 4 transformaciones de Close."""
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
    """Interpreta los resultados combinados ADF + KPSS."""
    print("\n=== INTERPRETACION (ADF + KPSS combinados) ===")
    print("ADF: H0=no estacionaria | KPSS: H0=estacionaria")
    print()

    for serie in df_results["serie"].unique():
        row_adf = df_results[(df_results["serie"] == serie) & (df_results["test"] == "ADF")]
        row_kpss = df_results[(df_results["serie"] == serie) & (df_results["test"] == "KPSS")]

        adf_stationary = row_adf["is_stationary"].values[0]
        kpss_stationary = row_kpss["is_stationary"].values[0]

        if adf_stationary and kpss_stationary:
            conclusion = "ESTACIONARIA (confirmado por ambas pruebas)"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "NO ESTACIONARIA (confirmado por ambas pruebas)"
        elif adf_stationary and not kpss_stationary:
            conclusion = "AMBIGUO (ADF dice estacionaria, KPSS dice no) - posible tendencia estacionaria"
        else:
            conclusion = "AMBIGUO (ADF dice no estacionaria, KPSS dice estacionaria)"

        print(f"  {serie}: {conclusion}")
        print(f"    ADF stat={row_adf['statistic'].values[0]:.4f}, p={row_adf['pvalue'].values[0]:.6f}")
        print(f"    KPSS stat={row_kpss['statistic'].values[0]:.4f}, p={row_kpss['pvalue'].values[0]:.6f}")
        print()


def plot_transformations(df: pd.DataFrame) -> str:
    """Grafico comparativo de las 4 transformaciones."""
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

    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    path = "output/figures/eda_transformations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


if __name__ == "__main__":
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = load_window(RAW_PATH, START_DATE)

    results = run_tests(df)
    print(results.to_string(index=False))
    print()

    interpret_results(results)

    path = os.path.join(TABLE_DIR, "stationarity_tests.csv")
    results.to_csv(path, index=False)
    print(f"Tabla guardada: {path}")

    plot_transformations(df)
