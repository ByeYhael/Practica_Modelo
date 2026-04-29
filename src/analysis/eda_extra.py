"""
Analisis visual complementario al EDA.
Heatmap de correlacion, ACF/PACF, y rango diario vs Close.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy.polynomial.polynomial import polyfit
import os

RAW_PATH = "db/raw/sbux.csv"
FIG_DIR = "output/figures"
FIG_SUFFIX = "v2"


def load_window(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()
    return df


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[["Open", "High", "Low", "Close", "Volume"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".4f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Correlacion OHLCV (feb 2025 - ene 2026)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"eda_correlation_heatmap_{FIG_SUFFIX}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


def plot_acf_pacf(df: pd.DataFrame) -> str:
    lr = np.log(df["Close"]).diff().dropna()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(lr, lags=40, alpha=0.05, ax=ax1, title="ACF - log_return (40 lags)")
    plot_pacf(lr, lags=40, alpha=0.05, ax=ax2, title="PACF - log_return (40 lags)", method="ywm")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"eda_acf_pacf_{FIG_SUFFIX}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


def plot_range_vs_close(df: pd.DataFrame) -> str:
    df_plot = df.copy()
    df_plot["daily_range"] = df_plot["High"] - df_plot["Low"]
    df_plot["log_vol"] = np.log1p(df_plot["Volume"])

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_plot["daily_range"], df_plot["Close"],
                         c=df_plot["log_vol"], cmap="viridis", alpha=0.6, s=15)
    fig.colorbar(scatter, ax=ax, label="log(Volume)")
    ax.set_xlabel("Rango diario (High - Low) [$]")
    ax.set_ylabel("Precio Close [$]")
    ax.set_title("Rango Diario vs Precio Close (color = Volumen)")
    ax.grid(True, alpha=0.3)

    x = df_plot["daily_range"].values
    y = df_plot["Close"].values
    b, m = polyfit(x, y, 1)
    ax.plot(x, b + m * x, color="red", linewidth=1, linestyle="--",
            label=f"Tendencia (r={df_plot['daily_range'].corr(df_plot['Close']):.2f})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"eda_range_vs_close_{FIG_SUFFIX}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_window(RAW_PATH, "2025-02-01", "2026-01-31")
    plot_correlation_heatmap(df)
    plot_acf_pacf(df)
    plot_range_vs_close(df)
