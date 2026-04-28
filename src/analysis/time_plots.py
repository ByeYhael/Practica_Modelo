"""
Paso 3: Visualizacion temporal basica (2015-2026).
Genera:
  - Close vs tiempo (linea)
  - Volatilidad movil (rolling std, ventana 20d)
  - Volume con y sin transformacion log
Guarda figuras en output/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

RAW_PATH = "db/raw/sbux.csv"
FIG_DIR = "output/figures"
START_DATE = "2015-01-01"
ROLLING_WINDOWS = [5, 20, 60]


def load_window(path: str, start: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[df.index >= start].copy()
    return df


def plot_close_timeseries(df: pd.DataFrame) -> str:
    """Grafico de linea: Close vs tiempo."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], linewidth=0.8, color="steelblue")
    ax.set_title("SBUX - Precio de Cierre (Close) 2015-2026")
    ax.set_ylabel("Precio ($)")
    ax.set_xlabel("Fecha")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "eda_close_timeseries.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


def plot_rolling_volatility(df: pd.DataFrame, windows: list) -> str:
    """Volatilidad movil: rolling std con varias ventanas."""
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 8), sharex=True)
    returns = df["Close"].pct_change() * 100  # retornos en %

    for ax, w in zip(axes, windows):
        vol = returns.rolling(w).std()
        ax.plot(df.index, vol, linewidth=0.7, label=f"Rolling std {w}d")
        ax.set_ylabel(f"Vol {w}d (%)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Fecha")
    axes[0].set_title("SBUX - Volatilidad Movil (Retornos %)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "eda_rolling_volatility.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


def plot_volume_transform(df: pd.DataFrame) -> str:
    """Comparacion Volume raw vs log(Volume)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(df.index, df["Volume"], linewidth=0.6, color="orange")
    ax1.set_title("SBUX - Volumen Transado (raw)")
    ax1.set_ylabel("Volumen")
    ax1.grid(True, alpha=0.3)

    log_vol = np.log1p(df["Volume"])  # log(1+Volume) para evitar log(0)
    ax2.plot(df.index, log_vol, linewidth=0.6, color="green")
    ax2.set_title("SBUX - Volumen Transado (log-transformado)")
    ax2.set_ylabel("log(Volumen)")
    ax2.set_xlabel("Fecha")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "eda_volume_transform.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_window(RAW_PATH, START_DATE)

    plot_close_timeseries(df)
    plot_rolling_volatility(df, ROLLING_WINDOWS)
    plot_volume_transform(df)
