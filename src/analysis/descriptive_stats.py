"""
Paso 2: Estadistica descriptiva y deteccion de anomalias.
Genera tabla de estadisticos, detecta outliers via IQR y Z-score,
y guarda resultados en output/tables/.
La ventana se define via funciones para reutilizar en diferentes versiones.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "db/raw/sbux.csv"
TABLE_DIR = "output/tables"
ZSCORE_THRESHOLD = 3.0
TABLE_SUFFIX = "v2"


def load_window(path: str, start: str, end: str) -> pd.DataFrame:
    """Carga SBUX y recorta a ventana [start, end]."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()
    return df


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadisticos descriptivos para Close y Volume."""
    cols = ["Close", "Volume", "Open", "High", "Low"]
    stats = df[cols].describe().T
    stats["skew"] = df[cols].skew()
    stats["kurtosis"] = df[cols].kurtosis()
    stats["cv_pct"] = (stats["std"] / stats["mean"]) * 100
    return stats.round(4)


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Detecta outliers usando rango intercuartil (IQR)."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].copy()
    outliers["metodo"] = "IQR"
    outliers["limite_inferior"] = lower
    outliers["limite_superior"] = upper
    return outliers


def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Detecta outliers usando Z-score."""
    z = (df[col] - df[col].mean()) / df[col].std()
    outliers = df[z.abs() > threshold].copy()
    outliers["metodo"] = "Z-score"
    outliers["z_score"] = z[z.abs() > threshold]
    return outliers


def save_table(df: pd.DataFrame, base_name: str) -> None:
    """Guarda DataFrame como CSV con sufijo de version."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    filename = f"{base_name}_{TABLE_SUFFIX}.csv"
    path = os.path.join(TABLE_DIR, filename)
    df.to_csv(path)
    print(f"Tabla guardada: {path}")


if __name__ == "__main__":
    START = "2025-02-01"
    END = "2026-01-31"

    df = load_window(RAW_PATH, START, END)
    print(f"Ventana EDA: {df.index.min()} -> {df.index.max()} ({len(df)} registros)\n")

    stats = descriptive_stats(df)
    print("=== ESTADISTICOS DESCRIPTIVOS ===")
    print(stats.to_string())
    print()

    for col in ["Close", "Volume"]:
        o_iqr = detect_outliers_iqr(df, col)
        o_z = detect_outliers_zscore(df, col, ZSCORE_THRESHOLD)
        print(f"=== OUTLIERS: {col} ===")
        print(f"  IQR:     {len(o_iqr)} outliers ({len(o_iqr)/len(df)*100:.2f}%)")
        print(f"  Z-score: {len(o_z)} outliers ({len(o_z)/len(df)*100:.2f}%)")
        if len(o_iqr) > 0:
            print(f"  Rango IQR normal: [{o_iqr['limite_inferior'].iloc[0]:.2f}, {o_iqr['limite_superior'].iloc[0]:.2f}]")
        print()

    save_table(stats, "estadisticos_descriptivos")
    outliers_close = detect_outliers_iqr(df, "Close")
    outliers_volume = detect_outliers_iqr(df, "Volume")
    save_table(outliers_close[["Close", "Volume"]], "outliers_close_iqr")
    save_table(outliers_volume[["Close", "Volume"]], "outliers_volume_iqr")

    # Guardar datos limpios de esta ventana
    out_path = "db/processed/cleaned_sbux_v2.csv"
    os.makedirs("db/processed", exist_ok=True)
    df.to_csv(out_path)
    print(f"Datos guardados: {out_path}")
