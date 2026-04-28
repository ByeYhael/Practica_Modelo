"""
Paso 2: Estadistica descriptiva y deteccion de anomalias (2015-2026).
Genera tabla de estadisticos, detecta outliers via IQR y Z-score,
y guarda resultados en output/tables/.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "db/raw/sbux.csv"
TABLE_DIR = "output/tables"
START_DATE = "2015-01-01"
ZSCORE_THRESHOLD = 3.0


def load_window(path: str, start: str) -> pd.DataFrame:
    """Carga SBUX y recorta a ventana >= start."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[df.index >= start].copy()
    return df


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadisticos descriptivos para Close y Volume."""
    cols = ["Close", "Volume", "Open", "High", "Low"]
    stats = df[cols].describe().T
    stats["skew"] = df[cols].skew()
    stats["kurtosis"] = df[cols].kurtosis()
    stats["cv_pct"] = (stats["std"] / stats["mean"]) * 100  # coeficiente variacion
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
    """Detecta outliers usando Z-score sobre Close."""
    z = (df[col] - df[col].mean()) / df[col].std()
    outliers = df[z.abs() > threshold].copy()
    outliers["metodo"] = "Z-score"
    outliers["z_score"] = z[z.abs() > threshold]
    return outliers


def save_table(df: pd.DataFrame, filename: str) -> None:
    """Guarda DataFrame como CSV en output/tables/."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, filename)
    df.to_csv(path)
    print(f"Tabla guardada: {path}")


if __name__ == "__main__":
    df = load_window(RAW_PATH, START_DATE)
    print(f"Ventana EDA: {df.index.min()} -> {df.index.max()} ({len(df)} registros)\n")

    # 1. Estadisticos descriptivos
    stats = descriptive_stats(df)
    print("=== ESTADISTICOS DESCRIPTIVOS (2015-2026) ===")
    print(stats.to_string())
    print()

    # 2. Outliers - Metodo IQR (elegido para distribuciones asimetricas)
    for col in ["Close", "Volume"]:
        o_iqr = detect_outliers_iqr(df, col)
        o_z = detect_outliers_zscore(df, col, ZSCORE_THRESHOLD)
        print(f"=== OUTLIERS: {col} ===")
        print(f"  IQR:     {len(o_iqr)} outliers ({len(o_iqr)/len(df)*100:.2f}%)")
        print(f"  Z-score: {len(o_z)} outliers ({len(o_z)/len(df)*100:.2f}%)")
        if len(o_iqr) > 0:
            print(f"  Rango IQR normal: [{o_iqr['limite_inferior'].iloc[0]:.2f}, {o_iqr['limite_superior'].iloc[0]:.2f}]")
        print()

    save_table(stats, "estadisticos_descriptivos.csv")

    # 3. Guardar outliers identificados
    outliers_close = detect_outliers_iqr(df, "Close")
    outliers_volume = detect_outliers_iqr(df, "Volume")
    save_table(outliers_close[["Close", "Volume"]], "outliers_close_iqr.csv")
    save_table(outliers_volume[["Close", "Volume"]], "outliers_volume_iqr.csv")
