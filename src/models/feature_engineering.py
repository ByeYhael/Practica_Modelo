"""
Feature Engineering para prediccion SBUX.
Construye features derivadas: log_return, volatilidad, rango, log_volume, lags, dummies.
Input:  db/processed/cleaned_sbux_v1.csv
Output: db/processed/features_v1.csv
"""

import pandas as pd
import numpy as np
import os

# Configuracion
DATA_PATH = "db/processed/cleaned_sbux_v1.csv"
OUTPUT_PATH = "db/processed/features_v1.csv"
WINDOW_START = "2015-01-01"
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye todas las features derivadas a partir de OHLCV."""
    df = df.copy()

    # Target: log_return (estacionario, estandar financiero)
    df["log_return"] = np.log(df["Close"]).diff()

    # Volatilidad (rolling std de log_return)
    df["volatilidad_20d"] = df["log_return"].rolling(window=20).std()
    df["volatilidad_5d"] = df["log_return"].rolling(window=5).std()

    # Rango diario (High - Low) - evita multicolinealidad de OHLC
    df["rango_diario"] = df["High"] - df["Low"]

    # Log del volumen - comprime cola larga (EDA: skew 9.06)
    df["log_volume"] = np.log(df["Volume"])

    # Dummies temporales
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # Lags de retornos (baja expectativa predictiva segun EDA)
    df["lag_1"] = df["log_return"].shift(1)
    df["lag_5"] = df["log_return"].shift(5)
    df["lag_20"] = df["log_return"].shift(20)

    return df


if __name__ == "__main__":
    # Cargar datos limpios
    print(f"Cargando: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Ventana EDA: 2015-2026
    df = df.loc[WINDOW_START:].copy()
    print(f"Shape inicial (2015-2026): {df.shape}")

    # Construir features
    df = build_features(df)

    # Definir columnas del modelo
    features = ["volatilidad_20d", "volatilidad_5d", "rango_diario", "log_volume",
                "day_of_week", "month", "lag_1", "lag_5", "lag_20"]
    target = "log_return"

    # Reportar NaN antes de dropear
    print(f"\nNaN antes de dropear:")
    for col in features + [target]:
        n_nan = df[col].isna().sum()
        print(f"  {col}: {n_nan} NaN ({n_nan/len(df)*100:.2f}%)")

    # Dropear filas con NaN (generados por lags/rolling)
    df_clean = df.dropna(subset=features + [target])
    print(f"\nFilas eliminadas por NaN: {len(df) - len(df_clean)} ({ (len(df)-len(df_clean))/len(df)*100:.2f}%)")
    print(f"Shape final: {df_clean.shape}")

    # Guardar
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH)
    print(f"Guardado: {OUTPUT_PATH}")
    print(f"Rango fechas: {df_clean.index.min()} a {df_clean.index.max()}")
