"""
Split temporal y escalado de features para SBUX.
Recibe la version como argumento (v1/v2 usa ventana larga 2015-2023, v3 usa ultimo año).
Escalado: StandardScaler fit en TRAIN, transform en VAL y TEST.
Output: db/processed/scaled_data_VERSION.npz, output/models/scaler_VERSION.joblib
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Features
FEATURES = ["volatilidad_20d", "volatilidad_5d", "rango_diario", "log_volume",
            "day_of_week", "month", "lag_1", "lag_5", "lag_20"]
TARGET = "log_return"
RANDOM_STATE = 42


def load_and_split_v1v2(path: str):
    """Split para v1/v2: Train <=2023, Val 2024, Test 2025-2026."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df_train = df.loc[: "2023-12-31"].copy()
    df_val = df.loc["2024-01-01":"2024-12-31"].copy()
    df_test = df.loc["2025-01-01":].copy()
    return df_train, df_val, df_test


def load_and_split_v3(path: str):
    """Split para v3: ultimo año 70/15/15 temporal."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Filtrar solo el ultimo año
    df = df.loc["2025-02-03":].copy()
    total = len(df)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    return df_train, df_val, df_test


def scale_features(df_train, df_val, df_test):
    """Escala features con StandardScaler. Fit en TRAIN, transform en VAL y TEST."""
    X_train = df_train[FEATURES].values
    y_train = df_train[TARGET].values
    X_val = df_val[FEATURES].values
    y_val = df_val[TARGET].values
    X_test = df_test[FEATURES].values
    y_test = df_test[TARGET].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nVerificacion de data leakage:")
    print(f"  Media train (debe ser ~0): {X_train_scaled.mean(axis=0).round(6)}")
    print(f"  Media val (debe ser !=0):  {X_val_scaled.mean(axis=0).round(6)}")
    print(f"  Media test (debe ser !=0): {X_test_scaled.mean(axis=0).round(6)}")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"

    if version == "v3":
        FEATURES_PATH = "db/processed/features_last_year.csv"
        SCALER_PATH = f"output/models/scaler_{version}.joblib"
        OUTPUT_NPZ = f"db/processed/scaled_data_{version}.npz"
        print(f"Cargando datos de ultimo año...")
        df_train, df_val, df_test = load_and_split_v3(FEATURES_PATH)
    else:
        FEATURES_PATH = "db/processed/features_v1.csv"
        SCALER_PATH = f"output/models/scaler_{version}.joblib"
        OUTPUT_NPZ = f"db/processed/scaled_data_{version}.npz"
        print(f"Cargando datos de ventana larga...")
        df_train, df_val, df_test = load_and_split_v1v2(FEATURES_PATH)

    print(f"Train: {len(df_train)} reg ({df_train.index.min().date()} -> {df_train.index.max().date()})")
    print(f"Val:   {len(df_val)} reg ({df_val.index.min().date()} -> {df_val.index.max().date()})")
    print(f"Test:  {len(df_test)} reg ({df_test.index.min().date()} -> {df_test.index.max().date()})")

    print("\nEscalando features...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = scale_features(df_train, df_val, df_test)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nScaler guardado: {SCALER_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             feature_names=FEATURES)
    print(f"Datos escalados guardados: {OUTPUT_NPZ}")
    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
