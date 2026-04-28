"""
Split temporal y escalado de features para SBUX.
Split: Train <=2023, Val 2024, Test 2025-2026 (NO aleatorio).
Escalado: StandardScaler fit en TRAIN, transform en VAL y TEST.
Input:  db/processed/features_v1.csv
Output: db/processed/scaled_data.npz, output/models/scaler.joblib
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Configuracion
FEATURES_PATH = "db/processed/features_v1.csv"
SCALER_PATH = "output/models/scaler.joblib"
OUTPUT_NPZ = "db/processed/scaled_data.npz"
RANDOM_STATE = 42

# Limites temporales (justificados por EDA)
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"

# Features
FEATURES = ["volatilidad_20d", "volatilidad_5d", "rango_diario", "log_volume",
            "day_of_week", "month", "lag_1", "lag_5", "lag_20"]
TARGET = "log_return"


def load_and_split(path: str):
    """Carga features y divide temporalmente (NO aleatorio)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df_train = df.loc[:TRAIN_END].copy()
    df_val = df.loc[VAL_START:VAL_END].copy()
    df_test = df.loc[TEST_START:].copy()

    print(f"Train: {len(df_train)} registros ({df_train.index.min().date()} -> {df_train.index.max().date()})")
    print(f"Val:   {len(df_val)} registros ({df_val.index.min().date()} -> {df_val.index.max().date()})")
    print(f"Test:  {len(df_test)} registros ({df_test.index.min().date()} -> {df_test.index.max().date()})")
    print(f"Total: {len(df_train) + len(df_val) + len(df_test)}")

    return df_train, df_val, df_test


def scale_features(df_train, df_val, df_test):
    """Escala features con StandardScaler. Fit en TRAIN, transform en VAL y TEST."""
    X_train = df_train[FEATURES].values
    y_train = df_train[TARGET].values
    X_val = df_val[FEATURES].values
    y_val = df_val[TARGET].values
    X_test = df_test[FEATURES].values
    y_test = df_test[TARGET].values

    # Fit SOLO en train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Verificar data leakage
    print(f"\nVerificacion de data leakage:")
    print(f"  Media train (debe ser ~0): {X_train_scaled.mean(axis=0).round(6)}")
    print(f"  Media val (debe ser !=0):  {X_val_scaled.mean(axis=0).round(6)}")
    print(f"  Media test (debe ser !=0): {X_test_scaled.mean(axis=0).round(6)}")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler


if __name__ == "__main__":
    print("Cargando datos...")
    df_train, df_val, df_test = load_and_split(FEATURES_PATH)

    print("\nEscalando features...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = scale_features(
        df_train, df_val, df_test
    )

    # Guardar scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nScaler guardado: {SCALER_PATH}")

    # Guardar datos escalados
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             feature_names=FEATURES)
    print(f"Datos escalados guardados: {OUTPUT_NPZ}")
    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
