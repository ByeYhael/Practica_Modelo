"""
Paso 1: Carga y validacion inicial del dataset SBUX.
Carga db/raw/sbux.csv, convierte Date a indice datetime,
ordena cronologicamente y reporta metadatos basicos.
"""

import pandas as pd
import os


RAW_PATH = "db/raw/sbux.csv"
PROCESSED_PATH = "db/processed/cleaned_sbux_v1.csv"


def load_sbux(path: str) -> pd.DataFrame:
    """Carga CSV de SBUX, parsea Date como datetime y la asigna como indice."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Reporta metadatos de validacion: shape, tipos, nulos, rango, duplicados."""
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "nulls": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "date_range": (df.index.min(), df.index.max()),
        "duplicated_rows": df.duplicated().sum(),
        "total_days": (df.index.max() - df.index.min()).days,
    }
    return info


def save_clean(df: pd.DataFrame, path: str) -> None:
    """Guarda version preliminar limpia."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Guardado: {path}")


if __name__ == "__main__":
    df = load_sbux(RAW_PATH)
    info = validate(df)

    print(f"Shape: {info['shape']}")
    print(f"Rango fechas: {info['date_range'][0]} -> {info['date_range'][1]}")
    print(f"Dias totales: {info['total_days']}")
    print(f"Duplicados: {info['duplicated_rows']}")
    print("Nulos por columna:")
    for col, n in info["nulls"].items():
        print(f"  {col}: {n} ({info['null_pct'][col]}%)")
    print("Tipos de dato:")
    for col, t in info["dtypes"].items():
        print(f"  {col}: {t}")

    save_clean(df, PROCESSED_PATH)
