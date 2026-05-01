"""
Genera graficos de predicciones vs real para RF y Stacking RF+LR.
Recibe version como argumento (v1/v2/v3).
Output en output/figures/models/ con sufijo _VERSION.
Uso: python3 src/models/plot_predictions.py v3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import os
import sys

# Configuracion
version = sys.argv[1] if len(sys.argv) > 1 else "v1"
RF_PRED_PATH = f"db/processed/rf_predictions_{version}.npz"
LR_MODEL_PATH = f"output/models/lr_meta_model_{version}.joblib"
FEATURES_PATH = "db/processed/features_last_year.csv" if version == "v3" else "db/processed/features_v1.csv"
FIG_DIR = "output/figures/models"


def reconvertir_a_close(y_log_return, df_original, start_idx):
    """Reconvierte log_return a precio Close: Close_pred = Close_actual * exp(log_return_pred)."""
    close_real = df_original["Close"].values[start_idx:start_idx + len(y_log_return)]
    close_prev = df_original["Close"].values[start_idx - 1] if start_idx > 0 else close_real[0]
    close_pred = np.zeros_like(y_log_return)

    for i in range(len(y_log_return)):
        if i == 0:
            close_pred[i] = close_prev * np.exp(y_log_return[i])
        else:
            close_pred[i] = close_pred[i-1] * np.exp(y_log_return[i])
    return close_real, close_pred


if __name__ == "__main__":
    # Cargar datos
    data = np.load(RF_PRED_PATH)
    y_test = data["y_test"]
    y_pred_rf_test = data["y_pred_test"]

    lr_meta = joblib.load(LR_MODEL_PATH)
    y_pred_stack_test = lr_meta.predict(y_pred_rf_test.reshape(-1, 1))

    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Encontrar inicio del test
    if version == "v3":
        total = len(df)
        val_end = int(total * 0.85)
        test_start_idx = val_end
    else:
        test_start_idx = df.index.get_loc("2025-01-02")

    # Reconversion
    close_real, close_pred_rf = reconvertir_a_close(y_pred_rf_test, df, test_start_idx)
    _, close_pred_stack = reconvertir_a_close(y_pred_stack_test, df, test_start_idx)
    dates = df.index[test_start_idx:test_start_idx + len(y_test)]

    os.makedirs(FIG_DIR, exist_ok=True)

    from sklearn.metrics import r2_score, mean_squared_error

    # ========== FIGURA 1: log_return (scatter) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pred, color, nombre in [
        (axes[0], y_pred_rf_test, 'blue', 'RF solo'),
        (axes[1], y_pred_stack_test, 'green', 'Stacking RF+LR')
    ]:
        ax.scatter(y_test, pred, alpha=0.5, s=20, c=color, label=nombre)
        r2_val = r2_score(y_test, pred)
        min_val = min(y_test.min(), pred.min())
        max_val = max(y_test.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Prediccion perfecta')
        ax.set_xlabel('log_return Real')
        ax.set_ylabel('log_return Predicho')
        ax.set_title(f'{nombre} (R2={r2_val:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()
    path_log = os.path.join(FIG_DIR, f"predictions_log_return_{version}.png")
    plt.savefig(path_log, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {path_log}")

    # ========== FIGURA 2: Close (serie temporal) ==========
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, close_real, 'b-', linewidth=1.5, label='Close Real', alpha=0.8)
    ax.plot(dates, close_pred_rf, 'orange', linewidth=1, label=f'RF {version} Predicho', alpha=0.7, linestyle='--')
    ax.plot(dates, close_pred_stack, 'green', linewidth=1, label=f'Stacking {version} RF+LR Predicho', alpha=0.7, linestyle='--')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio Close ($)')
    ax.set_title(f'Prediccion de Precio Close - SBUX {version} (Test)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path_close = os.path.join(FIG_DIR, f"predictions_close_{version}.png")
    plt.savefig(path_close, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {path_close}")

    # ========== FIGURA 3: Scatter de Close Predicho vs Real ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pred, color, nombre, close_pred in [
        (axes[0], y_pred_rf_test, 'blue', 'RF solo', close_pred_rf),
        (axes[1], y_pred_stack_test, 'green', 'Stacking RF+LR', close_pred_stack)
    ]:
        ax.scatter(close_real, close_pred, alpha=0.5, s=20, c=color, label=nombre)
        r2_val = r2_score(close_real, close_pred)
        min_val = min(close_real.min(), close_pred.min())
        max_val = max(close_real.max(), close_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Prediccion perfecta')
        ax.set_xlabel('Close Real ($)')
        ax.set_ylabel('Close Predicho ($)')
        ax.set_title(f'{nombre} - Close (R2={r2_val:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()
    path_close_scatter = os.path.join(FIG_DIR, f"predictions_close_scatter_{version}.png")
    plt.savefig(path_close_scatter, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {path_close_scatter}")

    # ========== FIGURA 4: Error absoluto por dia (serie temporal) ==========
    fig, ax = plt.subplots(figsize=(14, 5))

    error_rf = np.abs(y_test - y_pred_rf_test)
    error_stack = np.abs(y_test - y_pred_stack_test)
    error_acum_rf = np.cumsum(error_rf)
    error_acum_stack = np.cumsum(error_stack)

    ax.plot(dates, error_rf, 'o-', color='steelblue', linewidth=0.8, markersize=3, label=f'RF - Error diario (media={np.mean(error_rf):.4f})', alpha=0.7)
    ax.plot(dates, error_stack, 's-', color='seagreen', linewidth=0.8, markersize=3, label=f'Stacking - Error diario (media={np.mean(error_stack):.4f})', alpha=0.7)
    ax.axhline(y=np.mean(error_rf), color='steelblue', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=np.mean(error_stack), color='seagreen', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Error absoluto en log_return')
    ax.set_title(f'Error Absoluto por Dia - {version}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path_error = os.path.join(FIG_DIR, f"predictions_error_diario_{version}.png")
    plt.savefig(path_error, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {path_error}")

    # ========== FIGURA 5: Error acumulado en el tiempo ==========
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(dates, 0, error_acum_rf, alpha=0.3, color='steelblue', label=f'RF - Error acumulado')
    ax.fill_between(dates, 0, error_acum_stack, alpha=0.3, color='seagreen', label=f'Stacking - Error acumulado')
    ax.plot(dates, error_acum_rf, '-', color='steelblue', linewidth=1.5, label=f'RF (total={error_acum_rf[-1]:.4f})')
    ax.plot(dates, error_acum_stack, '-', color='seagreen', linewidth=1.5, label=f'Stacking (total={error_acum_stack[-1]:.4f})')

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Error absoluto acumulado')
    ax.set_title(f'Error Acumulado en el Tiempo - {version}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path_acum = os.path.join(FIG_DIR, f"predictions_error_acumulado_{version}.png")
    plt.savefig(path_acum, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {path_acum}")

    # ========== FIGURA 6: Histograma de errores ==========
    fig, ax = plt.subplots(figsize=(10, 5))

    residuos_rf = y_test - y_pred_rf_test
    residuos_stack = y_test - y_pred_stack_test

    ax.hist(residuos_rf, bins=12, alpha=0.5, color='steelblue', label=f'RF (media={np.mean(residuos_rf):.4f}, std={np.std(residuos_rf):.4f})', edgecolor='white')
    ax.hist(residuos_stack, bins=12, alpha=0.5, color='seagreen', label=f'Stacking (media={np.mean(residuos_stack):.4f}, std={np.std(residuos_stack):.4f})', edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Error cero')

    ax.set_xlabel('Residuo (real - predicho)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribucion de Errores (Residuos) - {version}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path_hist = os.path.join(FIG_DIR, f"predictions_error_histograma_{version}.png")
    plt.savefig(path_hist, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {path_hist}")

    print(f"Figuras {version} listas en: {FIG_DIR}/")
