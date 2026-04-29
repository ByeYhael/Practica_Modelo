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

    # ========== FIGURA 3: Comparativa metricas ==========
    r2_rf = r2_score(y_test, y_pred_rf_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    r2_stack = r2_score(y_test, y_pred_stack_test)
    rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack_test))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [r2_rf, r2_stack], width, label='R2', color=['steelblue', 'seagreen'])
    ax.bar(x + width/2, [rmse_rf, rmse_stack], width, label='RMSE', color=['lightblue', 'lightgreen'])
    ax.set_ylabel('Valor')
    ax.set_title(f'Metricas {version} - RF vs Stacking')
    ax.set_xticks(x)
    ax.set_xticklabels(['Random Forest', 'Stacking RF+LR'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (r2_v, rmse_v) in enumerate(zip([r2_rf, r2_stack], [rmse_rf, rmse_stack])):
        ax.text(i - width/2, r2_v + 0.001, f'{r2_v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, rmse_v + 0.001, f'{rmse_v:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path_comp = os.path.join(FIG_DIR, f"predictions_comparison_{version}.png")
    plt.savefig(path_comp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {path_comp}")

    print(f"Figuras {version} listas en: {FIG_DIR}/")
