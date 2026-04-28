"""
Genera graficos de predicciones vs real para RF solo y Stacking RF+LR.
- predictions_log_return.png: scatter en escala log_return
- predictions_close.png: predicciones reconvertidas a precio Close
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Configuracion
RF_PRED_PATH = "db/processed/rf_predictions.npz"
FEATURES_PATH = "db/processed/features_v1.csv"
LR_MODEL_PATH = "output/models/lr_meta_model.joblib"
FIG_DIR = "output/figures/models"
RANDOM_STATE = 42


def reconvertir_a_close(y_log_return, df_original, start_idx):
    """
    Reconvierte log_return a precio Close.
    Close_pred = Close_actual * exp(log_return_pred)
    """
    # Obtener Close real del periodo
    close_real = df_original["Close"].values[start_idx:start_idx + len(y_log_return)]

    # Close_pred requiere el Close del dia anterior para el primer valor
    close_prev = df_original["Close"].values[start_idx - 1] if start_idx > 0 else close_real[0]
    close_pred = np.zeros_like(y_log_return)

    for i in range(len(y_log_return)):
        if i == 0:
            close_pred[i] = close_prev * np.exp(y_log_return[i])
        else:
            close_pred[i] = close_pred[i-1] * np.exp(y_log_return[i])

    return close_real, close_pred


if __name__ == "__main__":
    import joblib

    # Cargar datos
    data = np.load(RF_PRED_PATH)
    y_test = data["y_test"]
    y_pred_rf_test = data["y_pred_test"]

    # Cargar LR meta-modelo
    lr_meta = joblib.load(LR_MODEL_PATH)
    y_pred_stack_test = lr_meta.predict(y_pred_rf_test.reshape(-1, 1))

    # Cargar features para obtener Close real
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Encontrar indice de inicio del test
    test_start_idx = df.index.get_loc("2025-01-02")

    # Reconversion a Close
    close_real, close_pred_rf = reconvertir_a_close(y_pred_rf_test, df, test_start_idx)
    _, close_pred_stack = reconvertir_a_close(y_pred_stack_test, df, test_start_idx)

    # Fechas para el eje X
    dates = df.index[test_start_idx:test_start_idx + len(y_test)]

    # Crear directorio de figuras
    os.makedirs(FIG_DIR, exist_ok=True)

    # ========== FIGURA 1: log_return (scatter) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RF solo
    ax = axes[0]
    ax.scatter(y_test, y_pred_rf_test, alpha=0.5, s=20, c='blue', label='RF solo')
    min_val = min(y_test.min(), y_pred_rf_test.min())
    max_val = max(y_test.max(), y_pred_rf_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Predicción perfecta')
    ax.set_xlabel('log_return Real')
    ax.set_ylabel('log_return Predicho')
    ax.set_title('Random Forest - Predicción vs Real')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Stacking RF+LR
    ax = axes[1]
    ax.scatter(y_test, y_pred_stack_test, alpha=0.5, s=20, c='green', label='Stacking RF+LR')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Predicción perfecta')
    ax.set_xlabel('log_return Real')
    ax.set_ylabel('log_return Predicho')
    ax.set_title('Stacking RF+LR - Predicción vs Real')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    path_log = os.path.join(FIG_DIR, "predictions_log_return.png")
    plt.savefig(path_log, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path_log}")

    # ========== FIGURA 2: Close (serie temporal) ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, close_real, 'b-', linewidth=1.5, label='Close Real', alpha=0.8)
    ax.plot(dates, close_pred_rf, 'orange', linewidth=1, label='RF Predicho', alpha=0.7, linestyle='--')
    ax.plot(dates, close_pred_stack, 'green', linewidth=1, label='Stacking RF+LR Predicho', alpha=0.7, linestyle='--')

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio Close ($)')
    ax.set_title('Predicción de Precio Close - SBUX (Test: 2025-2026)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path_close = os.path.join(FIG_DIR, "predictions_close.png")
    plt.savefig(path_close, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path_close}")

    # ========== FIGURA 3: Comparativa metricas ==========
    from sklearn.metrics import r2_score, mean_squared_error

    fig, ax = plt.subplots(figsize=(8, 5))
    modelos = ['Random Forest', 'Stacking RF+LR']
    r2_vals = [r2_score(y_test, y_pred_rf_test), r2_score(y_test, y_pred_stack_test)]
    rmse_vals = [np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
                 np.sqrt(mean_squared_error(y_test, y_pred_stack_test))]

    x = np.arange(len(modelos))
    width = 0.35

    ax.bar(x - width/2, r2_vals, width, label='R²', color=['blue', 'green'])
    ax.bar(x + width/2, rmse_vals, width, label='RMSE', color=['lightblue', 'lightgreen'])

    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas - RF vs Stacking')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, (r2, rmse) in enumerate(zip(r2_vals, rmse_vals)):
        ax.text(i - width/2, r2 + 0.001, f'{r2:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, rmse + 0.001, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path_comp = os.path.join(FIG_DIR, "predictions_comparison.png")
    plt.savefig(path_comp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path_comp}")

    print(f"\n✅ Todas las figuras guardadas en: {FIG_DIR}")
