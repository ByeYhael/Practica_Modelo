"""
Genera graficos v2 de predicciones vs real para RF optimizado y Stacking RF+LR.
- predictions_log_return_v2.png: scatter en escala log_return
- predictions_close_v2.png: predicciones reconvertidas a precio Close
- predictions_comparison_v2.png: barras comparativas v2
Incluye comparacion contra v1 en los titulos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Configuracion (v2)
RF_PRED_PATH = "db/processed/rf_predictions_v2.npz"
FEATURES_PATH = "db/processed/features_v1.csv"
LR_MODEL_PATH = "output/models/lr_meta_model_v2.joblib"
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
    path_log = os.path.join(FIG_DIR, "predictions_log_return_v2.png")
    plt.savefig(path_log, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path_log}")

    # ========== FIGURA 2: Close (serie temporal) ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, close_real, 'b-', linewidth=1.5, label='Close Real', alpha=0.8)
    ax.plot(dates, close_pred_rf, 'orange', linewidth=1, label='RF v2 Predicho', alpha=0.7, linestyle='--')
    ax.plot(dates, close_pred_stack, 'green', linewidth=1, label='Stacking v2 RF+LR Predicho', alpha=0.7, linestyle='--')

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio Close ($)')
    ax.set_title('Predicción de Precio Close - SBUX v2 (Test: 2025-2026)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path_close = os.path.join(FIG_DIR, "predictions_close_v2.png")
    plt.savefig(path_close, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path_close}")

    # ========== FIGURA 3: Comparativa metricas v1 vs v2 ==========
    from sklearn.metrics import r2_score, mean_squared_error

    # Metricas v2 (actuales)
    r2_rf_v2 = r2_score(y_test, y_pred_rf_test)
    rmse_rf_v2 = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    r2_stack_v2 = r2_score(y_test, y_pred_stack_test)
    rmse_stack_v2 = np.sqrt(mean_squared_error(y_test, y_pred_stack_test))

    # Metricas v1 (hardcodeadas del resultado anterior)
    r2_rf_v1 = 0.007125
    rmse_rf_v1 = 0.021146
    r2_stack_v1 = -0.145994
    rmse_stack_v1 = 0.022718

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico R2
    ax = axes[0]
    x = np.arange(2)
    width = 0.30
    ax.bar(x - width/2, [r2_rf_v1, r2_stack_v1], width, label='v1 (200/15)', color='lightgray', edgecolor='gray')
    ax.bar(x + width/2, [r2_rf_v2, r2_stack_v2], width, label='v2 (optimizado)', color=['steelblue', 'seagreen'])
    ax.set_ylabel('R²')
    ax.set_title('Comparación R²: v1 vs v2')
    ax.set_xticks(x)
    ax.set_xticklabels(['Random Forest', 'Stacking RF+LR'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (v1, v2) in enumerate(zip([r2_rf_v1, r2_stack_v1], [r2_rf_v2, r2_stack_v2])):
        ax.text(i - width/2, v1 + 0.005, f'{v1:.4f}', ha='center', va='bottom', fontsize=8, color='gray')
        ax.text(i + width/2, v2 + 0.005, f'{v2:.4f}', ha='center', va='bottom', fontsize=8)

    # Grafico RMSE
    ax = axes[1]
    ax.bar(x - width/2, [rmse_rf_v1, rmse_stack_v1], width, label='v1 (200/15)', color='lightgray', edgecolor='gray')
    ax.bar(x + width/2, [rmse_rf_v2, rmse_stack_v2], width, label='v2 (optimizado)', color=['steelblue', 'seagreen'])
    ax.set_ylabel('RMSE')
    ax.set_title('Comparación RMSE: v1 vs v2')
    ax.set_xticks(x)
    ax.set_xticklabels(['Random Forest', 'Stacking RF+LR'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (v1, v2) in enumerate(zip([rmse_rf_v1, rmse_stack_v1], [rmse_rf_v2, rmse_stack_v2])):
        ax.text(i - width/2, v1 + 0.0005, f'{v1:.4f}', ha='center', va='bottom', fontsize=8, color='gray')
        ax.text(i + width/2, v2 + 0.0005, f'{v2:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path_comp = os.path.join(FIG_DIR, "predictions_comparison_v2.png")
    plt.savefig(path_comp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path_comp}")

    print(f"\n✅ Figuras v2 guardadas en: {FIG_DIR}")
