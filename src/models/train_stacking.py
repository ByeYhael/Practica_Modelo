"""
Stacking: Regresion Lineal como meta-modelo sobre predicciones de Random Forest.
RF es el modelo base, LR calibra el sesgo residual.
Input:  db/processed/rf_predictions.npz
Output: output/models/lr_meta_model.joblib, output/tables/models/evaluation_results.csv
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Configuracion
RF_PRED_PATH = "db/processed/rf_predictions.npz"
LR_MODEL_PATH = "output/models/lr_meta_model.joblib"
EVAL_PATH = "output/tables/models/evaluation_results.csv"
RANDOM_STATE = 42


def calcular_metricas(y_true, y_pred, nombre):
    """Calcula MSE, RMSE, MAE, R2, RSE."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rse = np.sqrt(np.sum((y_true - y_pred)**2) / (len(y_true) - 2)) if len(y_true) > 2 else np.nan
    return {"model": nombre, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "rse": rse}


def imprimir_tabla(metricas_list, titulo):
    """Imprime tabla comparativa de metricas."""
    print(f"\n{titulo}")
    print("-" * 66)
    print(f"{'Modelo':<20} {'MSE':<14} {'RMSE':<14} {'MAE':<14} {'R2':<14}")
    print("-" * 66)
    for m in metricas_list:
        print(f"{m['model']:<20} {m['mse']:<14.8f} {m['rmse']:<14.8f} {m['mae']:<14.8f} {m['r2']:<14.6f}")
    print("-" * 66)


if __name__ == "__main__":
    # Cargar predicciones de RF
    print("Cargando predicciones de Random Forest...")
    data = np.load(RF_PRED_PATH)

    y_pred_rf_train = data["y_pred_train"]
    y_pred_rf_val = data["y_pred_val"]
    y_pred_rf_test = data["y_pred_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    print(f"Pred RF Train: {y_pred_rf_train.shape}")
    print(f"Pred RF Val:   {y_pred_rf_val.shape}")
    print(f"Pred RF Test:  {y_pred_rf_test.shape}")

    # --- Paso 1: Entrenar LR meta-modelo ---
    # Feature: SOLO la prediccion de RF (reshape a 2D para sklearn)
    print("\nEntrenando Regresion Lineal como meta-modelo...")
    lr_meta = LinearRegression(fit_intercept=True)
    lr_meta.fit(y_pred_rf_train.reshape(-1, 1), y_train)

    print(f"  Coeficiente (peso de RF pred): {lr_meta.coef_[0]:.6f}")
    print(f"  Intercept (sesgo calibrado):    {lr_meta.intercept_:.8f}")

    # --- Paso 2: Predecir con stacking ---
    y_pred_stack_train = lr_meta.predict(y_pred_rf_train.reshape(-1, 1))
    y_pred_stack_val = lr_meta.predict(y_pred_rf_val.reshape(-1, 1))
    y_pred_stack_test = lr_meta.predict(y_pred_rf_test.reshape(-1, 1))

    # --- Paso 3: Metricas comparativas ---
    # RF SOLO (Test)
    metrics_rf_test = calcular_metricas(y_test, y_pred_rf_test, "Random Forest (Test)")

    # Stacking RF+LR (Test)
    metrics_stack_test = calcular_metricas(y_test, y_pred_stack_test, "Stacking RF+LR (Test)")

    # Metricas completas por split
    metrics_rf_train = calcular_metricas(y_train, y_pred_rf_train, "RF Train")
    metrics_rf_val = calcular_metricas(y_val, y_pred_rf_val, "RF Val")
    metrics_rf_test_full = calcular_metricas(y_test, y_pred_rf_test, "RF Test")

    metrics_stack_train = calcular_metricas(y_train, y_pred_stack_train, "Stacking Train")
    metrics_stack_val = calcular_metricas(y_val, y_pred_stack_val, "Stacking Val")
    metrics_stack_test_full = calcular_metricas(y_test, y_pred_stack_test, "Stacking Test")

    # Tabla comparativa en test
    imprimir_tabla([metrics_rf_test, metrics_stack_test], "Comparacion en TEST")

    # Tabla completa (train/val/test)
    print("\nMetricas detalladas por split:")
    print("-" * 70)
    for m in [metrics_rf_train, metrics_rf_val, metrics_rf_test_full,
              metrics_stack_train, metrics_stack_val, metrics_stack_test_full]:
        print(f"  {m['model']:<20} MSE={m['mse']:.8f}  RMSE={m['rmse']:.6f}  R2={m['r2']:.6f}")

    # --- Paso 4: Determinar el mejor modelo ---
    print(f"\nAnalisis del mejor modelo:")
    print(f"  RF Test:      R2={metrics_rf_test['r2']:.6f}, RMSE={metrics_rf_test['rmse']:.6f}")
    print(f"  Stacking RF+LR: R2={metrics_stack_test['r2']:.6f}, RMSE={metrics_stack_test['rmse']:.6f}")

    if metrics_stack_test['r2'] > metrics_rf_test['r2']:
        best = "Stacking RF+LR"
        best_r2 = metrics_stack_test['r2']
        best_rmse = metrics_stack_test['rmse']
        print(f"\n  ✅ Mejor modelo: {best} (R2={best_r2:.6f})")
    else:
        best = "Random Forest"
        best_r2 = metrics_rf_test['r2']
        best_rmse = metrics_rf_test['rmse']
        print(f"\n  ✅ Mejor modelo: {best} (R2={best_r2:.6f})")

    # --- Paso 5: Guardar resultados ---
    os.makedirs(os.path.dirname(LR_MODEL_PATH), exist_ok=True)
    joblib.dump(lr_meta, LR_MODEL_PATH)
    print(f"\nMeta-modelo guardado: {LR_MODEL_PATH}")

    # Guardar tabla de evaluacion
    import pandas as pd
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    eval_data = [
        {"model": "random_forest",
         "mse": metrics_rf_test_full["mse"], "rmse": metrics_rf_test_full["rmse"],
         "mae": metrics_rf_test_full["mae"], "r2": metrics_rf_test_full["r2"],
         "rse": metrics_rf_test_full["rse"]},
        {"model": "stacking_rf_lr",
         "mse": metrics_stack_test_full["mse"], "rmse": metrics_stack_test_full["rmse"],
         "mae": metrics_stack_test_full["mae"], "r2": metrics_stack_test_full["r2"],
         "rse": metrics_stack_test_full["rse"]},
    ]
    df_eval = pd.DataFrame(eval_data)
    df_eval.to_csv(EVAL_PATH, index=False)
    print(f"Tabla de evaluacion guardada: {EVAL_PATH}")
    print(f"\n{df_eval.to_string()}")
