"""
Entrenamiento de Random Forest como modelo base para el stacking (v2).
Optimiza hiperparametros con GridSearchCV sobre validation set.
Guarda en output/models/rf_model_v2.joblib y db/processed/rf_predictions_v2.npz.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Configuracion
DATA_NPZ = "db/processed/scaled_data.npz"
MODEL_PATH = "output/models/rf_model_v2.joblib"
PRED_PATH = "db/processed/rf_predictions_v2.npz"
RANDOM_STATE = 42


def calcular_metricas(y_true, y_pred, nombre):
    """Calcula MSE, RMSE, MAE, R2, RSE."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rse = np.sqrt(np.sum((y_true - y_pred)**2) / (len(y_true) - 2)) if len(y_true) > 2 else np.nan
    return {"model": nombre, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "rse": rse}


def imprimir_metricas(metrics_train, metrics_val, metrics_test):
    """Imprime tabla comparativa de metricas."""
    print(f"{'Split':<10} {'MSE':<14} {'RMSE':<14} {'MAE':<14} {'R2':<14} {'RSE':<14}")
    print("-" * 66)
    for name, m in [("Train", metrics_train), ("Val", metrics_val), ("Test", metrics_test)]:
        print(f"{name:<10} {m['mse']:<14.8f} {m['rmse']:<14.8f} {m['mae']:<14.8f} {m['r2']:<14.6f} {m['rse']:<14.8f}")


def imprimir_importancia(rf, features):
    """Imprime importancia de features ordenada."""
    print(f"\nImportancia de features:")
    print("-" * 50)
    importancias = sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(importancias, 1):
        barra = "█" * int(imp * 100)
        print(f"  {i:2d}. {feat:<20} {imp:.4f}  {barra}")


if __name__ == "__main__":
    # Cargar datos escalados
    print("Cargando datos escalados...")
    data = np.load(DATA_NPZ, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    features = data["feature_names"].tolist()

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test: {y_test.shape}")

    # ================================================================
    # OPTIMIZACION DE HIPERPARAMETROS CON GRIDSEARCHCV
    # ================================================================
    # Grid sobre validation set (no CV cruzada para evitar data leakage temporal)
    print("\n" + "=" * 60)
    print("OPTIMIZACION DE HIPERPARAMETROS (GridSearch sobre validation set)")
    print("=" * 60)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [1, 2, 5],
    }

    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, bootstrap=True)

    # Usamos X_val como evaluador (equivalente a scoring en val)
    best_score = -np.inf
    best_params = None
    best_rf = None

    resultados_grid = []

    for n_est in param_grid["n_estimators"]:
        for max_d in param_grid["max_depth"]:
            for min_leaf in param_grid["min_samples_leaf"]:
                rf_temp = RandomForestRegressor(
                    n_estimators=n_est, max_depth=max_d,
                    min_samples_leaf=min_leaf,
                    random_state=RANDOM_STATE, n_jobs=-1
                )
                rf_temp.fit(X_train, y_train)
                y_pred_val_temp = rf_temp.predict(X_val)
                score_val = r2_score(y_val, y_pred_val_temp)

                resultados_grid.append({
                    "n_estimators": n_est, "max_depth": max_d,
                    "min_samples_leaf": min_leaf, "r2_val": score_val
                })

                if score_val > best_score:
                    best_score = score_val
                    best_params = {"n_estimators": n_est, "max_depth": max_d, "min_samples_leaf": min_leaf}
                    best_rf = rf_temp

                print(f"  n_est={n_est:3d}  max_depth={str(max_d):>4}  min_leaf={min_leaf}  →  R2_val={score_val:.6f}")

    print(f"\n✅ Mejores hiperparametros: {best_params}")
    print(f"✅ Mejor R2 en validation: {best_score:.6f}")

    # Usar el mejor modelo encontrado
    rf = best_rf

    # Predecir
    y_pred_train = rf.predict(X_train)
    y_pred_val = rf.predict(X_val)
    y_pred_test = rf.predict(X_test)

    # Metricas
    metrics_train = calcular_metricas(y_train, y_pred_train, "RF_v2_train")
    metrics_val = calcular_metricas(y_val, y_pred_val, "RF_v2_val")
    metrics_test = calcular_metricas(y_test, y_pred_test, "RF_v2_test")

    print(f"\nMetricas Random Forest v2 (optimizado):")
    imprimir_metricas(metrics_train, metrics_val, metrics_test)

    # Comparacion v1 vs v2 (v1 = config original 200/15)
    r2_test_v2 = metrics_test["r2"]
    rmse_test_v2 = metrics_test["rmse"]
    print(f"\nComparacion con v1:")
    print(f"  v1 (RF 200/15): R2_test={0.007125:.4f}, RMSE_test={0.021146:.4f}")
    print(f"  v2 (RF optimo): R2_test={r2_test_v2:.4f}, RMSE_test={rmse_test_v2:.4f}")

    # Overfitting
    diff_r2 = metrics_train["r2"] - metrics_test["r2"]
    print(f"\nR2 Train - R2 Test = {diff_r2:.4f}")
    if diff_r2 > 0.3:
        print("⚠️  Overfitting significativo (esperado en datos financieros)")

    # Importancia
    imprimir_importancia(rf, features)

    # Guardar modelo v2
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"\nModelo v2 guardado: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024:.1f} KB)")

    # Guardar predicciones para stacking v2
    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    np.savez(PRED_PATH,
             y_pred_train=y_pred_train, y_pred_val=y_pred_val, y_pred_test=y_pred_test,
             y_train=y_train, y_val=y_val, y_test=y_test)
    print(f"Predicciones v2 guardadas: {PRED_PATH}")
