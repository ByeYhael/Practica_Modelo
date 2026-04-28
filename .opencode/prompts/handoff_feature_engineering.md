# Handoff: EDA -> Feature Engineering + Modelado

## Resumen de lo completado (Seccion 1: EDA)
- Ventana de datos definida: 2015-2026 (2786 registros)
- Target definido: log_return (estacionario, estandar financiero)
- Outliers de Volume: conservar con transformacion log
- Features candidatas identificadas: volatilidad 20d/5d, rango diario (High-Low), log(Volume), dummies temporales
- Justificaciones documentadas en output/reports/eda_summary.md

## Proximo agente: PreprocessingModelAgent

### Rol
Agente especializado en construccion de features, entrenamiento de modelos y evaluacion comparativa.

### Pipeline interno del agente

```
PreprocessingModelAgent
  |
  |-- Step 1: Cargar datos limpios de db/raw/sbux.csv (ventana 2015-2026)
  |
  |-- Step 2: Feature Engineering
  |     - Target: log_return = diff(log(Close))
  |     - Features de volatilidad: rolling std 20d, rolling std 5d
  |     - Feature de rango: High - Low
  |     - Feature de volumen: log(Volume)
  |     - Dummies temporales: day_of_week, month
  |     - Lags de retornos: lag-1, lag-5, lag-20 (baja expectativa predictiva)
  |     - Drop de filas con NaN generados por lags/rolling
  |     - Guardar dataset en db/processed/features_v1.csv
  |
  |-- Step 3: Split temporal (NO aleatorio)
  |     - Train: <= 2023 (aprox 2260 registros)
  |     - Validation: 2024 (aprox 252 registros)
  |     - Test: 2025-2026 (aprox 270 registros)
  |
  |-- Step 4: Escalado de features
  |     - StandardScaler sobre features (NO sobre target)
  |     - Guardar scaler en output/models/scaler.joblib
  |
  |-- Step 5: Entrenamiento de modelos
  |     Modelos a probar (orden sugerido):
  |     1. Regresion Lineal (baseline)
  |     2. Regresion Polinomial (grado 2-3)
  |     3. Arbol de Decision (max_depth=5,10,15)
  |     4. Random Forest (n_estimators=100,200)
  |     5. SVR (kernel=rbf, poly)
  |     6. Prophet (baseline de series temporales)
  |
  |-- Step 6: Evaluacion
  |     - Metricas: MSE, RMSE, MAE, R2, RSE
  |     - Prediccion en test set (log_return)
  |     - Reconversion a Close: Close_pred = Close_actual * exp(log_return_pred)
  |     - Metricas adicionales sobre Close reconstruido
  |     - Plot: predicciones vs real (log_return y Close)
  |
  |-- Step 7: Seleccion del mejor modelo
  |     - Criterio: mayor R2 y menor RMSE
  |     - Tabla comparativa de modelos
  |     - Identificar overfitting (train vs test metrics)
  |
  |-- Output: output/models/*.joblib, output/tables/model_comparison.csv,
  |           output/figures/predictions_*.png
```

### Input contract para el agente
| Parametro | Valor |
|---|---|
| raw_data_path | db/raw/sbux.csv |
| processed_data_path | db/processed/features_v1.csv |
| scaler_path | output/models/scaler.joblib |
| fig_dir | output/figures/ |
| table_dir | output/tables/ |
| model_dir | output/models/ |
| start_date | 2015-01-01 |
| target_col | Close (se transforma a log_return internamente) |
| random_state | 42 |
| test_size_frac | 0.2 |

### Output contract esperado
| Archivo | Descripcion |
|---|---|
| db/processed/features_v1.csv | Dataset con features construidas |
| output/models/scaler.joblib | StandardScaler ajustado |
| output/models/linear_regression.joblib | Modelo 1 entrenado |
| output/models/random_forest.joblib | Modelo 4 entrenado |
| output/models/svr_model.joblib | Modelo 5 entrenado |
| output/models/prophet_model.joblib | Modelo 6 entrenado |
| output/tables/model_comparison.csv | Tabla comparativa de metricas |
| output/figures/predictions_log_return.png | Predicciones en escala log |
| output/figures/predictions_close.png | Predicciones reconvertidas a precio |

### Validation gates
- [ ] features_v1.csv existe y tiene > 0 filas
- [ ] No hay NaN en el dataset de features
- [ ] scaler.joblib tiene los atributos mean_ y scale_
- [ ] model_comparison.csv tiene al menos 3 modelos
- [ ] Las figuras de prediccion se generaron sin errores
