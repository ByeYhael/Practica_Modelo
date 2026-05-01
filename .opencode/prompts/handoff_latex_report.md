# Handoff: Architect -> LaTeXWriter_Agent

## Resumen del pipeline completado

El pipeline completo de prediccion de precios SBUX concluyo con 3 versiones exploradas. La version ganadora es **RF v3**: Random Forest entrenado con datos del ultimo ano (2025-02-03 a 2025-10-13), optimizado via GridSearch (n_estimators=100, max_depth=10, min_samples_leaf=1). Los resultados completos estan documentados en `output/reports/modeling_summary.md`.

---

## Input Contract (rutas a los archivos)

```
project_name: "Prediccion de Precios de Starbucks (SBUX)"
author: "Yhael Salvador Perez Balderas"
institution: "Universidad Autonoma de Queretaro"
professor: "Dr. Mariano Garduno Aparicio"
course: "Aprendizaje Automatico"
date: "Mayo 2026"

dataset:
  source: "Kaggle - SBUX Stock Price 2000-2026"
  records: 6559
  window_used: "Febrero 2025 - Enero 2026 (250 registros)"
  features: "[Date, Open, High, Low, Close, Volume]"
  target: "log_return (estacionario, derivado de Close)"

tables:
  eda: "output/tables/estadisticos_descriptivos.csv"
  stationarity: "output/tables/stationarity_tests.csv"
  evaluation_v3: "output/tables/models/evaluation_results_v3.csv"

figures:
  eda_close_timeseries: "output/figures/eda/eda_close_timeseries.png"
  eda_correlation: "output/figures/eda/eda_correlation_heatmap.png"
  eda_volatility: "output/figures/eda/eda_rolling_volatility.png"
  eda_volume: "output/figures/eda/eda_volume_transform.png"
  eda_acf: "output/figures/eda/eda_acf_pacf.png"
  
  model_log_return: "output/figures/models/predictions_log_return_v3.png"
  model_close_timeseries: "output/figures/models/predictions_close_v3.png"
  model_close_scatter: "output/figures/models/predictions_close_scatter_v3.png"
  model_error_diario: "output/figures/models/predictions_error_diario_v3.png"
  model_error_acumulado: "output/figures/models/predictions_error_acumulado_v3.png"
  model_error_histograma: "output/figures/models/predictions_error_histograma_v3.png"

reports:
  modeling_summary: "output/reports/modeling_summary.md"
  eda_summary: "output/reports/eda_summary_v2.md"

models:
  best: "output/models/rf_model_v3.joblib"
  parameters: "n_estimators=100, max_depth=10, min_samples_leaf=1"
  size: "780 KB"
```

---

## Prompt para LaTeXWriter_Agent

```
ROLE: LaTeXWriter_Agent
CONTEXT: Fase final del pipeline SBUX. Debes generar el reporte academico en LaTeX a partir de los outputs del modelado.

TASK: Crear un archivo report.tex compilable con pdflatex, breve, limpio y academico.

IMPORTANTE: 
- Debes preguntar al usuario ANTES de escribir cada seccion del reporte. Espera su confirmacion para continuar.
- El reporte debe ser BREVE pero completo. Sin simbolos raros. Bien explicado en lenguaje claro.
- Usa los archivos de input contract para llenar tablas y figuras.

ESTRUCTURA DEL REPORTE:

### 1. PORTADA
\title{Prediccion del Precio de Acciones de Starbucks (SBUX) mediante Random Forest}
\author{Yhael Salvador Perez Balderas}
\date{Mayo 2026}
Incluir logotipo de la universidad si existe, o espacio para el.

### 2. RESUMEN (abstract)
- 1 parrafo breve: que se hizo (predecir retorno SBUX con Random Forest), con que datos (ultimo ano, 250 registros), cual fue el mejor resultado (R2=0.028, RMSE=0.017), y conclusion principal.

### 3. INTRODUCCION
- Contexto: prediccion de precios de acciones, hipotesis de mercado eficiente, desafio de predecir retornos.
- Objetivo: implementar un modelo de regresion (Stacking RF + LR) para predecir log_return de SBUX.
- Estructura del documento.

### 4. MATERIALES Y METODOS
Separar en subsecciones:

#### 4.1 Datos
- Fuente: Kaggle, SBUX 2000-2026 (6559 obs diarias).
- Ventana usada: solo ultimo ano (feb 2025 - ene 2026, 250 registros) por cambio de regimen de mercado.
- Columnas: Open, High, Low, Close, Volume.
- Target: log_return = diff(log(Close)). Justificar: estacionariedad (ADF p~0, KPSS p>0.10).

#### 4.2 Preprocesamiento
- Feature engineering: volatilidad 20d y 5d (rolling std), rango diario (High-Low), log(Volume), dummies temporales (day_of_week, month), lags de retornos (1, 5, 20).
- Dropeo de NaN: 21 filas (0.75%) por ventanas rolling.
- Escalado: StandardScaler con fit en TRAIN, transform en VAL y TEST (cero data leakage). Target NO escalado.
- Split temporal (NO aleatorio): Train 70% (feb-oct 2025), Val 15% (oct-dic 2025), Test 15% (dic 2025-ene 2026).

#### 4.3 Modelos
- Random Forest: modelo base, 100 arboles, profundidad 10, min_samples_leaf=1.
- Regresion Lineal: meta-modelo que toma prediccion de RF como unica entrada (Stacking).
- Hiperparametros optimizados via GridSearch manual sobre validation set.

#### 4.4 Metricas de Evaluacion
Lista breve de metricas usadas:
- MSE, RMSE, MAE, RSE: errores
- R2 y R2 Ajustado: varianza explicada
- CC (Correlacion de Pearson): direccion de prediccion
- Max Error y Bias: error maximo y sesgo

### 5. RESULTADOS Y DISCUSION
Esta es la seccion mas importante. Incluir:

#### 5.1 Resultados del EDA
- Breve resumen: Close no estacionario, log_return si lo es.
- Correlacion: OHLC tienen correlacion >0.99, por eso no se usaron como predictores directos.
- Volume con cola larga (skew 9.06): se aplico log-transform.
- Sin autocorrelacion significativa en retornos (max |0.07|).

#### 5.2 Resultados del Modelo
- Tabla comparativa de las 3 versiones exploradas (v1, v2, v3).
- La tabla debe incluir: Version, Modelo, R2, RMSE, MSE, MAE.
- Texto explicando: v3 fue la mejor porque uso datos del ultimo ano.
- Tabla de metricas finales del modelo v3.

#### 5.3 Importancia de Features
- Lista ordenada: rango_diario (38.4%), volatilidad_5d (14.4%), log_volume (12.9%), etc.

#### 5.4 Figuras
Incluir las siguientes figuras (preguntar al usuario cual incluir):

a) Scatter log_return: prediccion vs real con linea diagonal (y=x). Muestra la dispersion del modelo.
b) Serie temporal Close: evolucion del precio real vs predicho.
c) Error diario: error absoluto dia por dia en el periodo de test.
d) Error acumulado: como se acumula el error en el tiempo.
e) Histograma de errores: distribucion de los residuos.

CADA FIGURA debe tener un pie explicativo breve.

### 6. CONCLUSION
- Resumen de 2-3 parrafos: 
  * El modelo RF v3 logro R2=0.028 en test, superando al modelo naive (R2=0).
  * Las limitaciones: baja senal en datos financieros, R2 bajo pero esperado en retornos diarios.
  * Trabajo futuro: incluir datos macroeconomicos, sentimiento de noticias, reducir features ruidosas.

### 7. BIBLIOGRAFIA
Incluir referencias:
- Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 2011.
- McKinney, pandas: a foundational Python library for data analysis, 2010.
- Fama, Efficient Capital Markets: A Review of Theory and Empirical Work, JF 1970.
- La bibliografia debe ir en un archivo references.bib separado.

### 8. ANEXOS (opcional, preguntar al usuario si desea incluirlos)
Contenido sugerido para anexos:
- Tabla completa de metricas de las 3 versiones (v1, v2, v3) con todas las metricas (R2, RMSE, MSE, MAE, CC, R2_adj, MaxErr, Bias).
- Descripcion breve de las metricas excluidas (RAE, MAPE, NRMSE) y por que no se usaron.
- Scripts clave (feature_engineering.py, train_rf.py) como codigo en lstlisting.
- Diagrama de flujo del pipeline (si existe).

ESTILO:
- Espanol academico formal pero claro.
- Sin simbolos raros ni emojis.
- Sin codigo basura: solo lo que aporta al reporte.
- Figuras con resolucion adecuada (width=0.8\textwidth o similar).
- Tablas limpias con booktabs.
- Numeros con 4 decimales consistentes.

PASOS (preguntar antes de cada uno):
1. "Deseas que escriba la portada y el resumen?"
2. "Deseas que escriba la introduccion?"
3. "Deseas que escriba la seccion de materiales y metodos?"
4. "Deseas que escriba la seccion de resultados y discusion?"
5. "Deseas que incluya la figura [nombre] en resultados?" (para cada figura)
6. "Deseas que escriba la conclusion?"
7. "Deseas incluir anexos? Que contenido deseas en ellos?"
8. "Deseas que genere el archivo .bib de bibliografia?"

OUTPUT CONTRACT:
- output/report.tex (compilable con pdflatex)
- output/references.bib
```
