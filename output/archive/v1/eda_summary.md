# EDA Summary - SBUX Stock Price Prediction

## Dataset
- **Fuente**: Kaggle - Starbucks (SBUX) 2000-2026
- **Ventana EDA**: 2015-01-02 a 2026-01-30 (2786 registros)
- **Columnas**: Open, High, Low, Close, Volume (sin nulos)

## Justificacion de Decisiones (Apuntes del Pipeline)

### 1. Ventana temporal: por que 2015-2026 y no 2000-2026
- SBUX cambio de regimen estructural: $5-20 (2000-2014, cadena emergente) vs $50-95 (2015-2026, empresa madura con dividendos).
- Incluir datos pre-2015 introduce ruido de regimen: correlaciones, distribuciones y volatilidad estarian dominadas por el cambio de nivel de precio, no por la dinamica actual del mercado.
- 2786 registros (2015-2026) son suficientes para EDA, feature engineering y modelos de series temporales.
- Los datos 2000-2014 se documentan en tabla comparativa historica pero no se usan en el pipeline principal.

### 2. Target: log_return en lugar de Close
- Close y log(Close) son NO estacionarios (ADF p>0.20, KPSS p<0.01). Modelos lineales asumen estacionaridad.
- log_return es estacionario (ADF p~0, KPSS p>0.10) y es el estandar financiero para prediccion de activos.
- Interpretacion directa: log_return ~ retorno porcentual. Prediccion de 0.01 = +1%.
- Close final se reconstruye en post-procesamiento: Close_pred = Close_actual * exp(log_return_pred).

### 3. Outliers de Volume: conservar con transformacion log
- Volume tiene distribucion extremadamente sesgada (skew 9.06, kurtosis 192) con 159 outliers IQR (5.71%).
- En series financieras, volumen alto es senal de eventos de mercado (earnings, splits, noticias), no ruido.
- Eliminar outliers pierde informacion de eventos relevantes. Transformacion log comprime la cola larga y permite escalas comparables entre features.
- Modelos basados en arboles (RF, DT) son inmunes a escala, pero log-transform no los perjudica.

### 4. Ventanas de rolling: 20d principal, 5d secundaria
- Ventana 20d captura regimen de volatilidad mensual (~22 trading days). Es el estandar en finanzas para prediccion semanal.
- Ventana 5d captura eventos intrasemanales como feature secundaria.
- Ventana 60d descartada: datos muy lejanos para horizonte semanal, anyade ruido en lugar de senal.

## Hallazgos Clave

### 1. Distribucion de Precios
- Close medio: $70.73, rango $31.57 - $112.84
- Distribucion casi simetrica (skew -0.02), platicurtica (kurtosis -1.42)
- Sin outliers en Close: los precios se mueven en un rango acotado y continuo

### 2. Volumen
- Altamente sesgado (skew 9.06, kurtosis 192)
- 159 outliers por IQR (5.71%) - esperado en series financieras

### 3. Tendencia y Volatilidad
- Rendimiento total 2015-2026: +181.5% (~9.8% anualizado)
- Volatilidad media 20d: 1.57%
- Pico maximo: 7.79% (abril 2020 - COVID)
- 124 dias (4.5%) superaron 2x la volatilidad media, concentrados en COVID y 2024-2025

### 4. Estacionaridad
| Serie | Estacionaria? | Pruebas |
|---|---|---|
| Close | NO | ADF p=0.33, KPSS p<0.01 |
| log(Close) | NO | ADF p=0.20, KPSS p<0.01 |
| diff(Close) | SI | ADF p~0, KPSS p>0.10 |
| log_return | SI | ADF p~0, KPSS p>0.10 |

### 5. Correlacion y Multicolinealidad
- Open, High, Low, Close tienen correlacion >0.998 entre si. Usar multiples OHLC como predictores causa multicolinealidad severa.
- **Decision**: No usar Open, High, Low como predictores directos. Usar solo Close + derivados (retornos, lags, volatilidad) y High-Low range como feature compuesta.
- Volume tiene correlacion negativa debil con Close (Pearson -0.17, Spearman -0.30).
- **Rango diario (High-Low)**: correlaciona con Close (0.46) y Volume (0.40). Es candidato fuerte a feature predictiva.

### 6. Autocorrelacion de retornos
- log_return NO presenta autocorrelacion significativa en ningun lag (maximo |0.07| en lag 1, dentro de la banda de ruido).
- Esto es consistente con la hipotesis de mercado eficiente para SBUX.
- **Implicacion**: los lags de retornos (lag-1, lag-5, lag-20) probablemente no seran predictores utiles.
- Las features de volatilidad, volumen y rango diario seran mas relevantes.

### 7. Cola pesada de distribucion
- 4.13% de retornos > 2 std (esperado ~5% en normal) y 1.47% > 3 std (esperado ~0.3%).
- Los retornos financieros tienen cola pesada: eventos extremos son mas frecuentes que en distribucion normal.
- **Implicacion**: modelos robustos (SVR con kernel rbf, Random Forest, arboles) pueden manejar mejor estos outliers que regresion lineal ordinaria.

## Recomendaciones para Feature Engineering

1. **Target = log_return** de Close (estacionario, interpretable como retorno %)
2. **Log-transform** de Volume para comprimir cola larga de outliers
3. **Rolling features**: volatilidad 20d (principal) y 5d (secundaria) como predictores
4. **Rango diario (High - Low)**: incluir como feature por su correlacion con Close y Volume
5. **Dummy temporal**: dia de semana y mes como variables categoricas para estacionalidad semanal/anual
6. **Lags de retornos**: incluirlos pero sin esperar alto poder predictivo (baja autocorrelacion)
