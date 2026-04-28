# SBUX Stock Price Prediction

Prediccion del precio de cierre (Close) de Starbucks (SBUX) usando datos historicos de Kaggle (2000-2026). Implementa regresion lineal, polinomial, arboles de decision, random forest, SVR, redes neuronales y Prophet como baseline.

## Estructura

```
.
├── config/            # Configuraciones (paths, parametros)
├── db/
│   ├── raw/           # Datos originales (SBUX.csv)
│   └── processed/     # Datos limpios y transformados
├── output/
│   ├── tables/        # Tablas de resultados (CSV, LaTeX)
│   ├── figures/       # Graficos (PNG, PDF)
│   ├── models/        # Modelos entrenados (joblib, h5)
│   └── reports/       # Documentacion generada
├── src/
│   ├── data/          # Carga, limpieza, transformacion
│   ├── analysis/      # EDA, visualizaciones
│   ├── models/        # Entrenamiento y evaluacion
│   ├── reports/       # Generacion de reportes LaTeX
│   └── utils/         # Funciones auxiliares
├── tests/             # Pruebas unitarias
├── requirements.txt   # Dependencias
└── README.md
```

## Instalacion

```bash
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

## Uso

```bash
python src/data/load_data.py
python src/analysis/eda.py
python src/models/train.py
python src/reports/generate_report.py
```

## Dataset

Fuente: Kaggle - Starbucks Stock Price (SBUX) 2000-2026.
Columnas: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits.
