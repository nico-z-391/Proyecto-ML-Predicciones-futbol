# Proyecto ML: Predicción de Partidos de Fútbol Internacional

Proyecto de Machine Learning para predecir resultados y marcadores de partidos de fútbol internacional, con foco en la Copa del Mundo 2026. Utiliza un sistema de predicción en cascada con tres modelos entrenados sobre ~25.000 partidos históricos (2000–2026).

## Tecnologías

- Python 3
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- LightGBM, XGBoost
- requests, joblib
- matplotlib, seaborn
- meteostat

## Arquitectura del sistema

El sistema de predicción usa tres modelos en cascada:

1. `rf_draw` — Random Forest que detecta si el partido termina en empate
2. `rf_winner` — Random Forest que predice ganador (si no es empate)
3. `lgbm` — LightGBM que predice el marcador (goles de cada equipo)

## Dataset

~25.000 partidos internacionales enriquecidos con:

- Ratings ELO de ambos equipos
- Forma reciente (últimos 5 y 10 partidos)
- Datos climáticos (temperatura, viento, lluvia, humedad, elevación)
- Features de similitud entre equipos (para detección de empates)
- Historial goles esperados (head-to-head)

## Cómo usarlo

### 1. Instalación

```bash
git clone https://github.com/nico-z-391/Proyecto-ML-Predicciones-futbol
cd Proyecto-ML-Predicciones-futbol
pip install pandas numpy scikit-learn lightgbm xgboost requests joblib meteostat matplotlib seaborn
jupyter notebook
```

### 2. Abrir el notebook

Abre `proyecto.ipynb` y ejecuta todas las celdas en orden.

`results.csv` — base original
`datos_prdi_temp.csv` — necesario para el sistema de predicción
`df2_datos_tratados_base2.csv` — necesario para los modelos
`dfa8_con_clima.csv` — checkpoint más actualizado
`eloratings.csv` — datos de ELO originales
`worldcities.csv` — datos de ciudades originales
`sedes_wc2026.json` — sedes del Mundial
`lista_torneos.txt y torneos.txt` — referencia de torneos

> Los modelos ya están entrenados y guardados como `.pkl`. Solo necesitas ejecutar las secciones **sedes_wc2026** - **categorizar_torneo** - **Sistema de predicción** para hacer predicciones.

### 3. Hacer una predicción

```python
predecir_partido(
    home_team='Argentina',
    away_team='Nigeria',
    tournament='FIFA World Cup',
    neutral=True,
    sede='Miami',
    fecha='2026-06-15'
)
```

El sistema obtiene automáticamente el clima de esa fecha y lugar, el ELO y forma reciente de ambos equipos, y devuelve:

==================================================
Argentina vs Nigeria
Miami | 2026-06-15 | FIFA World Cup
📊 Probabilidades:
🏠 Argentina gana: 61.3%
🤝 Empate:          22.1%
✈️  Nigeria gana:   16.6%
⚽ Marcador estimado:
Argentina 2 - 1 Nigeria

### 4. Sedes disponibles

El sistema soporta las 16 sedes del Mundial 2026:

**México:** Ciudad de Mexico, Monterrey, Guadalajara

**Canadá:** Toronto, Vancouver

**Estados Unidos:** Atlanta, Boston, Dallas, Houston, Kansas City, Los Angeles, Miami, Nueva York, Filadelfia, San Francisco, Seattle

## Aprendizajes clave

- Ingeniería de datos con APIs externas (Open-Meteo, geopy)
- Paralelismo con ThreadPoolExecutor para peticiones a APIs
- Feature engineering propio (similitud ELO, forma, goles esperados)
- Sistema de predicción en cascada con múltiples modelos
- Comparación de modelos de regresión y clasificación
- Serialización de modelos con joblib