
# Predicción del Alcance de Publicaciones en Redes Sociales Corporativas

## Universidad Central del Ecuador

**Facultad:** Ingeniería y Ciencias Aplicadas  
**Carrera:** Ingeniería en Sistemas de Información  
**Modalidad:** Examen Complexivo – Componente Práctico  
**Componente:** Minería de Datos  
**Subcomponente:** Métodos Predictivos

## 1. Descripción del proyecto

Este proyecto desarrolla y evalúa modelos de aprendizaje supervisado de regresión para estimar el número de veces que una publicación corporativa será compartida en redes sociales.

Se comparan diferentes estrategias de modelado predictivo mediante conjuntos independientes de entrenamiento, validación y prueba.

La solución incluye una aplicación web desarrollada con Streamlit para presentar los resultados del análisis y la evaluación de los modelos.

**Importante:** La variable objetivo `target` representa el número de veces que una publicación es compartida. No representa el número de personas alcanzadas ni las impresiones de la publicación.

## 2. Objetivo

Construir, comparar y evaluar modelos de regresión para predecir el número de compartidos de publicaciones corporativas, utilizando sus características textuales y metadatos.

## 3. Conjunto de datos

El proyecto utiliza tres archivos CSV proporcionados para el caso práctico académico.

| Conjunto | Registros | Proporción |
|---|---:|---:|
| Entrenamiento | 9.336 | 64 % |
| Validación | 2.334 | 16 % |
| Prueba | 2.918 | 20 % |
| **Total** | **14.588** | **100 %** |

Cada registro contiene 18 variables predictoras y la variable objetivo `target`.

Los archivos originales no están incluidos en el repositorio. Para reproducir el entrenamiento se requieren los conjuntos de datos obtenidos mediante una fuente autorizada.

Los archivos deben ubicarse en la carpeta `datos/`:

```text
datos/
├── T01_train_set.csv
├── T01_val_set.csv
└── T01_test_set.csv
```

## 4. Modelos implementados

Se desarrollaron y compararon las siguientes estrategias:

1. **Regresión lineal:** modelo de referencia utilizando las variables originales.
2. **Ingeniería de características:** creación de atributos derivados para ampliar la representación de los datos.
3. **Regresión polinómica:** evaluación de grados del 1 al 7 sobre tres características seleccionadas.
4. **Random Forest Regressor:** modelo avanzado basado en un conjunto de árboles de decisión.

Para seleccionar el modelo final se utilizó exclusivamente el menor RMSE obtenido en validación.

## 5. Resultados experimentales

### Comparación de modelos

| Modelo | RMSE de validación |
|---|---:|
| Regresión lineal | 943,58 |
| Variables derivadas | 943,67 |
| **Regresión polinómica (grado 4)** | **941,98** |
| Random Forest | 947,10 |

La regresión polinómica de grado 4 obtuvo el menor RMSE de validación.

### Evaluación final en prueba

| Métrica | Resultado |
|---|---:|
| MAE | 726,85 |
| RMSE | 951,24 |
| R² | 0,0643 |

El coeficiente R² indica una capacidad explicativa limitada. Por ello, las predicciones deben interpretarse como estimaciones aproximadas y no como resultados exactos.

La ventaja observada en validación fue pequeña y no se mantuvo frente a la regresión lineal en el conjunto de prueba.

## 6. Tecnologías utilizadas

- Python
- pandas y NumPy
- scikit-learn
- Matplotlib
- Streamlit
- Joblib
- pytest

## 7. Estructura del proyecto

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Punto de entrada del entrenamiento |
| `entrenamiento.py` | Preparación de datos, construcción y comparación de modelos |
| `evaluacion.py` | Cálculo de métricas y selección |
| `graficos.py` | Generación de gráficas de entrenamiento y residuos |
| `app.py` | Interfaz web mediante Streamlit |
| `requirements.txt` | Dependencias del proyecto |
| `pruebas/` | Pruebas automatizadas |
| `AUDITORIA_REQUISITOS.md` | Correspondencia con los requisitos académicos |

## 8. Instalación y ejecución

### Requisitos previos

- Python instalado
- Git
- Los tres conjuntos de datos CSV originales

### 8.1. Clonar el repositorio

```powershell
git clone https://github.com/CristianAdrianPulupa/prediccion-alcance-redes-sociales.git
cd prediccion-alcance-redes-sociales
```

### 8.2. Crear el entorno virtual

En Windows, utilizando PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell no permite activar el entorno, puede utilizarse CMD:

```cmd
.venv\Scripts\activate.bat
```

### 8.3. Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

### 8.4. Incorporar los datos

Crear la carpeta `datos/` e introducir los tres archivos CSV originales con los nombres indicados en la sección 3.

### 8.5. Entrenar y evaluar los modelos

```powershell
python main.py
```

Este procedimiento genera los archivos necesarios para el funcionamiento del aplicativo, incluyendo:

- `modelo_publicaciones.joblib`
- `resultados_modelos.json`
- `curva_polinomica.png`
- `residuos_modelo_final.png`

### 8.6. Ejecutar Streamlit

```powershell
python -m streamlit run app.py
```

La aplicación estará disponible normalmente en:

http://localhost:8501

## 9. Funcionalidades del aplicativo

### Resumen de datos

Presenta los tres conjuntos utilizados —entrenamiento, validación y prueba—, sus tamaños y las 18 variables predictoras.

### Preprocesamiento

Documenta la revisión de valores faltantes y las transformaciones aplicadas a los datos: `StandardScaler` para las variables numéricas y `OneHotEncoder` para la variable categórica `weekday`.

### Evaluación

Muestra las métricas finales del conjunto de prueba: MAE, RMSE y R². También presenta por separado el RMSE de validación utilizado para seleccionar el modelo.

### Comparación de modelos

Presenta la tabla comparativa de los modelos evaluados y sus RMSE de validación.

### Curva polinómica

Muestra la evolución de los errores de entrenamiento y validación para los grados polinómicos evaluados, permitiendo identificar el grado seleccionado y observar el comportamiento de subajuste y sobreajuste.

### Residuos

Incluye la gráfica de residuos del modelo final y una interpretación de los residuos positivos, negativos y alejados de cero.

## 10. Validación y pruebas

El proyecto incluye pruebas automatizadas para comprobar componentes del procesamiento y la validación de datos. Para ejecutar las pruebas disponibles:

```powershell
python -m pytest pruebas -q -p no:cacheprovider
```

Si pytest no se encuentra instalado:

```powershell
python -m pip install pytest
```

El comando anterior debe ejecutarse en el entorno virtual del proyecto. Las pruebas existentes deben actualizarse si todavía importan módulos retirados de la versión simplificada; por ello, esta documentación no afirma que la suite completa pase sin errores.

## 11. Limitaciones

- El modelo presenta un R² reducido y una capacidad predictiva limitada.
- Los predictores deben estar previamente calculados.
- El sistema no extrae automáticamente características desde publicaciones originales.
- La variable `target` representa compartidos, no personas alcanzadas.
- La evaluación se limita a los conjuntos proporcionados.
- No se garantiza el mismo rendimiento en campañas futuras.
- Las predicciones no deben utilizarse como única base para decisiones presupuestarias.

## 12. Reproducibilidad y consideraciones de seguridad

El preprocesamiento forma parte del Pipeline entrenado, evitando aplicar transformaciones diferentes durante las predicciones.

Los parámetros del preprocesamiento se ajustan únicamente con los datos de entrenamiento.

El conjunto de validación se utiliza para seleccionar el modelo y el conjunto de prueba para la evaluación final.

Los archivos serializados con Joblib deben cargarse únicamente desde fuentes confiables.

La reproducibilidad del entrenamiento requiere acceso autorizado a los conjuntos de datos y un entorno de Python con dependencias compatibles.

## 13. Contexto académico

Proyecto desarrollado para el componente práctico del examen complexivo de la carrera de Ingeniería en Sistemas de Información de la Universidad Central del Ecuador.

El repositorio permite consultar la implementación y la metodología aplicada para construir, comparar y evaluar modelos predictivos.
