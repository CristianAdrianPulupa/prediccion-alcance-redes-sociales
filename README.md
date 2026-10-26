# Complexivo UCE — Predicción de Alcance en Redes Sociales

Código reorganizado a partir de tus archivos `main.py` y `app.py`, conservando **las tres pestañas** de Streamlit.

## 1. Instalación (Windows, VS Code, PowerShell)

1. **Guarda tu proyecto actual como respaldo.** Extrae esta carpeta en otra ubicación; NO sobrescribas tu trabajo anterior antes de probarla.
2. Abre esta carpeta en Visual Studio Code.
3. Si aún no tienes un entorno virtual en esta copia, ejecuta:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   Si PowerShell no permite activar el script, abre una terminal **CMD** en VS Code y ejecuta `.venv\Scripts\activate.bat`.
4. Ejecuta el entrenamiento:

   ```powershell
   python main.py
   ```

   Al terminar, aparecen `modelo_publicaciones.joblib`, `resultados_modelos.json`, `curva_polinomica.png` y `residuos_modelo_final.png`.
5. Inicia el aplicativo:

   ```powershell
   python -m streamlit run app.py
   ```

   Abre `http://localhost:8501` si no se abre automáticamente.

## 2. ¿Para qué sirve cada archivo?

| Archivo | Función |
|---|---|
| `main.py` | Punto de entrada. Solo llama al entrenamiento. |
| `entrenamiento.py` | Lectura de CSV, EDA, cuatro modelos, selección por RMSE de validación y exportación. |
| `evaluacion.py` | MAE, RMSE, R² y selección del mejor candidato. |
| `graficos.py` | Curva de grados polinómicos y gráfico de residuos. |
| `exportacion.py` | Genera `.xlsx` real, con columnas, formatos numéricos y predicciones destacadas. |
| `validacion.py` | Validación de tipos, categorías, valores ausentes, infinitos y rangos conocidos. |
| `app.py` | Interfaz Streamlit con resultados, predicción CSV y predicción individual. |
| `datos/` | Los tres CSV originales proporcionados para el caso. |
| `pruebas/` | Pruebas automáticas de la validación. |

## 3. Decisiones técnicas

- **Train (9.336)** ajusta modelos y escaladores; **val (2.334)** elige el menor RMSE; **test (2.918)** evalúa el seleccionado y el baseline del enunciado. No se usa test para elegir modelos.
- Se comparan regresión lineal, variables derivadas, polinomios grados 1–7 y Random Forest.
- El modelo guardado incluye su preprocesamiento: **no debes volver a escalar manualmente en Streamlit**.
- La tabla de Streamlit obtiene valores del JSON generado tras entrenar; ya no contiene métricas codificadas manualmente.
- Se comprueba que el CSV tenga las 18 variables y que `weekday` sea un día de la semana; `target` es opcional y **no se usa como predictor**.
- Para el formulario individual se guardan las medianas del entrenamiento, para no cargar todo el CSV de entrenamiento cada vez que el usuario abre Streamlit.
- `joblib.load` solo debe abrir archivos propios/confiables, porque utiliza serialización de Python.

## 4. Comprobaciones y errores comunes

```powershell
python -m pytest pruebas -q
```

En la pestaña de CSV, puedes descargar **CSV** o **Excel (.xlsx)**. Abre el Excel directamente, sin usar «Texto en columnas». Si el entorno virtual ya existía, instala la nueva dependencia con `python -m pip install -r requirements.txt`.

En la pestaña de CSV, prueba con `datos/T01_test_set.csv` y con un archivo que tenga letras donde deba haber números. El primero debe funcionar y el segundo mostrar un mensaje de error descriptivo.

Si actualizas bibliotecas, ejecuta `python main.py` de nuevo para regenerar el modelo con las versiones instaladas localmente (los modelos joblib pueden no ser compatibles entre versiones diferentes de scikit-learn).

## 5. Defensa — lo esencial

- **Baseline:** regresión lineal simple con las 18 características originales.
- **Alternativa 1:** combina atributos existentes para construir variables derivadas.
- **Alternativa 2:** polinomios sobre tres características seleccionadas; grado elegido por validación.
- **Opcional:** Random Forest, que combina árboles de decisión, evaluado también en validación.
- **MAE:** media de errores absolutos; **RMSE:** penaliza más los errores grandes; **R²:** proporción de variabilidad explicada respecto al predictor medio.
- Un R² bajo significa que la precisión es limitada; **no debe afirmarse que el modelo predice exactamente**.
- Si el grado aumenta y baja el RMSE de entrenamiento pero sube el de validación, hay evidencia de **sobreajuste**.

Los números exactos se verán al ejecutar `python main.py`. Pueden diferir levemente de la ejecución original por versiones de bibliotecas o detalles numéricos, pero la selección se recalculará usando validación.

## 6. Auditoría contra el caso del examen

Consulta `AUDITORIA_REQUISITOS.md`. En Streamlit la pestaña Resultados muestra ahora la tabla de baseline para los 3 conjuntos, el análisis de subajuste/óptimo/sobreajuste y el análisis de residuos. No se alteraron los algoritmos ni las métricas guardadas.
