# Auditoría del caso UCE — Predicción del Alcance de Publicaciones

Se revisaron las dos páginas del enunciado proporcionado por el estudiante contra el código modularizado y los resultados reproducidos del proyecto.

## Lista de comprobación de los 8 apartados

1. **Carga de CSV:** entren/val/test: 9.336 / 2.334 / 2.918 filas y 19 columnas.
2. **EDA:** número de filas, tipos, ausencia de valores nulos, distribución de `target` y conteos por `weekday`. Se codifica `weekday` usando `OneHotEncoder`.
3. **Escalamiento:** `StandardScaler` ajustado con entrenamiento y aplicado al resto dentro del `Pipeline`.
4. **Baseline:** regresión lineal con todas las 18 predictoras originales. RMSE y MAE de entrenamiento, validación y prueba ahora se muestran también en Streamlit.
5. **Dos alternativas:** ingeniería de atributos (`balance_sentimiento`, `titulo_subjetividad`) y polinomios de grados 1–7. Se comparan por error de validación. Como fortalecimiento opcional, se pueden evaluar más combinaciones de atributos derivados.
6. **Gráfica por grado:** curva de entrenamiento vs validación, subajuste relativo del grado 1, óptimo grado 4 y señales de sobreajuste en grados 5–7. Ahora hay interpretación escrita.
7. **Opcional avanzado:** Random Forest de 200 árboles, profundidad máxima 12, hojas de mínimo 5 registros.
8. **Selección, holdout y residuos:** selección del menor RMSE de validación y evaluación final en prueba con MAE, RMSE, R² y dispersión de residuos. Ahora hay interpretación escrita de los residuos.

## Resultados que verificamos

| Modelo | RMSE validación |
|---|---:|
| Regresión lineal | 943,58 |
| Variables derivadas | 943,67 |
| Regresión polinómica grado 4 | **941,98** |
| Random Forest | 947,10 |

Final en TEST para el polinómico grado 4: **MAE 726,85, RMSE 951,24 y R² 0,0643**. Esta capacidad explicativa es limitada y debe admitirse en la defensa.

El enunciado UCE de estas dos fotografías **no exige explícitamente PDF, interfaz, ni exportación a Excel**. La aplicación Streamlit es un valor agregado. El documento de 2024 adjunto es material de referencia y no altera las instrucciones del examen actual.

**Importante:** las pruebas automáticas comprueban escenarios definidos de validación y de integración, pero no garantizan la calidad predictiva ni reemplazan la revisión del tribunal.
