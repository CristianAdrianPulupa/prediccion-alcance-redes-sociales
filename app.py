"""Interfaz Streamlit del examen complexivo."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

CARPETA = Path(__file__).resolve().parent

VARIABLES_PREDICTORAS = [
    "n_tokens_title", "average_token_length", "num_keywords", "kw_avg_max",
    "global_subjectivity", "global_sentiment_polarity",
    "global_rate_positive_words", "global_rate_negative_words",
    "rate_positive_words", "rate_negative_words", "avg_positive_polarity",
    "avg_negative_polarity", "log_n_tokens_content", "log_num_hrefs",
    "root2_num_self_hrefs", "log_self_reference_max_shares",
    "log_self_reference_avg_sharess", "weekday",
]


def numero_espanol(valor: float, decimales=2) -> str:
    """Ej.: 941.98 -> 941,98; 1689 -> 1.689."""
    return (f"{valor:,.{decimales}f}".replace(",", "#")
            .replace(".", ",").replace("#", "."))


def main():
    st.set_page_config(
        page_title="Predicción de Publicaciones", page_icon="📊", layout="wide"
    )

    ruta_reporte = CARPETA / "resultados_modelos.json"
    if not ruta_reporte.is_file():
        st.error(
            "Falta el archivo de resultados. "
            "Primero ejecuta `python main.py` desde la carpeta del proyecto."
        )
        st.stop()

    try:
        resultados = json.loads(ruta_reporte.read_text(encoding="utf-8"))
    except (ValueError, OSError, EOFError) as exc:
        st.error(f"No se pudieron cargar los resultados: {exc}")
        st.stop()

    st.title("📊 Predicción de Alcance en Redes Sociales")
    st.write(
        "Sistema de minería de datos para estimar el número de veces "
        "que una publicación será compartida."
    )

    st.subheader("Resumen de datos y preparación")
    conjuntos = resultados["conjuntos"]
    resumen_datos = pd.DataFrame([
        {
            "Conjunto": "Entrenamiento",
            "Registros": conjuntos["entrenamiento"],
            "Proporción": "64 %",
        },
        {
            "Conjunto": "Validación",
            "Registros": conjuntos["validacion"],
            "Proporción": "16 %",
        },
        {
            "Conjunto": "Prueba",
            "Registros": conjuntos["prueba"],
            "Proporción": "20 %",
        },
    ])
    st.dataframe(resumen_datos, hide_index=True, width="stretch")
    st.markdown(
        "**Variables predictoras (18):** "
        + ", ".join(VARIABLES_PREDICTORAS)
        + "."
    )
    st.markdown(
        "**Revisión de datos:** se verificaron los tres conjuntos y sus tipos "
        "antes del modelado; no se detectaron valores faltantes en las variables "
        "utilizadas."
    )
    st.markdown(
        "**Preprocesamiento:** las variables numéricas se estandarizaron con "
        "`StandardScaler` y la variable categórica `weekday` se transformó con "
        "`OneHotEncoder(handle_unknown=\"ignore\")`."
    )

    # 1. MÉTRICAS Y GRÁFICAS
    st.header("Evaluación del modelo")
    m = resultados["metricas"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE de prueba", numero_espanol(m["mae_prueba"]))
    c2.metric("RMSE de prueba", numero_espanol(m["rmse_prueba"]))
    c3.metric("R² de prueba", numero_espanol(m["r2_prueba"], 4))
    st.metric(
        "RMSE de validación (selección del modelo)",
        numero_espanol(m["rmse_validacion"]),
    )

    st.info(
        f'Modelo seleccionado: {resultados["modelo_seleccionado"]}. '
        "Se eligió por su menor RMSE de validación. "
        "El R² obtenido indica capacidad explicativa limitada."
    )
    st.subheader("Comparación de modelos")
    st.dataframe(
        pd.DataFrame(resultados["comparacion"]),
        hide_index=True, width="stretch",
    )

    # El enunciado solicita RMSE y MAE del baseline en los TRES conjuntos.
    st.subheader("Regresión lineal baseline: errores en los tres conjuntos")
    datos_base = resultados["detalle_modelos"]["baseline"]
    filas_base = [
        {"Conjunto": "Entrenamiento", **datos_base["entrenamiento"]},
        {"Conjunto": "Validación", **datos_base["validacion"]},
        {"Conjunto": "Prueba (holdout)", **resultados["baseline_prueba"]},
    ]
    tabla_base = pd.DataFrame(filas_base).rename(columns={
        "mae": "MAE", "rmse": "RMSE", "r2": "R²",
    })
    st.dataframe(tabla_base, hide_index=True, width="stretch")
    st.caption(
        "El baseline se entrenó exclusivamente con entrenamiento. "
        "Sus errores de prueba se presentan como evaluación, "
        "no como criterio para escoger el modelo final."
    )

    with st.expander("Resultados de entrenamiento y validación por modelo"):
        filas = []
        for candidato in resultados["detalle_modelos"].values():
            filas.append({
                "Modelo": candidato["nombre"],
                "RMSE entrenamiento": candidato["entrenamiento"]["rmse"],
                "RMSE validación": candidato["validacion"]["rmse"],
                "MAE validación": candidato["validacion"]["mae"],
                "R² validación": candidato["validacion"]["r2"],
            })
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    st.subheader("Curva de entrenamiento y validación")
    ruta_curva = CARPETA / "curva_polinomica.png"

    if ruta_curva.exists():
        izquierda, centro, derecha = st.columns([1, 4, 1])

        with centro:
            st.image(str(ruta_curva), width="stretch")
    else:
        st.warning(
            "No se encontró curva_polinomica.png. "
            "Ejecuta python main.py."
        )

    polinomios = resultados.get("polinomios", [])
    if polinomios:
        optimo = min(polinomios, key=lambda x: x["rmse_validacion"])
        st.write(
            f"**Subajuste relativo:** grado 1, una representación simple y "
            f"con mayor error de validación que el óptimo. "
            f"**Grado óptimo:** {optimo['grado']}, el menor RMSE de validación "
            f"({numero_espanol(optimo['rmse_validacion'])}). "
            "**Sobreajuste:** al aumentar el grado por encima del óptimo, "
            "el RMSE de entrenamiento disminuye, pero el de validación "
            "aumenta, especialmente en los grados 6 y 7."
        )

    st.subheader("Análisis de residuos del modelo final")
    ruta_residuos = CARPETA / "residuos_modelo_final.png"

    if ruta_residuos.exists():
        izquierda, centro, derecha = st.columns([1, 4, 1])

        with centro:
            st.image(str(ruta_residuos), width="stretch")
    else:
        st.warning(
            "No se encontró residuos_modelo_final.png. "
            "Ejecuta python main.py."
        )
    st.write(
        "Los residuos son las diferencias entre compartidos reales y predichos. "
        "En la gráfica se observan errores positivos y negativos, una dispersión "
        "amplia y bandas inclinadas vinculadas a los valores discretos de target. "
        "El R² de prueba es bajo, por lo que el modelo tiene capacidad predictiva "
        "limitada. No debemos interpretar las predicciones como valores exactos."
    )
    st.markdown(
        "**Interpretación:** los residuos positivos indican que el modelo "
        "subestimó los compartidos reales; los negativos indican una "
        "sobreestimación. Los residuos alejados de cero representan errores "
        "de mayor magnitud y muestran observaciones cuya predicción se separa "
        "más del valor observado."
    )


if __name__ == "__main__":
    main()
