"""Interfaz Streamlit del examen complexivo.

Tres pestañas: resultados, predicciones CSV y predicción individual.
Usa el MISMO pipeline preentrenado; no reentrena durante las predicciones.
"""

import hashlib
import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from validacion import ErrorDatos, redondear_predicciones, validar_predictoras
from exportacion import crear_excel_predicciones

CARPETA = Path(__file__).resolve().parent

ETIQUETAS = {
    "n_tokens_title": "Número de palabras del título",
    "average_token_length": "Longitud promedio de palabras",
    "num_keywords": "Número de palabras clave",
    "kw_avg_max": "Alcance histórico de la mejor palabra clave",
    "global_subjectivity": "Subjetividad del texto",
    "global_sentiment_polarity": "Sentimiento global del texto",
    "global_rate_positive_words": "Proporción global de palabras positivas",
    "global_rate_negative_words": "Proporción global de palabras negativas",
    "rate_positive_words": "Tasa de palabras positivas no neutras",
    "rate_negative_words": "Tasa de palabras negativas no neutras",
    "avg_positive_polarity": "Polaridad positiva promedio",
    "avg_negative_polarity": "Polaridad negativa promedio",
    "log_n_tokens_content": "Logaritmo de palabras del contenido",
    "log_num_hrefs": "Logaritmo de enlaces",
    "root2_num_self_hrefs": "Raíz cuadrada de enlaces internos",
    "log_self_reference_max_shares": "Logaritmo del máximo de compartidos históricos",
    "log_self_reference_avg_sharess": "Logaritmo del promedio de compartidos históricos",
    "weekday": "Día de publicación",
}

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado",
    "Sunday": "Domingo",
}


def numero_espanol(valor: float, decimales=2) -> str:
    """Ej.: 941.98 -> 941,98; 1689 -> 1.689."""
    return (f"{valor:,.{decimales}f}".replace(",", "#")
            .replace(".", ",").replace("#", "."))


@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo(ruta: str, fecha_modificacion: int):
    """Solo se cargan modelos propios/confiables: joblib usa pickle."""
    return joblib.load(ruta)


def predecir(datos: pd.DataFrame, columnas, modelo):
    x = validar_predictoras(datos, columnas)
    pred = modelo.predict(x)
    return redondear_predicciones(pred)


def main():
    st.set_page_config(
        page_title="Predicción de Publicaciones", page_icon="📊", layout="wide"
    )

    ruta_modelo = CARPETA / "modelo_publicaciones.joblib"
    ruta_reporte = CARPETA / "resultados_modelos.json"
    if not ruta_modelo.is_file() or not ruta_reporte.is_file():
        st.error(
            "Falta el modelo o el archivo de resultados. "
            "Primero ejecuta `python main.py` desde la carpeta del proyecto."
        )
        st.stop()

    try:
        paquete = cargar_modelo(str(ruta_modelo), ruta_modelo.stat().st_mtime_ns)
        modelo = paquete["modelo"]
        columnas = paquete["columnas"]
        resultados = json.loads(ruta_reporte.read_text(encoding="utf-8"))
    except (KeyError, ValueError, OSError, EOFError) as exc:
        st.error(f"No se pudo cargar el modelo o sus resultados: {exc}")
        st.stop()

    st.title("📊 Predicción de Alcance en Redes Sociales")
    st.write(
        "Sistema de minería de datos para estimar el número de veces "
        "que una publicación será compartida."
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Resultados del modelo",
        "📂 Predicciones CSV",
        "🔮 Predicción individual",
    ])

    # 1. MÉTRICAS Y GRÁFICAS
    with tab1:
        st.header("Evaluación del modelo")
        m = resultados["metricas"]
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE de validación", numero_espanol(m["rmse_validacion"]))
        c2.metric("MAE de prueba", numero_espanol(m["mae_prueba"]))
        c3.metric("R² de prueba", numero_espanol(m["r2_prueba"], 4))

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

    # 2. ARCHIVOS CSV
    with tab2:
        st.header("Predecir publicaciones")
        st.write(
            "Carga un CSV con las características de las publicaciones. "
            "Puedes usar T01_test_set.csv para hacer una demostración."
        )
        archivo = st.file_uploader("Seleccionar archivo CSV", type=["csv"])

        if archivo is None:
            st.session_state.pop("resultado_csv", None)
            st.session_state.pop("huella_csv", None)
        else:
            contenido = archivo.getvalue()
            huella = hashlib.sha256(contenido).hexdigest()
            if st.session_state.get("huella_csv") != huella:
                st.session_state["huella_csv"] = huella
                st.session_state.pop("resultado_csv", None)

            try:
                datos = pd.read_csv(BytesIO(contenido))
                st.subheader("Datos cargados")
                st.caption(f"Publicaciones: {len(datos):,} | Columnas: {len(datos.columns)}")
                st.dataframe(datos.head(10), width="stretch")
                x_validado = validar_predictoras(datos, columnas)
            except (pd.errors.ParserError, UnicodeError, ValueError, ErrorDatos) as exc:
                st.error(f"No se puede utilizar este CSV: {exc}")
            else:
                if st.button("Generar predicciones", type="primary"):
                    try:
                        predicciones = redondear_predicciones(
                            modelo.predict(x_validado)
                        )
                        tabla = datos.copy()
                        tabla["prediccion_compartidos"] = predicciones
                        st.session_state["resultado_csv"] = tabla
                    except (ValueError, OverflowError, ErrorDatos) as exc:
                        st.error(f"Error al predecir: {exc}")

                tabla = st.session_state.get("resultado_csv")
                if tabla is not None:
                    st.success("Predicciones generadas correctamente.")
                    st.subheader("Resultados")
                    primeras = ["prediccion_compartidos"]
                    if "target" in tabla.columns:
                        primeras.append("target")
                    otras = [c for c in tabla.columns if c not in primeras]
                    st.dataframe(tabla[primeras + otras].head(30), width="stretch")
                    col_csv, col_excel = st.columns(2)
                    with col_csv:
                        st.download_button(
                            "⬇️ Descargar CSV (.csv)",
                            data=tabla.to_csv(index=False).encode("utf-8-sig"),
                            file_name="predicciones.csv",
                            mime="text/csv",
                        )
                    with col_excel:
                        st.download_button(
                            "⬇️ Descargar Excel (.xlsx)",
                            data=crear_excel_predicciones(tabla),
                            file_name="predicciones.xlsx",
                            mime=("application/vnd.openxmlformats-officedocument."
                                  "spreadsheetml.sheet"),
                        )

    # 3. FORMULARIO INDIVIDUAL
    with tab3:
        st.header("Predicción individual")
        st.write(
            "Introduce las 18 características de una publicación para "
            "estimar el número de compartidos."
        )
        st.info(
            "Los valores iniciales proceden de las medianas de entrenamiento. "
            "Los campos logarítmicos son atributos ya calculados, no cantidades originales."
        )
        valores = {}
        valores_iniciales = paquete["valores_iniciales"]
        enteras = set(paquete["columnas_enteras"])

        with st.form("formulario_individual"):
            izq, der = st.columns(2)
            for i, col in enumerate(columnas):
                with (izq if i % 2 == 0 else der):
                    if col == "weekday":
                        valores[col] = st.selectbox(
                            "Día de publicación", options=list(DIAS_ES),
                            format_func=lambda dia: DIAS_ES[dia],
                        )
                    elif col in enteras:
                        valores[col] = st.number_input(
                            ETIQUETAS.get(col, col),
                            min_value=0, value=int(valores_iniciales[col]),
                            step=1, key=f"entrada_{col}",
                        )
                    else:
                        valores[col] = st.number_input(
                            ETIQUETAS.get(col, col),
                            value=float(valores_iniciales[col]),
                            format="%.4f", key=f"entrada_{col}",
                        )
            calcular = st.form_submit_button(
                "Predecir número de compartidos", type="primary"
            )

        if calcular:
            try:
                nueva = pd.DataFrame([valores], columns=columnas)
                numero = int(predecir(nueva, columnas, modelo)[0])
                st.success("Predicción generada correctamente.")
                st.metric("Número estimado de compartidos", numero_espanol(numero, 0))
                st.caption(
                    "Estimación aproximada: no garantiza el alcance real. "
                    "El modelo explica una parte limitada de la variación en los datos."
                )
            except (ValueError, OverflowError, ErrorDatos) as exc:
                st.error(f"Revisa los valores ingresados: {exc}")


if __name__ == "__main__":
    main()
