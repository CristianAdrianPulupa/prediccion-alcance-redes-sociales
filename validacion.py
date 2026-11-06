"""Validaciones reutilizables para entradas CSV y del formulario."""

import numpy as np
import pandas as pd

DIAS_VALIDOS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)

# Estos intervalos vienen de la definicion de las variables del caso.
INTERVALOS_CONOCIDOS = {
    "global_subjectivity": (0.0, 1.0),
    "global_sentiment_polarity": (-1.0, 1.0),
    "global_rate_positive_words": (0.0, 1.0),
    "global_rate_negative_words": (0.0, 1.0),
    "rate_positive_words": (0.0, 1.0),
    "rate_negative_words": (0.0, 1.0),
    "avg_positive_polarity": (0.0, 1.0),
    "avg_negative_polarity": (-1.0, 0.0),
}
NO_NEGATIVAS = ("n_tokens_title", "num_keywords", "kw_avg_max")


class ErrorDatos(ValueError):
    """Los datos no respetan el esquema esperado por el modelo."""


def validar_predictoras(datos: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Devuelve X limpio, ordenado y listo para el pipeline guardado.

    Se rechazan ausencias, textos en columnas numericas, infinitos, dias
    desconocidos y proporciones fuera de sus intervalos. `target` es opcional
    y se ignora al predecir (evita usar la respuesta como predictor).
    """
    if not isinstance(datos, pd.DataFrame):
        raise ErrorDatos("Se esperaba una tabla de datos (DataFrame).")
    if datos.empty:
        raise ErrorDatos("El archivo no contiene publicaciones.")
    if datos.columns.duplicated().any():
        raise ErrorDatos("El archivo contiene encabezados de columnas duplicados.")

    ausentes = [col for col in columnas if col not in datos.columns]
    if ausentes:
        raise ErrorDatos("Faltan columnas: " + ", ".join(ausentes))

    x = datos.loc[:, columnas].copy()
    numericas = [col for col in columnas if col != "weekday"]

    for col in numericas:
        serie = pd.to_numeric(x[col], errors="coerce")
        no_validos = serie.isna() | ~np.isfinite(serie.to_numpy(dtype=float))
        if no_validos.any():
            fila = int(np.flatnonzero(no_validos.to_numpy())[0]) + 2
            raise ErrorDatos(
                f"La columna '{col}' contiene un valor ausente, no numérico "
                f"o infinito (fila CSV aproximada: {fila})."
            )
        x[col] = serie.astype(float)

    if "weekday" in columnas:
        if x["weekday"].isna().any():
            raise ErrorDatos("Hay días de publicación vacíos en 'weekday'.")
        dias = x["weekday"].astype(str).str.strip().str.capitalize()
        desconocidos = sorted(set(dias) - set(DIAS_VALIDOS))
        if desconocidos:
            raise ErrorDatos(
                "Valores no válidos en 'weekday': " + ", ".join(desconocidos)
                + ". Usa días en inglés (Monday a Sunday)."
            )
        x["weekday"] = dias

    for col, (minimo, maximo) in INTERVALOS_CONOCIDOS.items():
        if col in numericas:
            fuera = (x[col] < minimo) | (x[col] > maximo)
            if fuera.any():
                raise ErrorDatos(
                    f"'{col}' debe estar entre {minimo:g} y {maximo:g}; "
                    f"se encontraron {int(fuera.sum())} valores fuera del intervalo."
                )

    for col in NO_NEGATIVAS:
        if col in numericas and (x[col] < 0).any():
            raise ErrorDatos(f"'{col}' no puede contener valores negativos.")

    return x


def validar_objetivo(datos: pd.DataFrame) -> pd.Series:
    """Obtiene target numérico y finito, solo para entrenar y evaluar."""
    if "target" not in datos.columns:
        raise ErrorDatos("El conjunto de datos no contiene la columna 'target'.")
    y = pd.to_numeric(datos["target"], errors="coerce")
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ErrorDatos("'target' contiene valores vacíos, no numéricos o infinitos.")
    if (y < 0).any():
        raise ErrorDatos("'target' contiene cantidades negativas de compartidos.")
    return y


def redondear_predicciones(predicciones):
    """Redondea predicciones finitas a enteros no negativos."""
    valores = np.asarray(predicciones, dtype=float)
    if not np.isfinite(valores).all():
        raise ErrorDatos("El modelo produjo una predicción no finita.")
    # Evita errores al convertir a int64 ante entradas numéricas extremas.
    max_seguro = 9_000_000_000_000_000_000
    if (valores > max_seguro).any() or (valores < -max_seguro).any():
        raise ErrorDatos("La predicción es demasiado grande; revisa las entradas.")
    return np.maximum(0, np.rint(valores)).astype(np.int64)
