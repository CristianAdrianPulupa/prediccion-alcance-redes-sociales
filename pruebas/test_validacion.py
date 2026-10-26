import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validacion import ErrorDatos, redondear_predicciones, validar_predictoras

COLUMNAS = ["n_tokens_title", "global_subjectivity", "weekday"]


def datos_correctos():
    return pd.DataFrame({
        "n_tokens_title": [10],
        "global_subjectivity": [0.4],
        "weekday": ["Monday"],
        "target": [1200],
    })


def test_valida_y_excluye_target():
    x = validar_predictoras(datos_correctos(), COLUMNAS)
    assert list(x.columns) == COLUMNAS
    assert x.iloc[0]["n_tokens_title"] == 10


def test_rechaza_numero_invalido():
    df = datos_correctos()
    df["n_tokens_title"] = ["abc"]
    with pytest.raises(ErrorDatos, match="no numérico"):
        validar_predictoras(df, COLUMNAS)


def test_rechaza_dia_invalido():
    df = datos_correctos()
    df["weekday"] = ["Funday"]
    with pytest.raises(ErrorDatos, match="weekday"):
        validar_predictoras(df, COLUMNAS)


def test_rechaza_fuera_rango():
    df = datos_correctos()
    df["global_subjectivity"] = [3.0]
    with pytest.raises(ErrorDatos, match="global_subjectivity"):
        validar_predictoras(df, COLUMNAS)


def test_rechaza_columna_faltante():
    with pytest.raises(ErrorDatos, match="Faltan columnas"):
        validar_predictoras(datos_correctos().drop(columns="weekday"), COLUMNAS)


def test_redondeo_no_negativo():
    assert redondear_predicciones([-5, 10.7, 2.1]).tolist() == [0, 11, 2]
