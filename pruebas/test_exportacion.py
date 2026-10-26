"""Pruebas sencillas de exportación Excel."""

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from exportacion import crear_excel_predicciones


def test_excel_tiene_numeros_columnas_y_filtro():
    datos = pd.DataFrame({
        "n_tokens_title": [10],
        "average_token_length": [4.3477],
        "weekday": ["Monday"],
        "target": [2800],
        "prediccion_compartidos": [1302],
    })
    xlsx = crear_excel_predicciones(datos)
    with ZipFile(BytesIO(xlsx)) as archivo:
        hoja = archivo.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "autoFilter" in hoja
    assert "4.3477" in hoja
    assert "1302" in hoja
    assert "2800" in hoja


def test_excel_no_convierte_texto_en_formula():
    datos = pd.DataFrame({
        "prediccion_compartidos": [1302],
        "observaciones": ["=SUM(1,2)"],
    })
    xlsx = crear_excel_predicciones(datos)
    with ZipFile(BytesIO(xlsx)) as archivo:
        hoja = archivo.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "<f>" not in hoja and "<f " not in hoja


def test_excel_requiere_predicciones():
    with pytest.raises(ValueError, match="predicciones"):
        crear_excel_predicciones(pd.DataFrame({"target": [1]}))
