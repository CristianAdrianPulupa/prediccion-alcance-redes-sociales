"""Prueba de que el artefacto entrenado admite el CSV real proporcionado."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from validacion import redondear_predicciones, validar_predictoras


def test_modelo_guardado_y_comparacion():
    paquete = joblib.load(RAIZ / "modelo_publicaciones.joblib")
    informe = json.loads((RAIZ / "resultados_modelos.json").read_text(encoding="utf-8"))
    tabla = pd.read_csv(RAIZ / "datos" / "T01_test_set.csv")
    x = validar_predictoras(tabla, paquete["columnas"])
    preds = redondear_predicciones(paquete["modelo"].predict(x))
    assert len(preds) == len(tabla) == 2918
    assert (preds >= 0).all() and np.isfinite(preds).all()
    ganador = min(informe["comparacion"], key=lambda r: r["RMSE de validación"])
    assert ganador["Modelo"] == informe["modelo_seleccionado"]
    assert informe["modelo_id"] == "polinomico"
    assert informe["grado"] == 4
