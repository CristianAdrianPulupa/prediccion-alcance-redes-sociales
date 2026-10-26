"""Carga de datos, EDA, preprocesamiento y entrenamiento de los cuatro modelos.

Se utiliza TRAIN para ajustar, VAL para elegir, TEST solo para informar el
resultado final (y el baseline solicitado por el enunciado). No ajusta
escaladores ni algoritmos con validación o prueba.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from evaluacion import calcular_metricas, elegir_mejor
from graficos import guardar_curva_polinomica, guardar_residuos
from validacion import validar_predictoras, validar_objetivo

CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "datos"
COLUMNAS_POLINOMICAS = [
    "log_self_reference_avg_sharess",
    "log_self_reference_max_shares",
    "num_keywords",
]
GRADOS = list(range(1, 8))


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Dos combinaciones de variables para la primera alternativa lineal."""
    datos = df.copy()
    datos["balance_sentimiento"] = (
        datos["global_rate_positive_words"] - datos["global_rate_negative_words"]
    )
    datos["titulo_subjetividad"] = (
        datos["n_tokens_title"] * datos["global_subjectivity"]
    )
    return datos


def crear_preprocesador(columnas_numericas: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numericas", StandardScaler(), columnas_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["weekday"]),
    ])


def modelo_baseline(columnas_numericas):
    return Pipeline([
        ("preprocesamiento", crear_preprocesador(columnas_numericas)),
        ("regresion", LinearRegression()),
    ])


def modelo_derivado(columnas_numericas):
    nuevas = columnas_numericas + ["balance_sentimiento", "titulo_subjetividad"]
    return Pipeline([
        ("variables", FunctionTransformer(crear_variables, validate=False)),
        ("preprocesamiento", crear_preprocesador(nuevas)),
        ("regresion", LinearRegression()),
    ])


def modelo_polinomico(grado, columnas_numericas):
    restantes = [col for col in columnas_numericas if col not in COLUMNAS_POLINOMICAS]
    preprocesador = ColumnTransformer([
        ("polinomicas", Pipeline([
            ("escalar", StandardScaler()),
            ("polinomio", PolynomialFeatures(degree=grado, include_bias=False)),
        ]), COLUMNAS_POLINOMICAS),
        ("numericas", StandardScaler(), restantes),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["weekday"]),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocesamiento", preprocesador),
        ("regresion", LinearRegression()),
    ])


def modelo_bosque(columnas_numericas):
    return Pipeline([
        ("preprocesamiento", crear_preprocesador(columnas_numericas)),
        ("regresion", RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )),
    ])


def cargar_conjuntos():
    tablas = {}
    for nombre, archivo in (
        ("entrenamiento", "T01_train_set.csv"),
        ("validacion", "T01_val_set.csv"),
        ("prueba", "T01_test_set.csv"),
    ):
        ruta = DATOS / archivo
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta}. Coloca el CSV dentro de datos/.")
        tablas[nombre] = pd.read_csv(ruta)
    columnas = [c for c in tablas["entrenamiento"].columns if c != "target"]
    x = {nombre: validar_predictoras(df, columnas) for nombre, df in tablas.items()}
    y = {nombre: validar_objetivo(df) for nombre, df in tablas.items()}
    return tablas, x, y, columnas


def ejecutar_entrenamiento():
    tablas, x, y, columnas = cargar_conjuntos()
    x_train, x_val, x_test = (x[k] for k in ("entrenamiento", "validacion", "prueba"))
    y_train, y_val, y_test = (y[k] for k in ("entrenamiento", "validacion", "prueba"))
    columnas_numericas = [c for c in columnas if c != "weekday"]

    print("=== ANÁLISIS EXPLORATORIO ===")
    for nombre, df in tablas.items():
        print(f"{nombre.capitalize()}: {df.shape[0]} filas, {df.shape[1]} columnas; "
              f"faltantes: {int(df.isna().sum().sum())}")
    print("\nTipos de las variables:")
    print(tablas["entrenamiento"].dtypes)
    print("\nPublicaciones por día:")
    print(x_train["weekday"].value_counts())
    print("\nDistribución de target en entrenamiento:")
    print(y_train.describe())

    # Candidatos: se comparan EXCLUSIVAMENTE usando RMSE de validacion.
    candidatos = {}

    base = modelo_baseline(columnas_numericas)
    base.fit(x_train, y_train)
    candidatos["baseline"] = {
        "nombre": "Regresión lineal",
        "modelo": base,
        "entrenamiento": calcular_metricas(y_train, base.predict(x_train)),
        "validacion": calcular_metricas(y_val, base.predict(x_val)),
        "grado": None,
    }

    derivado = modelo_derivado(columnas_numericas)
    derivado.fit(x_train, y_train)
    candidatos["derivadas"] = {
        "nombre": "Variables derivadas",
        "modelo": derivado,
        "entrenamiento": calcular_metricas(y_train, derivado.predict(x_train)),
        "validacion": calcular_metricas(y_val, derivado.predict(x_val)),
        "grado": None,
    }

    historial_polinomico = []
    modelos_pol = {}
    for grado in GRADOS:
        mod = modelo_polinomico(grado, columnas_numericas)
        mod.fit(x_train, y_train)
        ent = calcular_metricas(y_train, mod.predict(x_train))
        valid = calcular_metricas(y_val, mod.predict(x_val))
        historial_polinomico.append({
            "grado": grado,
            "rmse_entrenamiento": ent["rmse"],
            "rmse_validacion": valid["rmse"],
        })
        modelos_pol[grado] = (mod, ent, valid)
        print(f"Grado {grado}: RMSE entrenamiento={ent['rmse']:.2f} | "
              f"validación={valid['rmse']:.2f}")

    grado_elegido = min(GRADOS, key=lambda g: modelos_pol[g][2]["rmse"])
    mejor_pol, ent_pol, val_pol = modelos_pol[grado_elegido]
    candidatos["polinomico"] = {
        "nombre": f"Regresión polinómica (grado {grado_elegido})",
        "modelo": mejor_pol,
        "entrenamiento": ent_pol,
        "validacion": val_pol,
        "grado": grado_elegido,
    }

    rf = modelo_bosque(columnas_numericas)
    rf.fit(x_train, y_train)
    candidatos["random_forest"] = {
        "nombre": "Random Forest",
        "modelo": rf,
        "entrenamiento": calcular_metricas(y_train, rf.predict(x_train)),
        "validacion": calcular_metricas(y_val, rf.predict(x_val)),
        "grado": None,
    }

    print("\n=== COMPARACIÓN POR VALIDACIÓN ===")
    for item in candidatos.values():
        print(f"{item['nombre']}: RMSE={item['validacion']['rmse']:.2f}, "
              f"MAE={item['validacion']['mae']:.2f}")

    mejor_id = elegir_mejor(candidatos)
    elegido = candidatos[mejor_id]
    modelo_final = elegido["modelo"]
    pred_test = modelo_final.predict(x_test)
    metricas_finales = calcular_metricas(y_test, pred_test)
    metricas_baseline_test = calcular_metricas(y_test, base.predict(x_test))
    print("\n=== BASELINE: RMSE Y MAE EN CADA CONJUNTO ===")
    for nombre, medidas in (
        ("Entrenamiento", candidatos["baseline"]["entrenamiento"]),
        ("Validación", candidatos["baseline"]["validacion"]),
        ("Prueba", metricas_baseline_test),
    ):
        print(f"{nombre}: MAE={medidas['mae']:.2f} | RMSE={medidas['rmse']:.2f}")

    print("\n=== EVALUACIÓN FINAL (CONJUNTO DE PRUEBA) ===")
    print("Modelo elegido por validación:", elegido["nombre"])
    for metrica, valor in metricas_finales.items():
        print(f"{metrica.upper()}: {valor:.4f}")

    guardar_curva_polinomica(
        GRADOS,
        [fila["rmse_entrenamiento"] for fila in historial_polinomico],
        [fila["rmse_validacion"] for fila in historial_polinomico],
        CARPETA / "curva_polinomica.png",
    )
    guardar_residuos(
        pred_test, y_test.to_numpy() - pred_test,
        CARPETA / "residuos_modelo_final.png",
    )

    # Guarda el pipeline COMPLETO: la aplicación no vuelve a escalar los datos.
    # Al incluir un FunctionTransformer top-level, se necesita entrenamiento.py
    # junto a la aplicación si el modelo 'derivadas' resulta elegido.
    paquete = {
        "modelo": modelo_final,
        "columnas": columnas,
        "grado": elegido["grado"],
        "nombre_modelo": elegido["nombre"],
        "valores_iniciales": {
            col: ("Monday" if col == "weekday" else float(x_train[col].median()))
            for col in columnas
        },
        "columnas_enteras": [
            col for col in columnas_numericas
            if pd.api.types.is_integer_dtype(tablas["entrenamiento"][col])
        ],
    }
    joblib.dump(paquete, CARPETA / "modelo_publicaciones.joblib")

    comparacion = [
        {"Modelo": item["nombre"],
         "RMSE de validación": round(item["validacion"]["rmse"], 2)}
        for item in candidatos.values()
    ]
    reporte = {
        "modelo_seleccionado": elegido["nombre"],
        "modelo_id": mejor_id,
        "grado": elegido["grado"],
        "metricas": {
            "rmse_validacion": elegido["validacion"]["rmse"],
            "mae_validacion": elegido["validacion"]["mae"],
            "r2_validacion": elegido["validacion"]["r2"],
            "mae_prueba": metricas_finales["mae"],
            "rmse_prueba": metricas_finales["rmse"],
            "r2_prueba": metricas_finales["r2"],
        },
        "baseline_prueba": metricas_baseline_test,
        "comparacion": comparacion,
        "detalle_modelos": {
            nombre: {
                "nombre": item["nombre"],
                "entrenamiento": item["entrenamiento"],
                "validacion": item["validacion"],
            }
            for nombre, item in candidatos.items()
        },
        "polinomios": historial_polinomico,
        "conjuntos": {k: len(v) for k, v in tablas.items()},
        "variables_predictoras": len(columnas),
    }
    (CARPETA / "resultados_modelos.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nArchivos generados: modelo_publicaciones.joblib, "
          "resultados_modelos.json, curva_polinomica.png, residuos_modelo_final.png")
    return reporte
