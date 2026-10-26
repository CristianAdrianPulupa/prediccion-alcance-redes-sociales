"""Exportación de predicciones a un archivo de Excel con valores numéricos reales.

La exportación XLSX evita el asistente «Texto en columnas» de Excel: los números
no se interpretan como cadenas ni se confunden los separadores decimales.
"""

from io import BytesIO

import pandas as pd


def crear_excel_predicciones(tabla: pd.DataFrame) -> bytes:
    """Devuelve un XLSX en memoria, sin modificar la tabla de entrada."""
    if tabla.empty:
        raise ValueError("No hay publicaciones para exportar.")
    if "prediccion_compartidos" not in tabla.columns:
        raise ValueError("Primero debes generar las predicciones.")

    destacadas = ["prediccion_compartidos"]
    if "target" in tabla.columns:
        destacadas.append("target")
    restantes = [col for col in tabla.columns if col not in destacadas]
    vista = tabla[destacadas + restantes]

    archivo = BytesIO()
    # Desactivar interpretación automática de fórmulas/URLs en texto recibido
    # en los CSV que cargue un usuario.
    with pd.ExcelWriter(
        archivo,
        engine="xlsxwriter",
        engine_kwargs={"options": {
            "strings_to_formulas": False,
            "strings_to_urls": False,
        }},
    ) as escritor:
        vista.to_excel(escritor, sheet_name="Predicciones", index=False)
        libro = escritor.book
        hoja = escritor.sheets["Predicciones"]

        encabezado = libro.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#183153",
            "text_wrap": True,
            "valign": "vcenter",
            "border": 0,
        })
        entero = libro.add_format({"num_format": "#,##0"})
        decimal = libro.add_format({"num_format": "#,##0.0000"})
        estimacion = libro.add_format({
            "num_format": "#,##0", "bg_color": "#E2F3EB", "bold": True
        })

        hoja.set_row(0, 34, encabezado)
        hoja.freeze_panes(1, 0)
        hoja.autofilter(0, 0, len(vista), len(vista.columns) - 1)
        hoja.set_zoom(90)

        for posicion, columna in enumerate(vista.columns):
            ancho = max(14, min(33, len(str(columna)) + 3))
            if columna == "prediccion_compartidos":
                formato = estimacion
                ancho = 25
            elif pd.api.types.is_integer_dtype(vista[columna]):
                formato = entero
            elif pd.api.types.is_float_dtype(vista[columna]):
                formato = decimal
            else:
                formato = None
            hoja.set_column(posicion, posicion, ancho, formato)

    return archivo.getvalue()
