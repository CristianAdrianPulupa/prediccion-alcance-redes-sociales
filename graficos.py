"""Generación de figuras para el informe y Streamlit."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def guardar_curva_polinomica(grados, rmse_entrenamiento, rmse_validacion, ruta):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grados, rmse_entrenamiento, marker="o", label="Entrenamiento")
    ax.plot(grados, rmse_validacion, marker="s", label="Validación")
    ax.set(xlabel="Grado del polinomio", ylabel="RMSE",
           title="Entrenamiento vs. validación")
    ax.set_xticks(grados)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def guardar_residuos(predicciones, residuos, ruta):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(predicciones, residuos, alpha=0.3, s=12)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(xlabel="Valores predichos", ylabel="Residuos",
           title="Análisis de residuos del modelo final")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
