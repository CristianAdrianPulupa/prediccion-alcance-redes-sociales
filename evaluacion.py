"""Métricas y selección de modelos exclusivamente por validación."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(reales, predichos) -> dict:
    return {
        "mae": float(mean_absolute_error(reales, predichos)),
        "rmse": float(mean_squared_error(reales, predichos) ** 0.5),
        "r2": float(r2_score(reales, predichos)),
    }


def elegir_mejor(candidatos: dict) -> str:
    """candidatos: {id: {'validacion': {'rmse': ...}, ...}}"""
    if not candidatos:
        raise ValueError("No se entrenaron modelos candidatos.")
    return min(candidatos, key=lambda nombre: candidatos[nombre]["validacion"]["rmse"])
