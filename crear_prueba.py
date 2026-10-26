
import pandas as pd

# Leer los datos originales
datos = pd.read_csv("datos/T01_test_set.csv")

# Introducir intencionalmente un valor incorrecto
datos["n_tokens_title"] = datos["n_tokens_title"].astype(str)
datos.loc[0, "n_tokens_title"] = "texto_invalido"

# Guardar una copia independiente
datos.to_csv("prueba_invalida.csv", index=False)

print("CSV de prueba creado correctamente.")
