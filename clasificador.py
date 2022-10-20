import os
import numpy as np
from keras.models import model_from_json
from procesamiento_audio import espectrograma_mel


def cargar_cnn_json(path="./modelos/cnn.json"):
    print(f">> Abriendo archivo json {path}")
    with open(path) as jf:
        modelo_cnn_json = jf.read()
        return model_from_json(modelo_cnn_json)

def cargar_mlp_json(path="./modelos/mlp.json"):
  print(f">> Abriendo archivo json {path}")
  with open(path) as jf:
    modelo_mlp_json = jf.read()
    return model_from_json(modelo_mlp_json)

def cargar_cnn_h5(model, path="./modelos/cnn.h5"):
    print(f">> Abriendo archivo h5 {path}")
    model.load_weights(path)

def cargar_mlp_h5(model, path="./modelos/mlp.h5"):
    print(f">> Abriendo archivo h5 {path}")
    model.load_weights(path)

def cargar_cnn():
    print(">> Cargando CNN")
    modelo = cargar_cnn_json()
    cargar_cnn_h5(modelo)
    return modelo

def cargar_mlp():
    print(">> Cargando MLP")
    modelo = cargar_mlp_json()
    cargar_mlp_h5(modelo)
    return modelo

cnn = cargar_cnn()
mlp = cargar_mlp()

cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(">> CNN Cargada y lista para utilizar")

mlp.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(">> MLP Cargada y lista para utilizar")

def obtener_espectrograma(path):
    espectrograma = espectrograma_mel(path)
    espectrograma /= espectrograma.min()
    return espectrograma.reshape(espectrograma.shape[0], 128, 657, 1)

def obtener_una_prediccion(path, opcion):
    espectrograma= obtener_espectrograma(path)
    if opcion == "cnn":
        predicciones = cnn.predict(espectrograma, verbose=1)
        return predicciones
    if opcion == "mlp":
        predicciones = mlp.predict(espectrograma, verbose=1)
        return predicciones

def predecir(path, opcion="cnn", iteraciones=3):
  predicciones = []
  for _ in range(iteraciones):
    prediccion = obtener_una_prediccion(path, opcion)[0]
    predicciones.append(prediccion)
  return [np.average(predicciones, axis=0)]