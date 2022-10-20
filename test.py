
from clasificador import predecir
import numpy as np
import os


def iterar_db():
  directorios = ["blues", "classical", "country", "rock", "jazz"]
  resultados = []
  for directorio in directorios:
    actual = f"./db/{directorio}"

    for file in os.scandir(actual):
      print('>> Analizando {file}')

      labels = ["Blues", "Clásica", "Country", "Rock", "Jazz"]

      mlp = predecir(file, opcion="mlp")[0]
      cnn = predecir(file, opcion="cnn")[0]

      index_mlp = np.argmax(mlp)
      index_cnn = np.argmax(cnn)

      resultado = {
        'file': file,
        'mlp': {
          'name': labels[index_mlp],
          'value': int(mlp[index_mlp] * 100)
        },
        'cnn': {
          'name': labels[index_cnn],
          'value': int(cnn[index_cnn] * 100)
        }
      }
      print(resultado)
      resultados.append(resultado)
  print('Resultados::: ')
  print(resultados)

# xiterar_db()

def test_individual(file):
    labels = ["Blues", "Clásica", "Country", "Rock", "Jazz"]
    mlp = predecir(file, opcion="mlp")[0]
    cnn = predecir(file, opcion="cnn")[0]
    index_mlp = np.argmax(mlp)
    index_cnn = np.argmax(cnn)
    resultado = {
      'file': file,
      'mlp': {
        'name': labels[index_mlp],
        'value': int(mlp[index_mlp] * 100)
      },
      'cnn': {
        'name': labels[index_cnn],
        'value': int(cnn[index_cnn] * 100)
      }
    }
    print(resultado)

test_individual('/Users/luisaguilar/Downloads/spanishromance.wav')
test_individual('/Users/luisaguilar/Downloads/Guns N\' Roses - Don\'t Cry.wav')