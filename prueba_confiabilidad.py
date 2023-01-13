
from clasificador import predecir
import numpy as np
import os
import time
import pandas as pd

# Función para revisar si el resultado es correcto
def revisar_resultado(resultado, resultado_esperado):
  return 'Acierto' if resultado.lower() == resultado_esperado.lower() or (resultado == 'Clásica' and resultado_esperado == 'classical') else 'Fallo'

# Función para iterar la base de datos y obtener los resultados de los 500 archivos
# Esta función es útil para comprobar el rendimiento de la red neuronal
# Esta función no es necesaria para el funcionamiento del programa
# Un ejemplo del archivo resultante se puede encontrar en el repositorio
def iterar_db():
  directorios = ["blues", "classical", "country", "rock", "jazz"]
  generos = ["blues", "classical", "country", "rock", "jazz"]
  resultados = []
  for directorio in directorios:  # Iterar todos los directorios de la base de datos
    actual = f"./db/{directorio}"

    for archivo in os.scandir(actual): # Escanear todos los archivos en cada directorio
      print(f'>> Analizando {archivo}')

      # Predecir con la red neuronal perceptron multicapa
      vector_probabilidad_mlp = predecir(archivo, opcion="vector_probabilidad_mlp")[0] 
      # Predecir con la red neuronal convolucional
      vector_probabilidad_cnn = predecir(archivo, opcion="vector_probabilidad_cnn")[0]

      tiempo_inicio_mlp = time.time() # Tiempo de inicio de la predicción de la red neuronal perceptron multicapa
      # Obtener el índice del valor más alto (género más probable)
      indice_prediccion_mlp = np.argmax(vector_probabilidad_mlp)
      tiempo_fin_mlp = time.time() # Tiempo de fin de la predicción de la red neuronal perceptron multicapa
      
      # Obtener el índice del valor más alto (género más probable)
      indice_prediccion_cnn = np.argmax(vector_probabilidad_cnn) 
      tiempo_fin_cnn = time.time() # Tiempo de fin de la predicción de la red neuronal convolucional
      
      # Obtener el género más probable calculado con la red neuronal perceptron multicapa
      prediccion_mlp = generos[indice_prediccion_mlp]
      # Obtener el género más probable calculado con la red neuronal convolucional
      prediccion_cnn = generos[indice_prediccion_cnn] 

      indice_esperado = generos.index(directorio) # Obtener el índice del género esperado actual
      
      # Obtener el valor de la predicción del género esperado
      prediccion_mlp_esperado = vector_probabilidad_mlp[indice_esperado] 
      # Obtener el valor de la predicción del género esperado
      prediccion_cnn_esperado = vector_probabilidad_cnn[indice_esperado] 
      
      # Verificar si la predicción es correcta para la red neuronal perceptron multicapa
      estado_mlp = revisar_resultado(prediccion_mlp, directorio) 
      # Verificar si la predicción es correcta para la red neuronal convolucional
      estado_cnn = revisar_resultado(prediccion_cnn, directorio) 

      resultado = {
        # -- Datos generales --
        'nombre_archivo'          : archivo, # Nombre del archivo
        'directorio'              : directorio, # Nombre del directorio
        # -- Red neuronal perceptron multicapa --
        # Nombre del género más probable calculado con la red neuronal perceptron multicapa
        'nombre_prediccion_mlp'   : prediccion_mlp, 
        # Valor del género más probable calculado con la red neuronal perceptron multicapa
        'valor_prediccion_mlp'    : int(vector_probabilidad_mlp[indice_prediccion_mlp] * 100), 
        # Estado de la predicción, 'Acierto' si el resultado fue correcto, 'Fallo' si no
        'estado_prediccion_mlp'   : estado_mlp,
        # Valor de la predicción del género esperado
        'prediccion_esperada_mlp' : prediccion_mlp_esperado,
        # Tiempo de ejecución de la predicción de la red neuronal perceptron multicapa
        'tiempo_prediccion_mlp'   : round((tiempo_fin_mlp - tiempo_inicio_mlp) * 100000, 2),
        # -- Red neuronal convolucional --
        # Nombre del género más probable calculado con la red neuronal convolucional
        'nombre_prediccion_cnn'   : prediccion_cnn,
        # Valor del género más probable calculado con la red neuronal convolucional
        'valor_prediccion_cnn'    : int(vector_probabilidad_cnn[indice_prediccion_cnn] * 100),
        # Estado de la predicción, 'Acierto' si el resultado fue correcto, 'Fallo' si no
        'estado_prediccion_cnn'   : estado_cnn,
        # Valor de la predicción del género esperado
        'prediccion_esperada_cnn' : prediccion_cnn_esperado,
        # Tiempo de ejecución de la predicción de la red neuronal convolucional
        'tiempo_prediccion_cnn'   : round((tiempo_fin_cnn - tiempo_fin_mlp) * 100000, 2)
      }
      resultados.append(resultado) # Agregar el resultado a la lista de resultados

  df = pd.DataFrame(resultados) # Crear un dataframe con los resultados
  df.to_csv('resultados.csv', index=False) # Guardar los resultados en un archivo csv

iterar_db() # Ejecutar la prueba
