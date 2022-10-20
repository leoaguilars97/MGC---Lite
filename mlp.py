import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from tensorflow.keras.utils import to_categorical
from procesamiento_audio import espectrogramas_mel

print(">> Iniciando creacion y entrenamiento de MLP")

print(">> Obteniendo espectrogramas y etiquetas")
espectrogramas, etiquetas = espectrogramas_mel()

print(">> Recolectando datos de prueba")

entr_espect, prueba_espect, entr_etiquetas, prueba_etiquetas = train_test_split(
    espectrogramas, etiquetas, random_state=100, stratify=etiquetas, test_size=0.3
)

entr_espect /= entr_espect.min()
prueba_espect /= entr_espect.min()

# 657 es el maximo
entr_espect = entr_espect.reshape(entr_espect.shape[0], 128, 657, 1)
prueba_espect = prueba_espect.reshape(prueba_espect.shape[0], 128, 657, 1)

# 5 generos en labels
entr_etiquetas = to_categorical(entr_etiquetas, 5)
prueba_etiquetas = to_categorical(prueba_etiquetas, 5)


print(entr_espect.shape)


# Para replicar los ejemplos, hacer el random con un seed
np.random.seed(0)
tf.random.set_seed(0)

print(">> Informacion lista para el entrenamiento")

# Crear el modelo
red_neuronal = Sequential()
red_neuronal.add(Dense(128, input_shape=(128, 657, 1), activation='relu'))
red_neuronal.add(Dense(64, activation='relu'))
red_neuronal.add(Flatten())
# Ignorar 25% de los nodos resultantes para prevenir overfitting
red_neuronal.add(Dropout(0.30))
red_neuronal.add(Dense(5, activation='softmax'))

red_neuronal.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo utilizando la informacion de prueba generada anteriormente
history = red_neuronal.fit(
    entr_espect,
    entr_etiquetas,
    batch_size=16,
    validation_data=(prueba_espect, prueba_etiquetas),
    epochs=10,
)

print(">> Guardando red MLP")

# Guardando modelo
print(">> Guardando modelo en ./modelo.json")
model_json = red_neuronal.to_json()

with open("./modelos/mlp.json", "w") as jf:
    jf.write(model_json)

# Guardando los pesos del modelo para cargar posteriormente
print(">> Guardando pesos en ./modelo.h5")
red_neuronal.save_weights("./modelos/mlp.h5")

# FIN
print(">> Red Neuronal creada, entrenada y almacenada correctamente")

print(">> Creando reportes de la MLP")

print(">> Creando grafica de entrenamiento vs validacion por Epoch")
train_loss = history.history["loss"]
test_loss = history.history["val_loss"]

plt.figure(figsize=(12, 8))

plt.plot(train_loss, label="Pérdida de entrenamiento", color="blue")
plt.plot(test_loss, label="Pérdida de pruebas", color="red")

plt.title("Pruebas y entrenamiento por Epoch", fontsize=25)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Crossentropy", fontsize=18)
plt.xticks(range(1, 16), range(1, 16))

plt.legend(fontsize=18)

plt.savefig("./img/mlp/pe_epoch.png")

print(">> Creando grafica de exactitud vs validacion")
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

plt.figure(figsize=(12, 8))

plt.plot(accuracy, label="Precision de entrenamiento", color="blue")
plt.plot(val_accuracy, label="Precision de pruebas", color="red")

plt.title("Precisión por entrenamiento", fontsize=25)
plt.xlabel("Entrenamiento", fontsize=18)
plt.ylabel("Precisión", fontsize=18)
plt.xticks(range(1, 21), range(1, 21))

plt.savefig("./img/mlp/accuracy_epoch.png")

print(">> Creando matriz de confusion")

predicciones = red_neuronal.predict(prueba_espect, verbose=1)

for i in range(5):
    print(f"{i}: {sum([1 for aciertos in prueba_etiquetas if aciertos[i] == 1])}")

for i in range(5):
    print(
        f"{i}: {sum([1 for prediccion in predicciones if np.argmax(prediccion) == i])}"
    )

conf_matrix = confusion_matrix(
    np.argmax(prueba_etiquetas, 1), np.argmax(predicciones, 1)
)

confusion_df = pd.DataFrame(conf_matrix)

plt.figure(figsize=(20, 12))
sns.set(font_scale=2)
ax = sns.heatmap(confusion_df, annot=True, cmap=sns.cubehelix_palette(rot=-0.4))
ax.set(xlabel="Valores predecidos", ylabel="Valores correctos")
plt.savefig("./img/mlp/conf_mat.png")

print(">> Reportes creados correctamente")