# MGC

pydub
ffmpeg -i input.mp4 output.avi

# Comandos para instalar

```bash
# *** Asumiendo que ya se tiene instalado python 3.6 y pip ***

# --- Instalar y ejecutar el sistema ---
$ pip install virtualenv

# Crear un entorno virtual
$ virtualenv -p python3.6 venv

# Activar el entorno virtual
$ source venv/bin/activate

# Instalar las dependencias para Linux
$ pip install -r requirements.txt

# Instalar las dependencias para Windows
$ pip install -r requirements_windows.txt

# Ejecutar el programa
$ python main.py

# --- Limpiar la base de datos original ---
$ python limpieza_datos.py

# --- Entrenar a los modelos ---

# Entrenar el modelo de la red neuronal convolucional
$ python cnn.py

# Entrenar el modelo de la red neuronal perceptron multicapa
$ python mlp.py

# --- Ejecutar la prueba de confiabilidad y desempeño ---
$ python prueba_confiabilidad.py

# --- Compilar el programa para Windows ---
$ pip install pyinstaller
$ pyinstaller --windowed main.py --additional-hooks=extrahooks

```