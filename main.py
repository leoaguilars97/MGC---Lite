from tkinter import *
from tkinter.ttk import *

from tkinter import filedialog
from PIL import ImageTk, Image
from pyparsing import col

from gui_reportes import crear_ventana_reportes
from reportes import *
from clasificador import predecir

def imagen_por_defecto(): 
  img = Image.open('./img/default_img.jpeg')
  newimg = ImageTk.PhotoImage(img)
  return newimg
  # label.configure(image=newimg)
  # label.image = newimg

def mostrar_imagen(path, label, w, h):
    img = Image.open(path)
    # img = img.resize((w, h), Image.ANTIALIAS)
    newimg = ImageTk.PhotoImage(img)
    label.configure(image=newimg)
    label.image = newimg


def diagrama_frecuencia(y):
    _, path = diagrama_senal(y)
    mostrar_imagen(path, label_ds, 600, 50)


def diagrama_es(y, sr):
    _, path = diagrama_espectrograma(y, sr)
    mostrar_imagen(path, label_es, 300, 250)


def diagrama_em(y, sr):
    _, path = diagrama_espectrograma_mel(y, sr)
    mostrar_imagen(path, label_em, 300, 250)


def diagrama_mfc(y, sr):
    _, path = diagrama_mfccs(y, sr)
    mostrar_imagen(path, label_mf, 300, 250)


def diagrama_pred(path, type):
    prediccion = predecir(path, type)
    print(f'>> Prediccion:::: {prediccion}')
    _, path = diagrama_predicciones(prediccion)
    print(prediccion)
    mostrar_imagen(path, label_pred, 500, 400)


def analizar_cancion(path, type):
    y, sr = cargar_cancion(path)
    if y is None:
        return

    diagrama_frecuencia(y)
    diagrama_es(y, sr)
    diagrama_em(y, sr)
    diagrama_mfc(y, sr)
    diagrama_pred(path, type)


def seleccionar_cancion(type):
    filetypes = (("archivos wav", "*.wav"), ("archivos mp3", "*.mp3"))
    filename = filedialog.askopenfilename(title="Selecciona tu archivo", filetypes=filetypes)

    if filename == "":
        return

    root.title(filename)
    txt_direccion_archivo_update(filename)
    analizar_cancion(filename, type)

def analizar_cnn():
    seleccionar_cancion('cnn')

def analizar_mlp():
    seleccionar_cancion('mlp')


def txt_direccion_archivo_update(text):
    direccion_archivo.delete(1.0, "end")
    direccion_archivo.insert(1.0, text)


def abrir_reportes():
    crear_ventana_reportes(root)
    return


def quit():
    root.quit()
    root.destroy()

root = Tk()

topButtons = Frame(root)
topButtons.pack(side=TOP)

tabControl = Notebook(root)

tab1 = Frame(tabControl)
tab2 = Frame(tabControl)
tab3 = Frame(tabControl)
tab4 = Frame(tabControl)

tabControl.add(tab1, text='Predicción')
tabControl.add(tab2, text='Frecuencia')
tabControl.add(tab3, text='Frecuencia de Mel')
tabControl.add(tab4, text='MFCCs')

label_ds = Label(root, image=None)
label_es = Label(tab2, image=None)
label_em = Label(tab3, image=None)
label_mf = Label(tab4, image=None)
label_pred = Label(tab1, image=None)

root.title("Clasificador Musical")

btn_analizar_cnn = Button(topButtons, text="CNN", command=analizar_cnn)
# btn_analizar_cnn.place(x=500, y=10)

btn_analizar_mlp = Button(topButtons, text="MLP", command=analizar_mlp)
# btn_analizar_mlp.place(x=550, y=10)

btn_reportes = Button(topButtons, text="Reportes", command=abrir_reportes)
# btn_reportes.place(x=700, y=10)


direccion_archivo = Text(root, height=2, width=60)
direccion_archivo.bind("<Key>", lambda e: "break")
# direccion_archivo.place(x=10, y=10)

root.protocol("WM_DELETE_WINDOW", quit)


# GRID
# Buttons

# tabControl.grid(row=1, column=1)

# btn_analizar_cnn.grid(row=0, column=0, sticky=E)
# btn_analizar_mlp.grid(row=0, column=1, sticky=E)
# btn_reportes.grid(row=0, column=2, sticky=E)

# Labels
# label_ds.grid(row=1, column=0, sticky=W, pady=2)
# label_es.grid(row=2, column=0, sticky=W, pady=2)
# label_mf.grid(row=1, column=1, sticky=W, pady=2)
# label_pred.grid(row=2, column=1, sticky=W, pady=2)

# Buttons
btn_analizar_cnn.pack(side=LEFT)
btn_analizar_mlp.pack(side=LEFT)
btn_reportes.pack(side=LEFT)

# Labels
# label_ds.pack(side=LEFT, expand=1, fill="both")
# label_es.pack(side=CENTER, expand=1, fill='both')
label_es.place(relx=0.5, rely=0.5, anchor=CENTER)
label_em.place(relx=0.5, rely=0.5, anchor=CENTER)
label_mf.place(relx=0.5, rely=0.5, anchor=CENTER)
label_pred.place(relx=0.5, rely=0.5, anchor=CENTER)

tabControl.pack(expand=1, fill="both", side=BOTTOM)

mostrar_imagen('./img/default_img.jpeg', label_es, 1, 1)
mostrar_imagen('./img/default_img.jpeg', label_em, 1, 1)
mostrar_imagen('./img/default_img.jpeg', label_mf, 1, 1)
mostrar_imagen('./img/default_img.jpeg', label_pred, 1, 1)



# analizar_cancion('/Users/luisaguilar/Downloads/spanishromance.wav', 'cnn')

root.geometry("500x500")

mainloop()
