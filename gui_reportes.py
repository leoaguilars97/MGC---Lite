from cmath import exp
import tkinter
from tkinter import *
from tkinter.ttk import Notebook

from PIL import ImageTk, Image

accuracy_epoch = "./img/cnn/accuracy_epoch.png"
conf_mat = "./img/cnn/conf_mat.png"
pe_epoch = "./img/cnn/pe_epoch.png"


def mostrar_imagen(path, label, w, h):
    img = Image.open(path)
    img = img.resize((w, h), Image.ANTIALIAS)
    newimg = ImageTk.PhotoImage(img)
    label.configure(image=newimg)
    label.image = newimg


def mostrar_reportes_cnn():
  mostrar_imagen("./img/cnn/accuracy_epoch.png")

def mostrar_reportes_mlp():
  mostrar_imagen()

def crear_ventana_reportes(root):
    ventana_reportes = Toplevel(root)
    ventana_reportes.title("Ventana de reportes")
    ventana_reportes.geometry("500x500")

    # TAB PRINCIPAL, CONTROL DE VISTA CNN O MLP
    tabControl = Notebook(ventana_reportes)
    tab1 = Frame(tabControl)
    tab2 = Frame(tabControl)

    tabControl.add(tab1, text='CNN')
    tabControl.add(tab2, text='MLP')

    tabControl.pack(expand=1, fill="both", side=TOP)

    # REPORTES DE CNN
    cnnTabControl = Notebook(tab1)
    cnnTab1 = Frame(cnnTabControl)
    cnnTab2 = Frame(cnnTabControl)
    cnnTab3 = Frame(cnnTabControl)

    cnnTabControl.add(cnnTab1, text='Precisión por entrenamiento')
    cnnTabControl.add(cnnTab2, text='Precisión por época')
    cnnTabControl.add(cnnTab3, text='Matriz de confusión')

    cnnTabControl.pack(expand=1, fill="both", side=TOP)

    cnnLblExactitud = tkinter.Label(cnnTab1, image=None)
    cnnLblEpocas= tkinter.Label(cnnTab2, image=None)
    cnnLblMatriz = tkinter.Label(cnnTab3, image=None)

    mostrar_imagen('./img/cnn/accuracy_epoch.png', cnnLblExactitud, 400, 400)
    mostrar_imagen('./img/cnn/pe_epoch.png', cnnLblEpocas, 400, 400)
    mostrar_imagen('./img/cnn/conf_mat.png', cnnLblMatriz, 400, 400)

    cnnLblExactitud.place(relx=0.5, rely=0.5, anchor=CENTER)
    cnnLblEpocas.place(relx=0.5, rely=0.5, anchor=CENTER)
    cnnLblMatriz.place(relx=0.5, rely=0.5, anchor=CENTER)

    # REPORTES DE MLP
    mlpTabControl = Notebook(tab2)
    mlpTab1 = Frame(mlpTabControl)
    mlpTab2 = Frame(mlpTabControl)
    mlpTab3 = Frame(mlpTabControl)

    mlpTabControl.add(mlpTab1, text='Precisión por entrenamiento')
    mlpTabControl.add(mlpTab2, text='Precisión por época')
    mlpTabControl.add(mlpTab3, text='Matriz de confusión')

    mlpTabControl.pack(expand=1, fill="both", side=TOP)

    mlpLblExactitud = tkinter.Label(mlpTab1, image=None)
    mlpLblEpocas= tkinter.Label(mlpTab2, image=None)
    mlpLblMatriz = tkinter.Label(mlpTab3, image=None)

    mostrar_imagen('./img/mlp/accuracy_epoch.png', mlpLblExactitud, 400, 400)
    mostrar_imagen('./img/mlp/pe_epoch.png', mlpLblEpocas, 400, 400)
    mostrar_imagen('./img/mlp/conf_mat.png', mlpLblMatriz, 400, 400)

    mlpLblExactitud.place(relx=0.5, rely=0.5, anchor=CENTER)
    mlpLblEpocas.place(relx=0.5, rely=0.5, anchor=CENTER)
    mlpLblMatriz.place(relx=0.5, rely=0.5, anchor=CENTER)

    return ventana_reportes
