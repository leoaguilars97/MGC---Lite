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
    # cnnTab1 = Frame(cnnTabControl)
    # cnnTab2 = Frame(cnnTabControl)
    # cnnTab3 = Frame(cnnTabControl)
    cnnTab4 = Frame(cnnTabControl)
    cnnTab5 = Frame(cnnTabControl)
    cnnTab6 = Frame(cnnTabControl)
    cnnTab7 = Frame(cnnTabControl)

    # cnnTabControl.add(cnnTab1, text='Precisión por entrenamiento') - removidos del alcance de la entrega
    # cnnTabControl.add(cnnTab2, text='Precisión por época') - removidos del alcance de la entrega 
    # cnnTabControl.add(cnnTab3, text='Matriz de confusión') - removidos del alcance de la entrega
    cnnTabControl.add(cnnTab4, text='Confiabilidad')
    cnnTabControl.add(cnnTab5, text='Confiabilidad por género musical')
    cnnTabControl.add(cnnTab6, text='Performance')
    cnnTabControl.add(cnnTab7, text='Performance por género musical')

    cnnTabControl.pack(expand=1, fill="both", side=TOP)

    # cnnLblExactitud = tkinter.Label(cnnTab1, image=None)
    # cnnLblEpocas= tkinter.Label(cnnTab2, image=None)
    # cnnLblMatriz = tkinter.Label(cnnTab3, image=None)
    cnnLblConfiabilidad = tkinter.Label(cnnTab4, image=None)
    cnnLblConfiabilidad2 = tkinter.Label(cnnTab5, image=None)
    cnnLblPerformance = tkinter.Label(cnnTab6, image=None)
    cnnLblPerformance2 = tkinter.Label(cnnTab7, image=None)

    # mostrar_imagen('./img/cnn/accuracy_epoch.png', cnnLblExactitud, 400, 400) - removidos del alcance de la entrega
    # mostrar_imagen('./img/cnn/pe_epoch.png', cnnLblEpocas, 400, 400) - removidos del alcance de la entrega
    # mostrar_imagen('./img/cnn/conf_mat.png', cnnLblMatriz, 400, 400) - removidos del alcance de la entrega
    mostrar_imagen('./img/cnn/confiabilidad.png', cnnLblConfiabilidad, 400, 400)
    mostrar_imagen('./img/cnn/confiabilidad_por_genero.png', cnnLblConfiabilidad2, 400, 400)
    mostrar_imagen('./img/cnn/performance.png', cnnLblPerformance, 400, 400)
    mostrar_imagen('./img/cnn/performance_por_genero.png', cnnLblPerformance2, 400, 400)

    # cnnLblExactitud.place(relx=0.5, rely=0.5, anchor=CENTER)
    # cnnLblEpocas.place(relx=0.5, rely=0.5, anchor=CENTER)
    # cnnLblMatriz.place(relx=0.5, rely=0.5, anchor=CENTER)
    cnnLblConfiabilidad.place(relx=0.5, rely=0.5, anchor=CENTER)
    cnnLblConfiabilidad2.place(relx=0.5, rely=0.5, anchor=CENTER)
    cnnLblPerformance.place(relx=0.5, rely=0.5, anchor=CENTER)
    cnnLblPerformance2.place(relx=0.5, rely=0.5, anchor=CENTER)

    # REPORTES DE MLP
    mlpTabControl = Notebook(tab2)
    # mlpTab1 = Frame(mlpTabControl)
    # mlpTab2 = Frame(mlpTabControl)
    # mlpTab3 = Frame(mlpTabControl)
    mlpTab4 = Frame(mlpTabControl)
    mlpTab5 = Frame(mlpTabControl)
    mlpTab6 = Frame(mlpTabControl)
    mlpTab7 = Frame(mlpTabControl)

    mlpTabControl.add(mlpTab4, text='Confiabilidad')
    mlpTabControl.add(mlpTab5, text='Confiabilidad por género musical')
    mlpTabControl.add(mlpTab6, text='Performance')
    mlpTabControl.add(mlpTab7, text='Performance por género musical')
    # mlpTabControl.add(mlpTab1, text='Precisión por entrenamiento') - removidos del alcance de la entrega
    # mlpTabControl.add(mlpTab2, text='Precisión por época') - removidos del alcance de la entrega
    # mlpTabControl.add(mlpTab3, text='Matriz de confusión') - removidos del alcance de la entrega

    mlpTabControl.pack(expand=1, fill="both", side=TOP)

    # mlpLblExactitud = tkinter.Label(mlpTab1, image=None)
    # mlpLblEpocas= tkinter.Label(mlpTab2, image=None)
    # mlpLblMatriz = tkinter.Label(mlpTab3, image=None)
    mlpLblConfiabilidad = tkinter.Label(mlpTab4, image=None)
    mlpLblConfiabilidad2 = tkinter.Label(mlpTab5, image=None)
    mlpLblPerformance = tkinter.Label(mlpTab6, image=None)
    mlpLblPerformance2 = tkinter.Label(mlpTab7, image=None)

    mostrar_imagen('./img/mlp/confiabilidad.png', mlpLblConfiabilidad, 400, 400)
    mostrar_imagen('./img/mlp/confiabilidad_por_genero.png', mlpLblConfiabilidad2, 400, 400)
    mostrar_imagen('./img/mlp/performance.png', mlpLblPerformance, 400, 400)
    mostrar_imagen('./img/mlp/performance_por_genero.png', mlpLblPerformance2, 400, 400)
    # mostrar_imagen('./img/mlp/accuracy_epoch.png', mlpLblExactitud, 400, 400) - removidos del alcance de la entrega
    # mostrar_imagen('./img/mlp/pe_epoch.png', mlpLblEpocas, 400, 400) - removidos del alcance de la entrega 
    # mostrar_imagen('./img/mlp/conf_mat.png', mlpLblMatriz, 400, 400) - removidos del alcance de la entrega

    # mlpLblExactitud.place(relx=0.5, rely=0.5, anchor=CENTER)
    # mlpLblEpocas.place(relx=0.5, rely=0.5, anchor=CENTER)
    # mlpLblMatriz.place(relx=0.5, rely=0.5, anchor=CENTER)
    mlpLblConfiabilidad.place(relx=0.5, rely=0.5, anchor=CENTER)
    mlpLblConfiabilidad2.place(relx=0.5, rely=0.5, anchor=CENTER)
    mlpLblPerformance.place(relx=0.5, rely=0.5, anchor=CENTER)
    mlpLblPerformance2.place(relx=0.5, rely=0.5, anchor=CENTER)

    return ventana_reportes
